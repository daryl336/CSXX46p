"""
Sequential PPO → LLM Hybrid Agent

Architecture Flow:
1. PPO Model: Analyzes game state and suggests action with probabilities
2. LLM Review: Receives PPO's suggestion + action probabilities + game context
3. Final Decision: LLM makes the final action choice (accept, modify, or override)

This sequential approach allows LLM to:
- See PPO's learned policy (action probabilities)
- Validate PPO's neural network decisions
- Override when strategic considerations are more important
- Learn from PPO's deep learning expertise
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import requests
import json
from typing import Optional, Dict
import settings as s
from metrics.metrics_tracker import MetricsTracker

# Import PPO components from the base PPO agent
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ppo'))
from agent_code.ppo.callbacks import (
    PPONetwork,
    PPOAgent,
    build_feature_vector,
    build_action_mask,
    compute_danger_map,
    BOARD_SHAPE,
    FEATURE_VECTOR_SIZE,
    ACTIONS,
    ACTION_IDX,
    DANGER_THRESHOLD
)

# Import helper functions for LLM integration
from .helper import (
    check_valid_movement,
    check_bomb_radius_and_escape,
    should_plant_bomb,
    coin_collection_policy,
    get_self_pos,
    nearest_crate_action
)

BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000"
MODEL_PATH = "models/ppo_agent.pth"


# ===================================================================
# SETUP
# ===================================================================

def setup(self):
    """
    Initialize PPO model and LLM endpoint for sequential decision making.
    """
    self.name = "PPO→LLM Sequential Agent"

    # Initialize PPO agent
    self.ppo_agent = PPOAgent(
        input_dim=FEATURE_VECTOR_SIZE,
        action_dim=len(ACTIONS),
        lr=3e-4,
        batch_size=1024,
        minibatch_size=256,
        update_epochs=4,
        clip_eps=0.2,
        entropy_coef=0.04,
        target_kl=0.015,
    )

    # Try to load existing PPO model
    model_paths = [
        MODEL_PATH,                              # ppo_llm model (preferred)
        "agent_code/ppo/models/ppo_agent.pth",  # base PPO model (fallback)
        "models/ppo_agent.pth",                  # alternative location
    ]

    loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📂 Found model at: {model_path}")
            try:
                success = self.ppo_agent.load(model_path)
                if success:
                    print(f"✅ Loaded PPO model from: {model_path}")
                    loaded = True
                    break
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue

    if not loaded:
        print("⚠️  No existing model found. Starting with fresh PPO network.")

    # LLM configuration
    self.use_llm = True  # Set to False to use pure PPO
    self.llm_override_count = 0
    self.llm_accept_count = 0

    # State tracking
    self.movement_history = []
    self.ppo_suggestions = []  # Track PPO suggestions for analysis
    self.llm_decisions = []    # Track LLM final decisions

    # Metrics tracking
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.episode_counter = 0
    self.episode_active = False
    self.current_step = 0

    # Training state tracking (for PPO updates)
    self.last_obs = None
    self.last_action = None
    self.last_log_prob = None
    self.last_value = None
    self.last_action_mask = None

    print(f"🧠 PPO network initialized (input_dim={FEATURE_VECTOR_SIZE}, actions={len(ACTIONS)})")
    print(f"🤖 LLM enabled: {self.use_llm}")
    print(f"🎯 Strategy: PPO suggests → LLM decides")


# ===================================================================
# MAIN ACTION SELECTION (Sequential Flow)
# ===================================================================

def act(self, game_state) -> str:
    """
    Sequential decision making:
    1. PPO analyzes state and suggests action with probabilities
    2. LLM receives PPO suggestion + probabilities and makes final decision
    """
    round_num = game_state.get('round', 0)
    step = game_state.get('step', 0)

    # print(f"\n{'='*60}")
    # print(f"Round {round_num}, Step {step} - Sequential PPO→LLM Agent")
    # print(f"{'='*60}")

    # ===================================================================
    # PHASE 1: PPO SUGGESTION
    # ===================================================================
    ppo_action, action_probs, value_estimate, obs, action_mask = get_ppo_suggestion(self, game_state)
    # print(f"[PPO] Suggested action: {ppo_action}")
    # print(f"[PPO] Action probabilities: {format_probs(action_probs)}")
    # print(f"[PPO] Value estimate: {value_estimate:.3f}")

    # ===================================================================
    # PHASE 2: LLM FINAL DECISION
    # ===================================================================
    if self.use_llm:
        try:
            final_action = get_llm_final_decision(
                self,
                game_state,
                ppo_suggested_action=ppo_action,
                action_probabilities=action_probs,
                value_estimate=value_estimate
            )
            # print(f"[LLM] Final decision: {final_action}")

            # Track override vs acceptance
            if final_action == ppo_action:
                self.llm_accept_count += 1
                # print(f"[LLM] ✅ Accepted PPO suggestion (accept rate: {self.llm_accept_count}/{self.llm_accept_count + self.llm_override_count})")
            else:
                self.llm_override_count += 1
                # print(f"[LLM] 🔄 Overrode PPO: {ppo_action} → {final_action} (override rate: {self.llm_override_count}/{self.llm_accept_count + self.llm_override_count})")
        except Exception as e:
            print(f"[LLM] ❌ Error, falling back to PPO: {e}")
            final_action = ppo_action
    else:
        final_action = ppo_action
        # print(f"[PPO] LLM disabled, using PPO suggestion: {final_action}")

    # ===================================================================
    # PHASE 3: SAFETY VALIDATION
    # ===================================================================
    valid_actions = get_valid_actions_from_mask(action_mask)
    if final_action not in valid_actions:
        print(f"[SAFETY] ⚠️  Action {final_action} not valid, choosing from: {valid_actions}")
        final_action = valid_actions[0] if valid_actions else 'WAIT'

    # ===================================================================
    # PHASE 4: STORE FOR TRAINING (if in training mode)
    # ===================================================================
    final_action_idx = ACTION_IDX[final_action]

    # Get log prob for the FINAL action (not PPO's suggestion)
    # This is important for training with LLM's choices
    final_log_prob = np.log(action_probs[final_action_idx] + 1e-8)

    self.last_obs = obs
    self.last_action = final_action_idx
    self.last_log_prob = final_log_prob
    self.last_value = value_estimate
    self.last_action_mask = action_mask

    # ===================================================================
    # PHASE 5: METRICS TRACKING
    # ===================================================================
    track_episode_metrics(self, game_state, final_action)

    # ===================================================================
    # PHASE 6: STATE TRACKING
    # ===================================================================
    self.ppo_suggestions.append({
        'step': step,
        'action': ppo_action,
        'probs': action_probs.tolist(),
        'value': value_estimate
    })
    self.llm_decisions.append({'step': step, 'action': final_action})

    # print(f"✅ Final action: {final_action}")
    # print(f"{'='*60}\n")

    return final_action


# ===================================================================
# PPO SUGGESTION
# ===================================================================

def get_ppo_suggestion(self, game_state) -> tuple:
    """
    Get PPO's suggested action based on neural network policy.

    Returns:
        (action, action_probs, value_estimate, obs, action_mask)
    """
    # Build feature vector (same as PPO agent)
    obs = build_feature_vector(game_state)

    # Build action mask for safety
    danger_map = compute_danger_map(game_state)
    action_mask = build_action_mask(game_state, danger_map)

    # Get PPO's policy output
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.ppo_agent.device)
    mask_tensor = torch.from_numpy(action_mask).to(self.ppo_agent.device).unsqueeze(0)

    with torch.no_grad():
        logits, value = self.ppo_agent.model(obs_tensor)
        masked_logits = self.ppo_agent._apply_action_mask(logits, mask_tensor)

        # Get action probabilities
        dist = Categorical(logits=masked_logits)
        probs = dist.probs.cpu().numpy()[0]  # Action probabilities

        # Sample action
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        action = ACTIONS[int(action_idx.item())]
        value_est = float(value.item())
        log_prob_val = float(log_prob.item())

    # print(f"[PPO] Selected {action} with confidence {probs[ACTION_IDX[action]]:.2%}")

    return (action, probs, value_est, obs, action_mask)


# ===================================================================
# LLM FINAL DECISION
# ===================================================================

def get_llm_final_decision(
    self,
    game_state,
    ppo_suggested_action: str,
    action_probabilities: np.ndarray,
    value_estimate: float
) -> str:
    """
    Query LLM with PPO's suggestion and let it make the final decision.

    Args:
        game_state: Current game state
        ppo_suggested_action: PPO's recommended action
        action_probabilities: Probability distribution over all actions
        value_estimate: PPO's value estimate for current state

    Returns:
        final_action: LLM's final decision
    """
    # Extract game state components
    field = game_state.get('field', np.zeros(BOARD_SHAPE))
    self_info = game_state.get('self')
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros(BOARD_SHAPE))

    # Get tactical analysis from helper functions
    valid_movement = check_valid_movement(field, self_info, bombs)
    nearest_crate = nearest_crate_action(field, self_info, explosions)
    bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
    plant_bomb_full_data = should_plant_bomb(game_state, field, self_info, bombs, others)
    coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin=1)

    plant_bomb_data = {
        "plant": plant_bomb_full_data.get("plant"),
        "reason": plant_bomb_full_data.get("reason"),
        "current_status": plant_bomb_full_data.get("current_status"),
    }

    # Format action probabilities for LLM
    action_probs_dict = {action: float(prob) for action, prob in zip(ACTIONS, action_probabilities)}

    # Calculate confidence (entropy-based)
    entropy = -np.sum(action_probabilities * np.log(action_probabilities + 1e-8))
    max_entropy = np.log(len(ACTIONS))  # Maximum possible entropy
    confidence = 1.0 - (entropy / max_entropy)  # 1.0 = very confident, 0.0 = uniform distribution

    # Prepare payload for LLM with PPO's suggestion
    payload = {
        # Game state analysis
        "valid_movement": json.dumps(valid_movement),
        "nearest_crate": json.dumps(nearest_crate),
        "check_bomb_radius": json.dumps(bomb_radius_data),
        "plant_bomb_available": json.dumps(plant_bomb_data),
        "coins_collection_policy": json.dumps(coins_collection_data),
        "movement_history": json.dumps(self.movement_history[-5:]),

        # PPO suggestion (NEW!) - Using consistent naming with q_llm
        "rl_model_suggestion": json.dumps({
            "recommended_action": ppo_suggested_action,
            "action_probabilities": action_probs_dict,
            "value_estimate": float(value_estimate),
            "confidence": float(confidence),
            "top_3_actions": get_top_k_actions(action_probs_dict, k=3)
        })
    }

    headers = {'Content-Type': 'application/json'}
    response = requests.request("POST", BOMBERMAN_AGENT_ENDPOINT, headers=headers, json=payload, timeout=180)
    results = response.json()

    # print(f"[LLM] Received PPO suggestion: {ppo_suggested_action} (conf: {confidence:.2%})")
    # print(f"[LLM] Reasoning: {results.get('reasoning', 'N/A')}")

    llm_action = results.get('action', ppo_suggested_action)

    return llm_action


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def format_probs(probs: np.ndarray) -> str:
    """Format action probabilities for display."""
    prob_strs = [f"{action}={prob:.2%}" for action, prob in zip(ACTIONS, probs)]
    return "{" + ", ".join(prob_strs) + "}"


def get_top_k_actions(action_probs: dict, k: int = 3) -> list:
    """Get top-k actions by probability."""
    sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
    return [{"action": action, "probability": float(prob)} for action, prob in sorted_actions[:k]]


def get_valid_actions_from_mask(action_mask: np.ndarray) -> list:
    """Extract valid actions from action mask."""
    return [ACTIONS[i] for i, mask_val in enumerate(action_mask) if mask_val > 0]


# ===================================================================
# METRICS TRACKING
# ===================================================================

def track_episode_metrics(self, game_state, action):
    """Track metrics for both training and play modes."""
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")

    current_round = game_state.get('round', 0) if game_state else 0
    current_step = game_state.get('step', 0) if game_state else 0

    # Start new episode at step 1
    if current_step == 1:
        # End previous episode if one was active
        if self.episode_active and hasattr(self, 'last_game_state'):
            _end_episode_from_act(self, self.last_game_state)

        # Start new episode
        opponent_names = [o[0] for o in game_state.get('others', []) if o]
        scenario = "training" if self.train else "play"
        self.metrics_tracker.start_episode(
            episode_id=current_round,
            opponent_types=opponent_names,
            map_name="default",
            scenario=scenario
        )
        self.episode_active = True
        self.current_step = 0

    # Increment step counter
    if self.episode_active:
        self.current_step += 1

    # Store game state for potential episode end detection
    self.last_game_state = game_state


def _end_episode_from_act(self, game_state):
    """End the current episode and save metrics."""
    if not hasattr(self, 'metrics_tracker') or not self.episode_active:
        return

    # Determine outcome
    survived = True
    won = False
    rank = 4

    if game_state and 'self' in game_state:
        survived = True
        if 'others' in game_state:
            alive_others = sum(1 for o in game_state.get('others', []) if o is not None)
            if alive_others == 0:
                won = True
                rank = 1
            else:
                rank = 2

    survival_steps = self.current_step if hasattr(self, 'current_step') else 0
    total_steps = game_state.get('step', 400) if game_state else 400

    # End episode
    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=survival_steps,
        total_steps=total_steps,
        metadata={
            "mode": "sequential_ppo_llm",
            "llm_override_rate": self.llm_override_count / max(1, self.llm_accept_count + self.llm_override_count)
        }
    )

    self.metrics_tracker.save()
    self.episode_active = False


# ===================================================================
# END OF ROUND (for play mode)
# ===================================================================

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each round during play mode.
    """
    if hasattr(self, 'episode_active') and self.episode_active:
        _end_episode_from_act(self, last_game_state)
