"""
Sequential PPO → Rule-Based Hybrid Agent

Architecture Flow:
1. PPO Model: Analyzes game state and suggests action with probabilities
2. Rule-Based Review: Rule-based agent receives PPO's suggestion + game state
3. Final Decision: Rule-based agent makes the final action choice

This sequential approach allows rule-based agent to:
- See PPO's learned policy (action probabilities)
- Validate PPO's neural network decisions
- Override when hard-coded heuristics suggest a better move
- Train PPO model based on rule-based agent's proven strategies
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
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

# Import rule-based agent's act function
import agent_code.rule_based_agent.callbacks as rule_based_callbacks

MODEL_PATH = "agent_code/ppo_rule_based/models/ppo_rule_based_agent.pth"


# ===================================================================
# SETUP
# ===================================================================

def setup(self):
    """
    Initialize PPO model and rule-based decision making for sequential hybrid agent.
    """
    self.name = "PPO→Rule Sequential Agent"

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
        MODEL_PATH,                              # ppo_rule_based model (preferred)
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
        print("⚠️ No existing model found. Starting with fresh PPO network.")

    # Rule-based configuration
    self.use_rule_based = False  # Set to False to use pure PPO (for testing PPO progress)

    # Initialize rule-based agent's internal state by calling its setup
    # This ensures all required attributes (bomb_history, coordinate_history, etc.) are set
    rule_based_callbacks.setup(self)

    # Metrics tracking
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="metrics")
    self.episode_started = False

    # Tracking for analysis
    self.ppo_suggestions = []
    self.rule_decisions = []
    self.agreement_count = 0
    self.disagreement_count = 0

    # Training state tracking (for PPO updates)
    self.last_obs = None
    self.last_action = None
    self.last_log_prob = None
    self.last_value = None
    self.last_action_mask = None

    print(f"✅ {self.name} initialized!")
    print(f"   - PPO feature vector size: {FEATURE_VECTOR_SIZE}")
    print(f"   - Action space: {ACTIONS}")
    print(f"   - Rule-based enabled: {self.use_rule_based}")
    print(f"   - Sequential flow: PPO suggests → {'Rule-based decides' if self.use_rule_based else 'PPO decides (pure PPO mode)'}")


# ===================================================================
# ACTION SELECTION (SEQUENTIAL: PPO → RULE-BASED)
# ===================================================================

def act(self, game_state: dict) -> str:
    """
    Sequential decision making:
    1. PPO suggests action based on learned policy
    2. Rule-based agent makes final decision

    Returns the rule-based agent's final action.
    """
    # Start episode tracking
    if not self.episode_started and game_state.get("step", 0) == 1:
        opponent_names = [other[0] for other in game_state.get("others", []) if other]
        self.metrics_tracker.start_episode(
            episode_id=game_state.get("round", 0),
            opponent_types=opponent_names,
            scenario="training" if self.train else "evaluation"
        )
        self.episode_started = True

    # Phase 1: PPO Suggestion (get full metrics for training)
    ppo_suggested_action, action_probs, value_estimate, obs, action_mask = get_ppo_suggestion_full(self, game_state)

    # Phase 2: Final Decision (Rule-Based or Pure PPO)
    if self.use_rule_based:
        final_action = rule_based_callbacks.act(self, game_state)

        # Track agreement/disagreement
        if ppo_suggested_action == final_action:
            self.agreement_count += 1
        else:
            self.disagreement_count += 1

        self.logger.debug(f"PPO suggested: {ppo_suggested_action}, Rule-based decided: {final_action}")
    else:
        # Pure PPO mode - use PPO's suggestion directly
        final_action = ppo_suggested_action
        self.logger.debug(f"Pure PPO mode: {final_action}")

    # Phase 3: Store for training (if in training mode)
    final_action_idx = ACTION_IDX[final_action]

    # Get log prob for the FINAL action
    # If rule-based: trains PPO with rule-based agent's choices
    # If pure PPO: trains PPO with its own choices
    final_log_prob = np.log(action_probs[final_action_idx] + 1e-8)

    self.last_obs = obs
    self.last_action = final_action_idx
    self.last_log_prob = final_log_prob
    self.last_value = value_estimate
    self.last_action_mask = action_mask

    # Phase 4: Tracking for analysis
    self.ppo_suggestions.append(ppo_suggested_action)
    self.rule_decisions.append(final_action)

    return final_action


def get_ppo_suggestion_full(self, game_state: dict) -> tuple:
    """
    Get PPO model's suggested action with full training data.

    Returns:
        (suggested_action, action_probs, value_estimate, obs, action_mask)
    """
    try:
        # Build feature vector
        feature_vector = build_feature_vector(game_state)
        if feature_vector is None:
            self.logger.warning("Failed to build feature vector, defaulting to WAIT")
            default_probs = np.ones(len(ACTIONS)) / len(ACTIONS)
            default_mask = np.ones(len(ACTIONS), dtype=bool)
            return "WAIT", default_probs, 0.0, None, default_mask

        # Build action mask for safety
        action_mask = build_action_mask(game_state)

        # Get PPO model prediction
        with torch.no_grad():
            state_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)
            action_logits, value_estimate = self.ppo_agent.network(state_tensor)

            # Apply action mask (set invalid actions to very low probability)
            masked_logits = action_logits.clone()
            masked_logits[0, ~action_mask] = -1e8

            # Get action probabilities
            action_probs = torch.softmax(masked_logits, dim=1)
            action_probs_np = action_probs.squeeze(0).cpu().numpy()

            # Select action with highest probability
            action_idx = torch.argmax(action_probs, dim=1).item()
            ppo_suggested_action = ACTIONS[action_idx]

            # Calculate confidence (probability of chosen action)
            confidence = action_probs[0, action_idx].item()

        self.logger.debug(f"PPO suggestion: {ppo_suggested_action} (confidence: {confidence:.2f})")
        return ppo_suggested_action, action_probs_np, value_estimate.item(), feature_vector, action_mask

    except Exception as e:
        self.logger.error(f"Error in get_ppo_suggestion_full: {e}")
        default_probs = np.ones(len(ACTIONS)) / len(ACTIONS)
        default_mask = np.ones(len(ACTIONS), dtype=bool)
        return "WAIT", default_probs, 0.0, None, default_mask


# ===================================================================
# END OF ROUND
# ===================================================================

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each game round to finalize metrics.
    """
    import events as e

    # Determine if won
    won = False
    rank = 4
    if e.SURVIVED_ROUND in events and "others" in last_game_state:
        alive = sum(1 for o in last_game_state["others"] if o)
        won = alive == 0
        rank = 1 if won else 2

    # End episode tracking
    current_step = last_game_state.get("step", 0)
    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=current_step,
        total_steps=400
    )

    # Log agreement statistics
    total_decisions = self.agreement_count + self.disagreement_count
    if total_decisions > 0:
        agreement_rate = 100.0 * self.agreement_count / total_decisions
        self.logger.info(f"PPO-Rule agreement rate: {agreement_rate:.1f}% ({self.agreement_count}/{total_decisions})")

    # Save metrics
    self.metrics_tracker.save()

    # Reset tracking
    self.episode_started = False
    self.agreement_count = 0
    self.disagreement_count = 0
    self.ppo_suggestions = []
    self.rule_decisions = []
