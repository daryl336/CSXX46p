# agent_code/ppo_llm/train.py
"""
Training module for Sequential PPO → LLM Agent

This module handles:
1. PPO updates based on LLM's final action choices
2. Reward shaping for PPO component
3. Model persistence and metrics tracking
4. LLM acts as final decision layer (trains PPO to align with LLM's choices)
"""

import os
from typing import List

import events as e
import settings as s
from events import SURVIVED_ROUND
from metrics.metrics_tracker import MetricsTracker

# Import from callbacks
try:
    from . import callbacks as ppo_llm_callbacks
except Exception:
    import callbacks as ppo_llm_callbacks

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = "models/ppo_llm_agent.pth"

# Reward structure (same as base PPO, normalized to [-1, 1])
GAME_REWARDS = {
    "COIN_COLLECTED": 0.2,
    "KILLED_OPPONENT": 0.0,
    "CRATE_DESTROYED": 0.08,
    "BOMB_DROPPED": 0.0,
    "BOMB_EXPLODED": 0.0,
    "KILLED_SELF": -1.0,
    "GOT_KILLED": 0.0,
    "INVALID_ACTION": -0.02,
    "WAITED": 0.0,
    "SURVIVED_ROUND": 0.8,
    "MOVED_UP": 0.0,
    "MOVED_RIGHT": 0.0,
    "MOVED_DOWN": 0.0,
    "MOVED_LEFT": 0.0,
}

# Shaped rewards
ESCAPED_DANGER_REWARD = 0.3
SAFE_BOMB_REWARD = 0.4
CRATE_IN_RANGE_REWARD = 0.03
APPROACHING_COIN_REWARD = 0.05
EXPLORATION_BONUS = 0.006


def setup_training(self):
    """
    Called once before training starts.
    Ensures the PPO agent is properly initialized.
    """
    self.name = "PPO→LLM Training"
    self.logger.info("Setting up PPO→LLM training.")
    self.round_counter = getattr(self, "round_counter", 0)

    # Ensure model directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir != "":
        os.makedirs(model_dir, exist_ok=True)

    # Check if agent already initialized by callbacks.setup()
    if hasattr(self, "ppo_agent") and self.ppo_agent is not None:
        self.logger.info("Using existing ppo_agent (already initialized by callbacks.setup()).")
    else:
        # Agent not initialized - call setup() to create it
        self.logger.info("ppo_agent not found. Calling callbacks.setup() to initialize.")
        try:
            ppo_llm_callbacks.setup(self)
            if hasattr(self, "ppo_agent") and self.ppo_agent is not None:
                self.logger.info("ppo_agent successfully initialized via callbacks.setup().")
            else:
                raise RuntimeError("callbacks.setup() did not create ppo_agent")
        except Exception as exc:
            self.logger.error(f"Failed to initialize ppo_agent: {exc}")
            raise

    # Ensure metrics tracker exists
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(
            agent_name=self.name,
            save_dir="evaluation_metrics"
        )
        self.episode_counter = 0

    # Initialize exploration tracking
    if not hasattr(self, 'visited_positions'):
        self.visited_positions = set()
        self.last_coin_distance = None

    self.logger.info("setup_training completed.")


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called each time an action was performed.
    Logs transition (s, a, r, s') to PPO buffer.

    NOTE: The action here is the FINAL action (potentially modified by LLM),
    and we train PPO on this action. This helps PPO learn from LLM's decisions.
    """
    if old_game_state is None or new_game_state is None:
        return

    # Safety check
    if not hasattr(self, "ppo_agent") or self.ppo_agent is None:
        self.logger.warning("ppo_agent not available in game_events_occurred.")
        return

    if not hasattr(self, "last_obs") or self.last_obs is None:
        self.logger.debug("last_obs not available - skipping transition storage.")
        return

    # START EPISODE ON FIRST STEP
    if old_game_state:
        if old_game_state.get('step', 0) == 1:
            opponent_names = []
            if 'others' in old_game_state and old_game_state['others']:
                for other in old_game_state['others']:
                    if other is not None:
                        opponent_names.append(other[0])
            self.episode_counter = old_game_state.get('round')

            self.metrics_tracker.start_episode(
                episode_id=self.episode_counter,
                opponent_types=opponent_names,
                scenario="training"
            )

            # Reset exploration tracking
            self.visited_positions = set()
            self.last_coin_distance = None

            self.logger.debug(f"Started episode {self.episode_counter} with opponents: {opponent_names}")

    # Compute reward (same logic as base PPO)
    reward = 0.0
    for event in events:
        event_reward = GAME_REWARDS.get(event, 0)
        reward += event_reward
        self.metrics_tracker.record_event(event, reward=event_reward)

    # Import PPO callbacks for danger computation
    from agent_code.ppo import callbacks as ppo_callbacks

    # Shaped rewards (EXACT SAME as base PPO for fair comparison)
    old_danger = None
    new_danger = None
    was_in_danger = False
    if old_game_state and old_game_state.get("self"):
        old_danger = ppo_callbacks.compute_danger_map(old_game_state)
        _, _, _, (ox, oy) = old_game_state["self"]
        was_in_danger = old_danger[ox, oy] >= ppo_callbacks.DANGER_THRESHOLD

    # Safe bomb reward
    caused_damage = (
        'CRATE_DESTROYED' in events
        or 'KILLED_OPPONENT' in events
        or 'OPPONENT_ELIMINATED' in events
    )
    if 'BOMB_EXPLODED' in events and caused_damage and 'KILLED_SELF' not in events and 'GOT_KILLED' not in events:
        reward += SAFE_BOMB_REWARD
        self.metrics_tracker.record_event("SAFE_BOMB_EXPLOSION", reward=SAFE_BOMB_REWARD)

    # Escaped danger reward
    if new_game_state:
        new_danger = ppo_callbacks.compute_danger_map(new_game_state)
        if new_game_state.get("self"):
            _, _, _, (nx, ny) = new_game_state["self"]
            if was_in_danger and new_danger[nx, ny] < ppo_callbacks.DANGER_THRESHOLD:
                reward += ESCAPED_DANGER_REWARD
                self.metrics_tracker.record_event("ESCAPED_DANGER", reward=ESCAPED_DANGER_REWARD)

    # Crates in bomb range
    if self_action == 'BOMB' and old_game_state and old_game_state.get("self"):
        _, _, _, (ox, oy) = old_game_state["self"]
        from agent_code.ppo.train import count_crates_in_bomb_range
        crates_in_range = count_crates_in_bomb_range(old_game_state, (ox, oy))
        if crates_in_range > 0:
            crate_reward = CRATE_IN_RANGE_REWARD * crates_in_range
            reward += crate_reward
            self.metrics_tracker.record_event("CRATES_IN_BOMB_RANGE", reward=crate_reward)

    # Approaching coins
    if old_game_state and new_game_state:
        if old_game_state.get("self") and new_game_state.get("self"):
            _, _, _, old_pos = old_game_state["self"]
            _, _, _, new_pos = new_game_state["self"]
            coins = new_game_state.get("coins", [])
            if coins:
                from agent_code.ppo.train import is_approaching_target
                if is_approaching_target(old_pos, new_pos, coins):
                    reward += APPROACHING_COIN_REWARD
                    self.metrics_tracker.record_event("APPROACHING_COIN", reward=APPROACHING_COIN_REWARD)

    # Exploration bonus
    if new_game_state and new_game_state.get("self"):
        _, _, _, (cx, cy) = new_game_state["self"]
        if (cx, cy) not in self.visited_positions:
            reward += EXPLORATION_BONUS
            self.visited_positions.add((cx, cy))
            self.metrics_tracker.record_event("EXPLORED_NEW_TILE", reward=EXPLORATION_BONUS)

    # Not terminal yet
    done = False

    # Store transition
    try:
        self.ppo_agent.store_transition(
            obs=self.last_obs,
            action=self.last_action,
            log_prob=self.last_log_prob,
            reward=reward,
            value=self.last_value,
            done=done,
            mask=getattr(self, "last_action_mask", None)
        )
    except Exception as exc:
        self.logger.error(f"Failed to store transition: {exc}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once at the end of each round.
    Finalizes reward, updates PPO, and saves model checkpoint.
    """
    # Safety check
    if not hasattr(self, "ppo_agent") or self.ppo_agent is None:
        self.logger.warning("ppo_agent not available at end_of_round - skipping update.")
        return

    # Record ALL final events to metrics tracker
    # NOTE: end_of_round receives events that occur at round end (death, survival, etc.)
    # These must be recorded for accurate metrics tracking
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        for event in events:
            event_reward = GAME_REWARDS.get(event, 0)
            self.metrics_tracker.record_event(event, reward=event_reward)

    # Final reward computation
    final_reward = 0.0
    for event in events:
        event_reward = GAME_REWARDS.get(event, 0)
        final_reward += event_reward

    # Store final transition if we have last_obs
    if hasattr(self, "last_obs") and self.last_obs is not None:
        try:
            self.ppo_agent.store_transition(
                obs=self.last_obs,
                action=self.last_action,
                log_prob=self.last_log_prob,
                reward=final_reward,
                value=self.last_value,
                done=True,  # Terminal state
                mask=getattr(self, "last_action_mask", None)
            )
        except Exception as exc:
            self.logger.error(f"Failed to store final transition: {exc}")

    # Perform PPO update
    try:
        update_info = self.ppo_agent.update()
        if isinstance(update_info, dict):
            if update_info.get("updated"):
                self.logger.info(
                    "PPO update completed: steps=%d actor_loss=%.4f critic_loss=%.4f entropy=%.4f kl=%.4f",
                    update_info.get("num_samples", 0),
                    update_info.get("avg_actor_loss", 0.0),
                    update_info.get("avg_critic_loss", 0.0),
                    update_info.get("avg_entropy", 0.0),
                    update_info.get("avg_kl", 0.0),
                )
            else:
                self.logger.debug(
                    "Skipping PPO update (buffer=%d / required=%d)",
                    update_info.get("num_samples", 0),
                    getattr(self.ppo_agent, "batch_size", 0),
                )
        else:
            self.logger.info("PPO update completed successfully.")
    except Exception as exc:
        self.logger.error(f"PPO update failed: {exc}")

    # Save model checkpoint
    try:
        self.ppo_agent.save(MODEL_PATH)
        self.logger.info(f"Model checkpoint saved to '{MODEL_PATH}'")
    except Exception as exc:
        self.logger.error(f"Failed to save PPO model: {exc}")

    # Increment round counter
    self.round_counter = getattr(self, "round_counter", 0) + 1
    self.logger.info(f"Round {self.round_counter} ended.")

    # End episode metrics
    won = False
    rank = 4

    if e.SURVIVED_ROUND in events:
        if 'others' in last_game_state:
            alive_opponents = sum(1 for o in last_game_state['others'] if o)
            won = (alive_opponents == 0)
            rank = 1 if won else 2

    current_step = last_game_state.get('step')
    self.metrics_tracker.end_episode(
        won="WON" in events,
        rank=rank,
        survival_steps=current_step,
        total_steps=400
    )
    self.metrics_tracker.save()

    # Log training statistics
    if len(self.ppo_agent.memory["rewards"]) > 0:
        total_reward = sum(self.ppo_agent.memory["rewards"])
        self.logger.info(f"Round {self.round_counter} - Total reward: {total_reward:.2f}")
