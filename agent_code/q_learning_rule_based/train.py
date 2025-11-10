"""
Training module for Sequential Q-Learning → Rule-Based Agent

This module handles:
1. Q-learning updates based on state transitions
2. Custom reward shaping for Q-learning component
3. Epsilon decay for exploration/exploitation balance
4. Model persistence and metrics tracking
5. Rule-based agent acts as final decision layer (not trained, just provides actions)
"""

import os
import pickle
import numpy as np
import events as e
from typing import List
from metrics.metrics_tracker import MetricsTracker
from .callbacks import state_to_features, ACTIONS

# ===================================================================
# SETUP TRAINING
# ===================================================================

def setup_training(self):
    """
    Setup Q-table and hyperparameters for hybrid agent training.
    """
    model_path = "agent_code/q_learning_rule_based/my-saved-model.pt"
    self.name = "Hybrid Q-Rule Training"

    # === Load or create model ===
    if os.path.isfile(model_path):
        print("Loading existing model for continued training...")
        with open(model_path, "rb") as file:
            saved = pickle.load(file)
            if isinstance(saved, dict) and "model" in saved:
                self.model = saved["model"]
                self.epsilon = saved.get("epsilon", 0.3)
                self.total_rounds_trained = saved.get("total_rounds_trained", 0)
            else:
                self.model = saved if isinstance(saved, dict) else {}
                self.epsilon = 0.3
                self.total_rounds_trained = 0
    else:
        print("No existing model found. Starting new Q-table.")
        self.model = {}
        self.epsilon = 0.3  # Moderate exploration for hybrid training
        self.total_rounds_trained = 0

    # === Hyperparameters ===
    # Learning rate with adaptive decay
    self.alpha = 0.1
    self.alpha_initial = 0.2  # Higher for hybrid learning
    self.alpha_final = 0.05
    self.alpha_decay_episodes = 5000

    # Discount factor
    self.gamma = 0.99

    # Epsilon decay stages (moderate exploration with rule-based guidance)
    self.epsilon_stages = [
        (0, 1000, 0.998, 0.3),      # Episodes 0-1000: decay to 0.3
        (1000, 3000, 0.997, 0.15),  # Episodes 1000-3000: decay to 0.15
        (3000, 10000, 0.995, 0.05)  # Episodes 3000+: decay to 0.05
    ]
    self.min_epsilon = 0.01

    # === Tracking ===
    self.total_reward = 0
    self.round_rewards = []
    self.rule_agreement_count = 0  # Track how often Q-learning agrees with rule-based
    self.rule_disagreement_count = 0

    # === Metrics tracking ===
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="metrics")

    self.logger.info("Hybrid Q-learning → Rule-based training setup complete.")
    self.logger.info(f"Initial epsilon: {self.epsilon:.3f}")
    self.logger.info(f"Q-table size: {len(self.model)}")


# ===================================================================
# GAME EVENTS OCCURRED (Q-Learning Update)
# ===================================================================

def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    """
    Update Q-values based on transition with hybrid reward shaping.
    """
    self.logger.debug(f"Encountered events: {', '.join(map(repr, events))}")

    # === Add custom shaping events ===
    events = add_custom_events(old_game_state, new_game_state, events, self_action)

    # === Episode initialization (for metrics tracking) ===
    if old_game_state and old_game_state.get("step", 0) == 1:
        opponent_names = []
        if "others" in old_game_state:
            for other in old_game_state["others"]:
                if other is not None:
                    opponent_names.append(other[0])

        self.metrics_tracker.start_episode(
            episode_id=old_game_state.get("round", 0),
            opponent_types=opponent_names,
            scenario="training"
        )

    # === Compute reward ===
    reward = reward_from_events(self, events)

    self.total_reward += reward

    # === Extract features ===
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # === EXPLICIT EVENT RECORDING WITH DEBUGGING ===
    if hasattr(self, "metrics_tracker") and self.metrics_tracker.current_episode:
        self.logger.info(f"Recording {len(events)} events to metrics: {events}")

        for event in events:
            # Get event name (handle both string and event module constants)
            event_name = event if isinstance(event, str) else getattr(event, 'name', str(event))
            self.metrics_tracker.record_event(event_name)
            self.logger.debug(f"Recorded event: {event_name}")

        # Verify recording
        if hasattr(self.metrics_tracker.current_episode, 'events'):
            total_events = len(self.metrics_tracker.current_episode.events)
            self.logger.info(f"Total events recorded in episode: {total_events}")

    # === Skip if invalid ===
    if old_features is None or new_features is None:
        self.logger.warning("Skipping Q-update (invalid features)")
        return

    # === Q-learning update ===
    update_q_value(self, old_features, self_action, reward, new_features)
    self.logger.debug(f"Reward: {reward:.2f}, Total reward: {self.total_reward:.2f}")


# ===================================================================
# END OF ROUND
# ===================================================================

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Handle end of round: final Q-update, epsilon decay, model save.
    """
    self.logger.debug(f"Final events: {', '.join(map(repr, events))}")

    # === Add custom events ===
    events = add_custom_events(last_game_state, None, events, last_action)

    # === Final reward ===
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # === Final Q-update ===
    last_features = state_to_features(last_game_state)
    if last_features is not None:
        update_q_value(self, last_features, last_action, reward, None)

    # === Epsilon decay ===
    self.total_rounds_trained += 1

    for start, end, decay_rate, min_eps in self.epsilon_stages:
        if start <= self.total_rounds_trained < end:
            self.epsilon = max(min_eps, self.epsilon * decay_rate)
            break

    if self.total_rounds_trained >= self.epsilon_stages[-1][1]:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_stages[-1][2])

    # === Log statistics ===
    self.round_rewards.append(self.total_reward)
    self.logger.info(f"Round {last_game_state['round']} (Total: {self.total_rounds_trained}) — Reward: {self.total_reward:.2f}")
    self.logger.info(f"Epsilon: {self.epsilon:.3f} | Q-table size: {len(self.model)}")

    if hasattr(self, 'rule_agreement_count'):
        total_rule_interactions = self.rule_agreement_count + self.rule_disagreement_count
        if total_rule_interactions > 0:
            agreement_rate = 100.0 * self.rule_agreement_count / total_rule_interactions
            self.logger.info(f"Rule-based agreement rate: {agreement_rate:.1f}% ({self.rule_agreement_count}/{total_rule_interactions})")

    if len(self.round_rewards) >= 100:
        avg_reward = np.mean(self.round_rewards[-100:])
        self.logger.info(f"Average reward (last 100): {avg_reward:.2f}")

    # === Save model every round ===
    model_dir = os.path.dirname("agent_code/q_learning_rule_based/my-saved-model.pt")
    os.makedirs(model_dir, exist_ok=True)

    with open("agent_code/q_learning_rule_based/my-saved-model.pt", "wb") as file:
        pickle.dump({
            "model": self.model,
            "epsilon": self.epsilon,
            "total_rounds_trained": self.total_rounds_trained
        }, file)
    self.logger.info(f"Model saved (Round {last_game_state['round']}, Total rounds: {self.total_rounds_trained})")

    # === Determine win/rank for metrics ===
    won = False
    rank = 4
    if e.SURVIVED_ROUND in events and "others" in last_game_state:
        alive = sum(1 for o in last_game_state["others"] if o)
        won = alive == 0
        rank = 1 if won else 2

    # === Record episode end ===
    current_step = last_game_state.get("step", 0)
    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=current_step,
        total_steps=400
    )
    self.metrics_tracker.save()

    # === Reset counters ===
    self.total_reward = 0
    if hasattr(self, 'rule_agreement_count'):
        self.rule_agreement_count = 0
        self.rule_disagreement_count = 0


# ===================================================================
# Q-VALUE UPDATE
# ===================================================================

def update_q_value(self, state, action, reward, next_state):
    """
    Q-learning update: Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s',a') - Q(s,a)]
    """
    if state is None:
        self.logger.warning("Cannot update Q-value: state is None")
        return

    # Adaptive alpha
    if hasattr(self, 'total_rounds_trained'):
        progress = min(1.0, self.total_rounds_trained / self.alpha_decay_episodes)
        alpha = self.alpha_initial * (1 - progress) + self.alpha_final * progress
    else:
        alpha = self.alpha

    # Initialize state
    if state not in self.model:
        self.model[state] = {a: 0.0 for a in ACTIONS}

    old_q = self.model[state][action]

    # Compute target
    if next_state is None:
        max_next_q = 0.0
    else:
        if next_state not in self.model:
            self.model[next_state] = {a: 0.0 for a in ACTIONS}
        max_next_q = max(self.model[next_state].values())

    # Q-update
    td_target = reward + self.gamma * max_next_q
    new_q = old_q + alpha * (td_target - old_q)
    self.model[state][action] = float(np.clip(new_q, -500, 500))

    self.logger.debug(f"Q[{action}]: {old_q:.2f} → {new_q:.2f} (α={alpha:.3f})")


# ===================================================================
# REWARD SHAPING
# ===================================================================

def reward_from_events(self, events: List) -> float:
    """
    SYNCHRONIZED with q_learning for fair comparison.
    Balanced reward shaping for both coin-collection AND crate-destruction gameplay.
    """
    game_rewards = {
        # PRIMARY GOALS - High positive rewards
        e.COIN_COLLECTED: 25.0,           # Main objective (SYNC with q_learning)
        e.KILLED_OPPONENT: 100.0,         # Bonus for combat (SYNC with q_learning)

        # COIN SEEKING - Core gameplay without crates
        "MOVED_TOWARDS_COIN": 3.0,        # Encourage coin seeking (SYNC with q_learning)
        "MOVED_AWAY_FROM_COIN": -1.0,     # Penalty for wrong direction (SYNC with q_learning)

        # SURVIVAL
        "ESCAPED_DANGER": 10.0,           # Strong reward for survival (SYNC with q_learning)
        e.SURVIVED_ROUND: 20.0,           # Big bonus for survival (SYNC with q_learning)

        # BOMB PLACEMENT - Small penalty (discourage spam in no-crate mode)
        e.BOMB_DROPPED: -0.5,             # Small penalty (SYNC with q_learning)

        # CRITICAL PENALTIES - Must avoid
        e.KILLED_SELF: -100.0,            # Biggest penalty (SYNC with q_learning)
        e.GOT_KILLED: -50.0,              # Death is bad (SYNC with q_learning)
        e.INVALID_ACTION: -10.0,          # Should not happen with valid action filter (SYNC with q_learning)
        "MOVED_INTO_DANGER": -20.0,       # Strong discouragement (SYNC with q_learning)

        # MINOR PENALTIES - Discourage bad habits
        "WAITED_UNNECESSARILY": -2.0,     # Don't waste time (SYNC with q_learning)
        e.WAITED: -1.0,                   # Waiting has cost (SYNC with q_learning)

        # POSITIVE REINFORCEMENT
        "SAFE_MOVE": 0.3,                 # Small reward for valid movement (SYNC with q_learning)
    }

    reward_sum = 0.0
    for ev in events:
        reward_sum += game_rewards.get(ev, 0.0)

    reward_sum = float(np.clip(reward_sum, -150, 300))  # SYNC with q_learning
    self.logger.debug(f"Reward: {reward_sum:.2f} for {events}")
    return reward_sum


def add_custom_events(old_game_state, new_game_state, events, self_action=None):
    """
    Add custom events for granular learning signals.
    """
    if old_game_state is None or "self" not in old_game_state:
        return events

    if new_game_state is None or "self" not in new_game_state:
        return events

    old_x, old_y = old_game_state["self"][3]
    new_x, new_y = new_game_state["self"][3]
    field = old_game_state.get("field")

    # --- Coin distance feedback ---
    old_coins = old_game_state.get("coins", [])
    if len(old_coins) > 0:
        old_min_dist = min(abs(cx - old_x) + abs(cy - old_y) for cx, cy in old_coins)
        new_min_dist = min(abs(cx - new_x) + abs(cy - new_y) for cx, cy in old_coins)

        if new_min_dist < old_min_dist:
            events.append("MOVED_TOWARDS_COIN")
        elif new_min_dist > old_min_dist:
            events.append("MOVED_AWAY_FROM_COIN")

    # --- Crate proximity ---
    if field is not None:
        old_crate_dist = min_distance_to_crate(old_x, old_y, field)
        new_crate_dist = min_distance_to_crate(new_x, new_y, field)

        if old_crate_dist is not None and new_crate_dist is not None:
            if new_crate_dist < old_crate_dist:
                events.append("MOVED_TOWARDS_CRATE")

    # --- Bomb placement analysis ---
    if e.BOMB_DROPPED in events and field is not None:
        crate_nearby = False
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx) + abs(dy) <= 3:
                    nx, ny = old_x + dx, old_y + dy
                    if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                        if field[nx, ny] == 1:
                            crate_nearby = True
                            break
            if crate_nearby:
                break
        if crate_nearby:
            events.append("BOMB_NEAR_CRATE")

    # --- Waiting unnecessarily ---
    if old_x == new_x and old_y == new_y and e.BOMB_DROPPED not in events:
        events.append("WAITED_UNNECESSARILY")

    # --- Danger awareness ---
    old_explosion_map = old_game_state.get("explosion_map")
    new_explosion_map = new_game_state.get("explosion_map")
    if old_explosion_map is not None and new_explosion_map is not None:
        if old_explosion_map[old_x, old_y] == 0 and new_explosion_map[new_x, new_y] > 0:
            events.append("MOVED_INTO_DANGER")
        elif old_explosion_map[old_x, old_y] > 0 and new_explosion_map[new_x, new_y] == 0:
            events.append("ESCAPED_DANGER")

    # --- Safe move reward ---
    if e.INVALID_ACTION not in events and "WAITED_UNNECESSARILY" not in events:
        events.append("SAFE_MOVE")

    return events


def min_distance_to_crate(x, y, field):
    """Find minimum Manhattan distance to any crate."""
    crate_positions = np.argwhere(field == 1)
    if len(crate_positions) == 0:
        return None
    distances = [abs(cx - x) + abs(cy - y) for cx, cy in crate_positions]
    return min(distances)
