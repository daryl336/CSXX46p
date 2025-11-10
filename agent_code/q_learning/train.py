import os
import pickle
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS, reward_from_events
from metrics.metrics_tracker import MetricsTracker

def setup_training(self):
    """
    Setup Q-table and hyperparameters for Q-learning training.
    Loads saved model if available for continued training.
    """
    model_path = "my-saved-model.pt"
    self.name = "Q Learning"

    # === Load or create model ===
    if os.path.isfile(model_path):
        print("Loading existing model for continued training...")
        with open(model_path, "rb") as file:
            saved = pickle.load(file)
            if isinstance(saved, dict) and "model" in saved:
                self.model = saved["model"]
                self.epsilon = saved.get("epsilon", 1.0)
                self.total_rounds_trained = saved.get("total_rounds_trained", 0)
            else:
                self.model = saved
                self.epsilon = 1.0
                self.total_rounds_trained = 0
    else:
        print("No existing model found. Starting new Q-table.")
        self.model = {}
        self.epsilon = 1.0
        self.total_rounds_trained = 0

    # === Hyperparameters ===
    # STAGED EPSILON DECAY
    self.epsilon_decay = 0.998  # Aligned with callbacks.py
    self.min_epsilon = 0.05     # Aligned with callbacks.py

    # Adaptive decay based on performance
    self.epsilon_stages = [
        (0, 2000, 0.998, 0.2),      # Episodes 0-2000: decay to 0.2
        (2000, 5000, 0.997, 0.1),   # Episodes 2000-5000: decay to 0.1
        (5000, 10000, 0.995, 0.05)  # Episodes 5000+: decay to 0.05
    ]
    # ADAPTIVE LEARNING RATE
    self.alpha = 0.1             # Base alpha (aligned with callbacks.py)
    self.alpha_initial = 0.3     # Optimal for coin-collection learning
    self.alpha_final = 0.05      # Low final for stable convergence
    self.alpha_decay_episodes = 5000

    # DISCOUNT FACTOR
    self.gamma = 0.99  # Aligned with callbacks.py
    
    # === Metrics tracking ===
    # Use separate folder for evaluation metrics
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.total_reward = 0
    self.round_rewards = []

    self.logger.info("Training setup complete.")


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    """
    Update Q-values based on transition, with dense reward shaping.
    """
    self.logger.debug(f"Encountered events: {', '.join(map(repr, events))}")

    # Add custom shaping events
    events = add_custom_events(old_game_state, new_game_state, events)

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


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Handle end of round, apply final Q-update, save model, and log metrics.
    """
    self.logger.debug(f"Final events: {', '.join(map(repr, events))}")

    # Add custom events
    events = add_custom_events(last_game_state, None, events)

    # === Record final events to metrics tracker ===
    if hasattr(self, "metrics_tracker") and self.metrics_tracker.current_episode:
        self.logger.info(f"Recording {len(events)} FINAL events to metrics: {events}")
        for event in events:
            event_name = event if isinstance(event, str) else getattr(event, 'name', str(event))
            self.metrics_tracker.record_event(event_name)
            self.logger.debug(f"Recorded final event: {event_name}")

    # === Final reward ===
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # === Final Q-update ===
    last_features = state_to_features(last_game_state)
    if last_features is not None:
        update_q_value(self, last_features, last_action, reward, None)

    # === Epsilon decay ===
    # Increment cumulative round counter
    self.total_rounds_trained += 1

    # Normal epsilon decay (removed freeze - was causing too much random bombing)
    for start, end, decay_rate, min_eps in self.epsilon_stages:
        if start <= self.total_rounds_trained < end:
            self.epsilon = max(min_eps, self.epsilon * decay_rate)
            break

    # After last stage, continue decaying to min_epsilon
    if self.total_rounds_trained >= self.epsilon_stages[-1][1]:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_stages[-1][2])
    # === Log statistics ===
    self.round_rewards.append(self.total_reward)
    self.logger.info(f"Round {last_game_state['round']} (Total: {self.total_rounds_trained}) — Reward: {self.total_reward:.2f}")
    self.logger.info(f"Epsilon: {self.epsilon:.3f} | Q-table size: {len(self.model)}")

    if len(self.round_rewards) >= 100:
        avg_reward = np.mean(self.round_rewards[-100:])
        self.logger.info(f"Average reward (last 100): {avg_reward:.2f}")

    # === Save model periodically ===
    if last_game_state["round"] % 100 == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump({
                "model": self.model,
                "epsilon": self.epsilon,
                "total_rounds_trained": self.total_rounds_trained
            }, file)
        self.logger.info(f"Model saved (Total rounds: {self.total_rounds_trained})")

    # === Determine win/rank for metrics ===
    won = "WON" in events
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

def add_custom_events(old_game_state, new_game_state, events):
    """
    Adds custom events for more granular learning signals.

    :param old_game_state: State before action
    :param new_game_state: State after action (or None if terminal)
    :param events: Base events triggered by environment
    :return: Updated events list
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

    # --- Strategic bomb placement ---
    # NOTE: Disabled for CRATE_DENSITY=0 training (no crates to bomb)
    # if e.BOMB_DROPPED in events and field is not None:
    #     # Check if bomb is near crates (within blast radius)
    #     crate_nearby = False
    #     for dx in range(-3, 4):
    #         for dy in range(-3, 4):
    #             if abs(dx) + abs(dy) <= 3:  # Manhattan distance
    #                 nx, ny = old_x + dx, old_y + dy
    #                 if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
    #                     if field[nx, ny] == 1:  # Crate
    #                         crate_nearby = True
    #                         break
    #         if crate_nearby:
    #             break
    #     if crate_nearby:
    #         events.append("BOMB_NEAR_CRATE")

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

    # --- Encourage safe, valid moves ---
    if e.INVALID_ACTION not in events and "WAITED_UNNECESSARILY" not in events:
        events.append("SAFE_MOVE")

    return events

def update_q_value(self, state, action, reward, next_state):
    """
    Perform Q-learning update:
    Q(s,a) ← Q(s,a) + α [r + γ * max_a' Q(s',a') - Q(s,a)]

    Handles initialization of unseen states.
    """

    if state is None:
        self.logger.warning("Cannot update Q-value: state is None")
        return

    # Adaptive alpha based on episode
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        episode = self.metrics_tracker.current_episode.episode_id
        progress = min(1.0, episode / self.alpha_decay_episodes)
        alpha = self.alpha_initial * (1 - progress) + self.alpha_final * progress
    else:
        alpha = self.alpha
        
    # Initialize
    if state not in self.model:
        self.model[state] = {a: 0.0 for a in ACTIONS}

    old_q = self.model[state][action]

    # Future reward
    if next_state is None:
        max_next_q = 0.0
    else:
        if next_state not in self.model:
            self.model[next_state] = {a: 0.0 for a in ACTIONS}
        max_next_q = max(self.model[next_state].values())

    # Q-update with adaptive alpha
    td_target = reward + self.gamma * max_next_q
    new_q = old_q + alpha * (td_target - old_q)
    self.model[state][action] = float(np.clip(new_q, -500, 500))

    self.logger.debug(f"Q[{action}]: {old_q:.2f} → {new_q:.2f} (α={alpha:.3f})")