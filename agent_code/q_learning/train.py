import pickle
import numpy as np
from typing import List
import events as e
from .callbacks import state_to_features, ACTIONS


def setup_training(self):
    """
    Initialize training-specific variables.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up training mode.")

    # Track some statistics for debugging
    self.round_rewards = []
    self.total_reward = 0

    # Epsilon decay parameters
    self.epsilon_start = 0.9
    self.epsilon_end = 0.05
    self.epsilon_decay = 0.995
    self.epsilon = self.epsilon_start


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Update Q-values based on the transition.
    This is called after each action during training.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from old_game_state to new_game_state
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Add custom events based on game state analysis
    events = add_custom_events(old_game_state, new_game_state, events)

    # Calculate reward
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # Extract features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    # Skip update if features are invalid
    if old_features is None or new_features is None:
        self.logger.warning("Skipping Q-update due to None features")
        return

    # Q-learning update
    update_q_value(self, old_features, self_action, reward, new_features)

    self.logger.debug(f'Reward for this step: {reward}')


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died.
    Update Q-values for final transition and save model.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: The last game state before episode ended.
    :param last_action: The last action taken.
    :param events: Events that occurred in the final step.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Add custom events
    events = add_custom_events(last_game_state, None, events)

    # Calculate final reward
    reward = reward_from_events(self, events)
    self.total_reward += reward

    # Q-learning update for terminal state (no next state)
    last_features = state_to_features(last_game_state)

    if last_features is not None:
        update_q_value(self, last_features, last_action, reward, None)

    # Decay epsilon
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # Log statistics
    self.round_rewards.append(self.total_reward)
    self.logger.info(f'End of round {last_game_state["round"]}')
    self.logger.info(f'Total reward this round: {self.total_reward}')
    self.logger.info(f'Epsilon: {self.epsilon:.3f}')
    self.logger.info(f'Q-table size: {len(self.model)} states')

    # Calculate average reward over last 100 rounds
    if len(self.round_rewards) >= 100:
        avg_reward = np.mean(self.round_rewards[-100:])
        self.logger.info(f'Average reward (last 100 rounds): {avg_reward:.2f}')

    # Reset for next round
    self.total_reward = 0

    # Save the model periodically
    if last_game_state["round"] % 100 == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)
        self.logger.info("Model saved.")


def update_q_value(self, state, action, reward, next_state):
    """
    Perform Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

    :param self: Agent object with model and hyperparameters
    :param state: Current state features (tuple)
    :param action: Action taken
    :param reward: Reward received
    :param next_state: Next state features (tuple), or None if terminal
    """
    # Skip if state is None
    if state is None:
        self.logger.warning("Cannot update Q-value: state is None")
        return

    # Initialize Q-values for new states
    if state not in self.model:
        self.model[state] = {a: 0.0 for a in ACTIONS}

    # Get current Q-value
    old_q = self.model[state][action]

    # Calculate max Q-value for next state
    if next_state is None:
        # Terminal state: no future rewards
        max_next_q = 0.0
    else:
        if next_state not in self.model:
            self.model[next_state] = {a: 0.0 for a in ACTIONS}
        max_next_q = max(self.model[next_state].values())

    # Q-learning update
    new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
    self.model[state][action] = new_q

    self.logger.debug(f'Q-update: {action} {old_q:.3f} -> {new_q:.3f}')


def add_custom_events(old_game_state, new_game_state, events):
    """
    Add custom events to provide more granular feedback.

    :param old_game_state: Previous state
    :param new_game_state: Current state (None if terminal)
    :param events: List of events that occurred
    :return: Updated events list
    """
    if old_game_state is None:
        return events

    # If new_game_state is None, just return events as-is
    if new_game_state is None:
        return events

    old_x, old_y = old_game_state['self'][3]
    new_x, new_y = new_game_state['self'][3]

    # Custom event: moved towards nearest coin
    old_coins = old_game_state['coins']
    if len(old_coins) > 0:
        old_min_dist = min(abs(cx - old_x) + abs(cy - old_y) for cx, cy in old_coins)
        new_min_dist = min(abs(cx - new_x) + abs(cy - new_y) for cx, cy in old_coins)

        if new_min_dist < old_min_dist:
            events.append('MOVED_TOWARDS_COIN')
        elif new_min_dist > old_min_dist:
            events.append('MOVED_AWAY_FROM_COIN')

    # Custom event: waited when could have moved
    if old_x == new_x and old_y == new_y and e.BOMB_DROPPED not in events:
        events.append('WAITED_UNNECESSARILY')

    # Custom event: moved into danger
    old_explosion_map = old_game_state['explosion_map']
    new_explosion_map = new_game_state['explosion_map']

    if old_explosion_map[old_x, old_y] == 0 and new_explosion_map[new_x, new_y] > 0:
        events.append('MOVED_INTO_DANGER')

    return events


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate reward based on events that occurred.
    Modify these values to shape agent behavior.

    :param self: Agent object
    :param events: List of events that occurred
    :return: Total reward
    """
    game_rewards = {
        # Official events
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -25,
        e.GOT_KILLED: -20,
        e.INVALID_ACTION: -5,
        e.WAITED: -0.5,
        e.BOMB_DROPPED: 2,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 5,

        # Custom events
        'MOVED_TOWARDS_COIN': 1,
        'MOVED_AWAY_FROM_COIN': -1,
        'WAITED_UNNECESSARILY': -2,
        'MOVED_INTO_DANGER': -1,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum