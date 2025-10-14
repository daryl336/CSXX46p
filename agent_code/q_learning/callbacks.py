import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    """Setup Q-table"""
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = {}  # Q-table: {state: {action: value}}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    # Hyperparameters
    self.epsilon = 0.15  # exploration rate
    self.alpha = 0.1  # learning rate
    self.gamma = 0.9  # discount factor


def act(self, game_state: dict) -> str:
    """
    Choose action using epsilon-greedy policy.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    features = state_to_features(game_state)

    # Epsilon-greedy exploration
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Exploring: choosing random action")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .15, .05])

    # Exploitation: choose best action based on Q-values
    if features not in self.model:
        # Initialize Q-values for unseen state
        self.model[features] = {action: 0.0 for action in ACTIONS}

    q_values = self.model[features]
    best_action = max(q_values, key=q_values.get)

    self.logger.debug(f"Exploiting: choosing action {best_action}")
    return best_action

def state_to_features(game_state: dict) -> tuple:
    """
    Convert game state to feature tuple for Q-table lookup.

    Features extracted:
    - What's in each adjacent tile (up, right, down, left)
    - Direction to nearest coin
    - Whether bomb is available
    - Basic danger assessment

    :param game_state: A dictionary describing the current game board.
    :return: tuple of features (hashable for Q-table)
    """
    if game_state is None:
        return None

    # Extract basic info
    _, score, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']

    features = []

    # Feature 1-4: Adjacent tiles (UP, RIGHT, DOWN, LEFT)
    adjacent_positions = [
        (x, y - 1),  # UP
        (x + 1, y),  # RIGHT
        (x, y + 1),  # DOWN
        (x - 1, y),  # LEFT
    ]

    for pos_x, pos_y in adjacent_positions:
        tile_feature = get_tile_feature(pos_x, pos_y, field, bombs, explosion_map)
        features.append(tile_feature)

    # Feature 5: Direction to nearest coin
    if len(coins) > 0:
        coin_direction = get_direction_to_nearest(x, y, coins)
        features.append(coin_direction)
    else:
        features.append('no_coin')

    # Feature 6: Can place bomb?
    features.append('can_bomb' if bomb_available else 'no_bomb')

    # Feature 7: Am I in danger? (standing on explosion or near bomb)
    in_danger = is_in_danger(x, y, bombs, explosion_map)
    features.append('danger' if in_danger else 'safe')

    return tuple(features)


def get_tile_feature(x, y, field, bombs, explosion_map):
    """
    Determine what's at a specific tile.
    Returns: 'wall', 'crate', 'bomb', 'explosion', 'free'
    """
    # Check if out of bounds or wall
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
        return 'wall'

    if field[x, y] == -1:
        return 'wall'

    if field[x, y] == 1:
        return 'crate'

    # Check for active explosion
    if explosion_map[x, y] > 0:
        return 'explosion'

    # Check for bomb
    for bomb_pos, _ in bombs:
        if bomb_pos == (x, y):
            return 'bomb'

    return 'free'


def get_direction_to_nearest(x, y, targets):
    """
    Get general direction to nearest target (coin, crate, etc.).
    Returns: 'up', 'right', 'down', 'left', 'here'
    """
    if len(targets) == 0:
        return 'here'

    # Find nearest target using Manhattan distance
    min_distance = float('inf')
    nearest = None

    for target in targets:
        if isinstance(target, tuple):
            tx, ty = target
        else:
            tx, ty = target  # assuming target is array-like

        distance = abs(tx - x) + abs(ty - y)
        if distance < min_distance:
            min_distance = distance
            nearest = (tx, ty)

    if nearest is None:
        return 'here'

    tx, ty = nearest

    # Determine primary direction
    dx = tx - x
    dy = ty - y

    # Prioritize horizontal or vertical based on larger difference
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    elif abs(dy) > abs(dx):
        return 'down' if dy > 0 else 'up'
    elif dx > 0:
        return 'right'
    elif dx < 0:
        return 'left'
    elif dy > 0:
        return 'down'
    elif dy < 0:
        return 'up'
    else:
        return 'here'


def is_in_danger(x, y, bombs, explosion_map):
    """
    Check if current position is dangerous.
    Returns True if standing on explosion or in bomb blast radius.
    """
    # Currently in explosion
    if explosion_map[x, y] > 0:
        return True

    # Check if in blast radius of any bomb
    for (bx, by), timer in bombs:
        # Bombs have blast radius of 3 in each direction
        if bx == x and abs(by - y) <= 3:
            return True
        if by == y and abs(bx - x) <= 3:
            return True

    return False