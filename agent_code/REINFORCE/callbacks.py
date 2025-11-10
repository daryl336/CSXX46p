import os
import pickle
import random
import numpy as np
from metrics.metrics_tracker import MetricsTracker
from collections import deque
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x,dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def setup(self):
    """
    Setup REINFORCE agent.
    Loads saved model if available, otherwise creates new one.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    model_path = "my-saved-model.pth"
    self.name = "REINFORCE"

    # Metrics tracker
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="metrics")
    self.episode_counter = 0
    self.episode_active = False
    self.current_step = 0

    # Hyperparameters
    self.state_size = 297  # Will be determined by state_to_features
    self.action_size = len(ACTIONS)
    self.hidden_size = 128
    self.learning_rate = 1e-5

    self.policy = Policy(self.state_size, self.action_size, self.hidden_size).to(device)
    self.optimizer = optim.Adam(self.policy.parameters(), lr = self.learning_rate)

    # Load saved model if available and not training
    if not self.train and os.path.isfile(model_path):
        self.logger.info("Loading model from saved state.")
        checkpoint = torch.load(model_path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {model_path}")
    else:
        self.logger.info("Setting up model from scratch.")

def act(self, game_state: dict) -> str:
    """
    Choose action using the policy network.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # Extract features
    features = state_to_features(game_state)

    if features is None:
        self.logger.warning("Features are None, choosing random action")
        return np.random.choice(ACTIONS)
    # =========================================================================
    # START EPISODE ON FIRST STEP (ONLY FOR GAMEPLAY MODE, NOT TRAINING)
    # =========================================================================
    # Note: During training mode, train.py handles episode start/end
    # This block is ONLY for pure gameplay mode (no --train flag)
    if game_state and game_state.get('step', 0) == 1 and not self.episode_active:
        # Extract opponent information
        opponent_names = []
        if 'others' in game_state and game_state['others']:
            for other in game_state['others']:
                if other is not None:
                    opponent_names.append(other[0])

        # Get episode ID from game state
        self.episode_counter = game_state.get('round', self.episode_counter)

        # START THE EPISODE (only if not already started by train.py)
        # Check if we're in training mode by seeing if train.py started it already
        if not self.metrics_tracker.current_episode:
            self.metrics_tracker.start_episode(
                episode_id=self.episode_counter,
                opponent_types=opponent_names,
                map_name="default",
                scenario="gameplay"
            )
            self.logger.debug(f"Started gameplay episode {self.episode_counter}")

        self.episode_active = True
        self.current_step = 0

    # Get action from policy
    if self.train:
        # During training: sample from policy
        action_idx, log_prob = self.policy.act(features)

        # Store log_prob for training (will be used in train.py)
        self.last_log_prob = log_prob
    else:
        # During evaluation: take most probable action (greedy)
        with torch.no_grad():
            action_idx, log_prob = self.policy.act(features)
    if hasattr(self, 'metrics_tracker'):
        self.metrics_tracker.record_action(ACTIONS[action_idx], is_valid=True)
    return ACTIONS[action_idx]

def state_to_features(game_state: dict) -> np.array:
    _, score, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    features = []

    #add board
    board = np.array(field).flatten()
    features.extend(board.astype(np.float32))

    # 2. Agent position (normalized to [0, 1])
    field_size = field.shape[0]
    features.append(x / field_size)
    features.append(y / field_size)

    # 3. Nearest coin information (4 features)
    coin_features = get_nearest_coin_features(x, y, coins, field_size)
    features.extend(coin_features)

    # 4. Bomb availability (1 feature)
    features.append(1.0 if bomb_available else 0.0)

    # 5. Danger level at current position (1 feature)
    danger = get_danger_level(x, y, bombs, explosion_map)
    features.append(danger)

    return np.array(features, dtype=np.float32)


def get_nearest_coin_features(x, y, coins, field_size):
    """
    Get features for the nearest coin.
    Returns: [relative_x, relative_y, distance, exists]

    :param x: Agent x position
    :param y: Agent y position
    :param coins: List of coin positions
    :param field_size: Size of the field for normalization
    :return: List of 4 features
    """
    if len(coins) == 0:
        return [0.0, 0.0, 0.0, 0.0]  # No coin exists

    # Find nearest coin
    min_distance = float('inf')
    nearest_coin = None

    for coin in coins:
        cx, cy = coin if isinstance(coin, tuple) else (coin[0], coin[1])
        distance = abs(cx - x) + abs(cy - y)  # Manhattan distance

        if distance < min_distance:
            min_distance = distance
            nearest_coin = (cx, cy)

    if nearest_coin is None:
        return [0.0, 0.0, 0.0, 0.0]

    cx, cy = nearest_coin
    # Relative position (normalized)
    rel_x = (cx - x) / field_size
    rel_y = (cy - y) / field_size

    # Distance (normalized)
    norm_distance = min_distance / (2 * field_size)

    # Exists flag
    exists = 1.0

    return [rel_x, rel_y, norm_distance, exists]


def get_danger_level(x, y, bombs, explosion_map):
    """
    Calculate danger level at position (x, y).
    Returns value between 0 (safe) and 1 (very dangerous).

    :param x: X position
    :param y: Y position
    :param bombs: List of bombs [(position, timer), ...]
    :param explosion_map: Map of current explosions
    :return: Danger level (0.0 to 1.0)
    """
    # Currently exploding
    if explosion_map[x, y] > 0:
        return 1.0

    # Check bombs
    max_danger = 0.0

    for (bx, by), timer in bombs:
        # Check if in blast line (same row or column within range 3)
        in_blast_line = (bx == x and abs(by - y) <= 3) or (by == y and abs(bx - x) <= 3)

        if in_blast_line:
            # Manhattan distance to bomb
            dist = abs(bx - x) + abs(by - y)

            # Danger increases with proximity and decreases with timer
            # If bomb is about to explode (timer=1) and close (dist=1): high danger
            # If bomb is far away (timer=4) and distant (dist=3): low danger
            danger = (4 - dist) / 4.0 * (5 - timer) / 5.0
            max_danger = max(max_danger, danger)

    return max_danger