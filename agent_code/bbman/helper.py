from collections import deque
import os
import numpy as np
import events as e
from typing import List

SEQUENCE_LENGTH = 8 
ACTIONS: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# -------------------------
# Enhanced state feature extraction with multi-channel representation
# -------------------------
def state_to_features(game_state: dict, frame_buffer: deque = None):
    """
    Convert game_state to multi-channel representation for CNN-LSTM processing.
    Returns: (sequence_length, channels, height, width) array
    """
    if frame_buffer is None:
        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        
    if game_state is None:
        # Return zeros with proper shape for sequence
        return np.zeros((SEQUENCE_LENGTH, 4, 17, 17), dtype=np.float32)
    
    # Extract game components
    field = game_state.get('field', np.zeros((17, 17)))
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros((17, 17)))
    
    # Handle self position - FIXED: self info format is (name, score, bomb_possible, (x, y))
    self_info = game_state.get('self')
    if self_info is not None and len(self_info) >= 4:
        # self_info format: (name, score, bomb_possible, (x, y))
        self_pos = self_info[3]  # Position is the 4th element (index 3)
        if isinstance(self_pos, (tuple, list)) and len(self_pos) >= 2:
            self_pos = (int(self_pos[0]), int(self_pos[1]))
        else:
            self_pos = (0, 0)
    else:
        self_pos = (0, 0)
    
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    
    # Create multi-channel representation
    channels = np.zeros((4, 17, 17), dtype=np.float32)
    
    # Channel 0: Field (walls, crates, free space)
    channels[0] = field.astype(np.float32) / 2.0
    
    # Channel 1: Bombs and explosion timers
    bomb_map = np.zeros((17, 17), dtype=np.float32)
    for bomb_info in bombs:
        if len(bomb_info) >= 3:
            x, y, t = bomb_info[:3]
            if 0 <= x < 17 and 0 <= y < 17:
                bomb_map[x, y] = t / 4.0
    channels[1] = bomb_map
    channels[1] += explosions.astype(np.float32) * 2.0
    
    # Channel 2: Player positions
    player_map = np.zeros((17, 17), dtype=np.float32)
    # Self position
    if 0 <= self_pos[0] < 17 and 0 <= self_pos[1] < 17:
        player_map[self_pos[0], self_pos[1]] = 1.0
    # Other players
    for other in others:
        if len(other) >= 4:
            other_pos = other[3]
            if isinstance(other_pos, (tuple, list)) and len(other_pos) >= 2:
                x, y = int(other_pos[0]), int(other_pos[1])
                if 0 <= x < 17 and 0 <= y < 17:
                    player_map[x, y] = -1.0
    channels[2] = player_map
    
    # Channel 3: Coins
    coin_map = np.zeros((17, 17), dtype=np.float32)
    for coin in coins:
        if isinstance(coin, (tuple, list)) and len(coin) >= 2:
            x, y = int(coin[0]), int(coin[1])
            if 0 <= x < 17 and 0 <= y < 17:
                coin_map[x, y] = 1.0
    channels[3] = coin_map
    
    # Add current frame to buffer
    frame_buffer.append(channels)
    
    # Return sequence of frames
    if len(frame_buffer) < SEQUENCE_LENGTH:
        padded_sequence = []
        first_frame = list(frame_buffer)[0] if frame_buffer else channels
        for i in range(SEQUENCE_LENGTH):
            if i < len(frame_buffer):
                padded_sequence.append(list(frame_buffer)[i])
            else:
                padded_sequence.append(first_frame)
        return np.stack(padded_sequence, axis=0)
    else:
        return np.stack(list(frame_buffer), axis=0)

# -------------------------
# Enhanced Epsilon Decay
# -------------------------
def update_epsilon(self):
    """Linear epsilon decay over specified steps"""
    try:
        self.epsilon_step += 1
    except Exception as e:
        self.epsilon_step = 0
        self.epsilon_step += 1
    frac = min(1.0, self.epsilon_step / self.epsilon_decay_steps)
    self.epsilon = 1.0 + frac * (self.epsilon_min - 1.0)

# -------------------------
# Enhanced reward shaping
# -------------------------
def reward_from_events(events: List[str], old_state=None) -> float:
    reward = 0.0

    game_rewards = {
        e.COIN_COLLECTED: 5.,
        e.KILLED_OPPONENT: 0.0,
        e.BOMB_DROPPED: -10.0,
        #e.KILLED_SELF: -9.,
        e.MOVED_DOWN: .05,
        e.MOVED_UP: .05,
        e.MOVED_RIGHT: .05,
        e.MOVED_LEFT: .05,
        e.INVALID_ACTION: -.5,
        e.WAITED: -.3,
        e.CRATE_DESTROYED: -.2,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    
    # # Primary objectives
    # if e.COIN_COLLECTED in events:
    #     reward += 15.0
    # if e.KILLED_OPPONENT in events:
    #     reward += 30.0

    # # Penalties for death
    # if e.KILLED_SELF in events:
    #     reward -= 200.0   # stronger penalty
    # if e.GOT_KILLED in events:
    #     reward -= 50.0

    # # Movement / positioning
    # if any(move in events for move in (e.MOVED_LEFT, e.MOVED_RIGHT, e.MOVED_UP, e.MOVED_DOWN)):
    #     reward += 0.1
    # if e.INVALID_ACTION in events:
    #     reward -= 3.0

    # # Bomb placement
    # if e.BOMB_DROPPED in events:
    #     # Only lightly reward or even skip positive reward
    #     # reward += 0.1
    #     # Or immediate penalty if unsafe
    #     if old_state is not None and not is_safe_to_bomb(old_state):
    #         reward -= 20.0

    # # Crate destruction
    # if e.CRATE_DESTROYED in events:
    #     reward += 5.0

    # Optional: survival bonus (reduced)
    # reward += 0.02  
    reward = 0
    for event in events:
        if event in game_rewards:
            reward += game_rewards[event]
    # print(f"Awarded {reward} for events {', '.join(events)}")
    return reward


def is_safe_to_bomb(game_state, frame_buffer):
    """
    Heuristic: Check if agent has a path away from explosion range.
    You might approximate explosion radius = 2 or 3. Check neighbors.
    """
    # extract self position
    pos = game_state.get("self", None)
    field = game_state.get("field", None)
    if pos is None or field is None:
        return False
    x, y = pos[0], pos[1]
    # check adjacent moves (UP, DOWN, LEFT, RIGHT)
    deltas = [(0,1),(0,-1),(1,0),(-1,0)]
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        # bounds and not wall
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] != -1:  # assuming -1 is wall/obstacle
                return True
    return False
