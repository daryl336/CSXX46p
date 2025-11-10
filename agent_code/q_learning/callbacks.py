import os
import pickle
import random
import numpy as np
from metrics.metrics_tracker import MetricsTracker
from collections import deque

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup Q-table and hyperparameters.
    Loads saved model if available for continued training.
    """
    model_path = "my-saved-model.pt"
    self.name = "Q Learning"

    # Load model if exists
    if os.path.isfile(model_path):
        print("Loading existing model for continued training...")
        with open(model_path, "rb") as file:
            saved = pickle.load(file)
            self.model = saved.get("model", {})
            self.epsilon = saved.get("epsilon", 1.0)
    else:
        print("No existing model found. Starting new Q-table.")
        self.model = {}
        self.epsilon = 1.0  # initial exploration

    # Metrics tracker - using separate folder for evaluation
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.episode_counter = 0
    self.episode_active = False
    self.current_step = 0

    # Hyperparameters (shared with train.py)
    self.alpha = 0.1   # learning rate (will be adaptive in training)
    self.gamma = 0.99  # discount factor
    self.epsilon_decay = 0.998  # Aligned with train.py
    self.min_epsilon = 0.05


def act(self, game_state) -> str:
    """
    Choose action using epsilon-greedy policy with safety filtering.
    """
    features = state_to_features(game_state)
    if features is None:
        return np.random.choice(ACTIONS)

    # Initialize unseen state
    if features not in self.model:
        self.model[features] = {action: 0.0 for action in ACTIONS}

    # Get valid/safe actions
    valid_actions = get_valid_actions(game_state)
    if len(valid_actions) == 0:
        valid_actions = ACTIONS  # Fallback if all actions seem invalid

    # Epsilon-greedy exploration
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Exploring: choosing random action")
        # Prefer movement over waiting/bombing during exploration
        action = np.random.choice(valid_actions)
    else:
        # Exploitation: choose best valid action
        q_values = self.model[features]
        # Filter Q-values to only valid actions
        valid_q = {a: q_values[a] for a in valid_actions if a in q_values}
        if valid_q:
            action = max(valid_q, key=valid_q.get)
        else:
            action = np.random.choice(valid_actions)

    # ======================================
    # METRICS TRACKING (works for both training and play modes)
    # ======================================
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="metrics")

    # Detect if this is a new episode (round start)
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

    return action


def _end_episode_from_act(self, game_state):
    """
    End the current episode and save metrics.
    Called from act() when a new round is detected (for play mode).
    """
    if not hasattr(self, 'metrics_tracker') or not self.episode_active:
        return

    # Determine outcome based on final game state
    # In play mode, we detect round end by seeing step=1 of next round
    # So this is called with the FINAL state of the previous round

    # Simple heuristic: if agent is still alive in others list or is 'self', survived
    survived = True
    won = False
    rank = 4

    if game_state and 'self' in game_state:
        # Agent is alive (otherwise wouldn't be in game_state)
        survived = True

        # Check if won (no other agents alive)
        if 'others' in game_state:
            alive_others = sum(1 for o in game_state.get('others', []) if o is not None)
            if alive_others == 0:
                won = True
                rank = 1
            else:
                rank = 2  # Survived but didn't win

    # Get survival steps
    survival_steps = self.current_step if hasattr(self, 'current_step') else 0
    total_steps = game_state.get('step', 400) if game_state else 400

    # End episode
    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=survival_steps,
        total_steps=total_steps,
        metadata={"mode": "play_detection"}
    )

    # Save metrics
    self.metrics_tracker.save()

    # Mark episode as inactive
    self.episode_active = False


# ======================================================
# FEATURE ENGINEERING
# ======================================================

def state_to_features(game_state: dict) -> tuple:
    """
    Enhanced feature engineering with better state representation.

    Feature composition:
    1. Movement safety (4 directions) - can move safely
    2. Immediate danger level - none/low/high
    3. Coin direction + distance bucket - where to go for coins
    4. Escape routes available - number of safe exits
    5. Bomb value - should I bomb here?
    6. Crate proximity - are there crates nearby to destroy?

    Estimated state space: ~3-5K commonly visited states (vs 259 before)
    """
    if game_state is None:
        return None

    _, score, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']

    features = []

    # ========================================
    # FEATURE 1: MOVEMENT SAFETY (4 directions)
    # ========================================
    # For each direction, can we move there safely?
    moves = {
        'UP': (x, y - 1),
        'RIGHT': (x + 1, y),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y)
    }

    movement_safety = []
    for direction in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        nx, ny = moves[direction]
        safety = get_move_safety(nx, ny, field, bombs, explosion_map)
        movement_safety.append(safety)
    features.extend(movement_safety)

    # ========================================
    # FEATURE 2: DANGER LEVEL AT CURRENT POSITION
    # ========================================
    danger_level = get_danger_level(x, y, bombs, explosion_map, field)
    features.append(danger_level)

    # ========================================
    # FEATURE 3: COIN NAVIGATION (direction + distance)
    # ========================================
    if len(coins) > 0:
        coin_direction = get_direction_to_nearest(x, y, coins)
        coin_distance = get_distance_bucket(x, y, coins)
        features.append(coin_direction)
        features.append(coin_distance)
    else:
        features.append('no_coin')
        features.append('no_distance')

    # ========================================
    # FEATURE 4: ESCAPE ROUTES AVAILABLE
    # ========================================
    escape_count = count_escape_routes(x, y, field, bombs, explosion_map)
    features.append(escape_count)

    # ========================================
    # FEATURE 5: BOMB PLACEMENT VALUE
    # ========================================
    if bomb_available:
        bomb_value = evaluate_bomb_placement(x, y, field, bombs, explosion_map)
        features.append(bomb_value)
    else:
        features.append('no_bomb')

    # ========================================
    # FEATURE 6: CRATE PROXIMITY (for strategic bombing)
    # ========================================
    crate_nearby = count_nearby_crates(x, y, field)
    features.append(crate_nearby)

    return tuple(features)

# ========================================
# HELPER FUNCTIONS FOR ENHANCED FEATURES
# ========================================

def get_move_safety(x, y, field, bombs, explosion_map):
    """
    Evaluate safety of moving to position (x, y).
    Returns: 'safe', 'risky', 'blocked'
    """
    # Out of bounds or solid obstacle
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
        return 'blocked'
    if field[x, y] == -1 or field[x, y] == 1:  # Wall or crate
        return 'blocked'

    # Check if there's an active explosion
    if explosion_map[x, y] > 0:
        return 'blocked'

    # Check if position has a bomb
    bomb_positions = [pos for pos, _ in bombs]
    if (x, y) in bomb_positions:
        return 'blocked'

    # Check if position is in danger zone
    if is_in_danger(x, y, bombs, explosion_map, field):
        return 'risky'

    return 'safe'


def get_danger_level(x, y, bombs, explosion_map, field):
    """
    Evaluate danger level at current position.
    Returns: 'none', 'low', 'high', 'critical'
    """
    # Already in explosion
    if explosion_map[x, y] > 0:
        return 'critical'

    # Check proximity to bombs
    min_bomb_timer = 999
    in_blast_radius = False

    bomb_power = 3
    for (bx, by), timer in bombs:
        # Check if we're in blast radius
        if (bx == x and abs(by - y) <= bomb_power) or (by == y and abs(bx - x) <= bomb_power):
            # Verify no walls blocking
            blocked = False
            if bx == x:  # Same column
                for check_y in range(min(y, by), max(y, by) + 1):
                    if field[x, check_y] == -1:
                        blocked = True
                        break
            else:  # Same row
                for check_x in range(min(x, bx), max(x, bx) + 1):
                    if field[check_x, y] == -1:
                        blocked = True
                        break

            if not blocked:
                in_blast_radius = True
                min_bomb_timer = min(min_bomb_timer, timer)

    if not in_blast_radius:
        return 'none'

    # Danger based on timer
    if min_bomb_timer <= 1:
        return 'critical'
    elif min_bomb_timer <= 2:
        return 'high'
    else:
        return 'low'


def get_distance_bucket(x, y, targets):
    """
    Categorize distance to nearest target.
    Returns: 'very_close' (<= 2), 'close' (3-5), 'medium' (6-10), 'far' (>10)
    """
    if len(targets) == 0:
        return 'none'

    min_dist = min(abs(tx - x) + abs(ty - y) for tx, ty in targets)

    if min_dist <= 2:
        return 'very_close'
    elif min_dist <= 5:
        return 'close'
    elif min_dist <= 10:
        return 'medium'
    else:
        return 'far'


def count_escape_routes(x, y, field, bombs, explosion_map):
    """
    Count number of safe adjacent tiles (escape routes).
    Returns: '0', '1', '2', '3+'
    """
    safe_count = 0
    adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    for nx, ny in adjacent:
        if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
            continue
        if field[nx, ny] != 0:  # Not free
            continue
        if explosion_map[nx, ny] > 0:  # In explosion
            continue
        if is_in_danger(nx, ny, bombs, explosion_map, field):
            continue
        safe_count += 1

    if safe_count == 0:
        return '0'
    elif safe_count == 1:
        return '1'
    elif safe_count == 2:
        return '2'
    else:
        return '3+'


def evaluate_bomb_placement(x, y, field, bombs, explosion_map):
    """
    Evaluate if placing a bomb here is valuable.
    Returns: 'good' (destroys crates/traps enemies), 'neutral', 'bad' (no value or unsafe)
    """
    # Check if we can escape after placing bomb
    test_bombs = bombs + [((x, y), 4)]
    escape_possible = False

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and not is_in_danger(nx, ny, test_bombs, explosion_map, field):
                escape_possible = True
                break

    if not escape_possible:
        return 'bad'

    # Count valuable targets in blast radius
    crates_hit = 0
    bomb_power = 3

    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        dx, dy = direction
        for dist in range(1, bomb_power + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
                break
            if field[nx, ny] == -1:  # Wall
                break
            if field[nx, ny] == 1:  # Crate
                crates_hit += 1
                break

    if crates_hit >= 2:
        return 'good'
    elif crates_hit == 1:
        return 'neutral'
    else:
        return 'bad'


def count_nearby_crates(x, y, field):
    """
    Count crates in surrounding area (Manhattan distance <= 3).
    Returns: '0', '1-2', '3-5', '6+'
    """
    crate_count = 0
    search_radius = 3

    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            if abs(dx) + abs(dy) > search_radius:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 1:
                    crate_count += 1

    if crate_count == 0:
        return '0'
    elif crate_count <= 2:
        return '1-2'
    elif crate_count <= 5:
        return '3-5'
    else:
        return '6+'


def get_direction_to_nearest(x, y, targets):
    """
    Get direction to nearest target (coin, crate, etc.).
    Returns: 'up', 'down', 'left', 'right', 'here'
    """
    if len(targets) == 0:
        return 'here'
    tx, ty = min(targets, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
    dx, dy = tx - x, ty - y
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    elif abs(dy) > abs(dx):
        return 'down' if dy > 0 else 'up'
    return 'here'


def is_in_danger(x, y, bombs, explosion_map, field):
    """
    Improved danger detection with escape path checking and boundary safety.
    """
    if explosion_map[x, y] > 0:
        return True

    danger_tiles = set()
    bomb_power = 3  # From settings.py

    for (bx, by), timer in bombs:
        if timer <= 3:
            # Add bomb position itself
            danger_tiles.add((bx, by))

            # Check all four directions with boundary checks
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in directions:
                for dist in range(1, bomb_power + 1):
                    nx, ny = bx + dx * dist, by + dy * dist
                    # Boundary check
                    if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
                        break
                    # Wall blocks explosion
                    if field[nx, ny] == -1:
                        break
                    danger_tiles.add((nx, ny))
                    # Crate blocks further propagation
                    if field[nx, ny] == 1:
                        break

    if (x, y) in danger_tiles:
        # Check for escape path
        safe_dirs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in safe_dirs:
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and (nx, ny) not in danger_tiles:
                    return False
        return True
    return False


def count_crates_in_blast_radius(x, y, field):
    """
    Count how many crates would be destroyed by a bomb at (x, y).
    """
    crate_count = 0
    bomb_power = 3

    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        dx, dy = direction
        for dist in range(1, bomb_power + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
                break
            if field[nx, ny] == -1:  # Wall blocks
                break
            if field[nx, ny] == 1:  # Crate!
                crate_count += 1
                break  # Crate blocks further propagation

    return crate_count


def get_valid_actions(game_state):
    """
    Returns list of valid actions that don't lead to immediate danger or walls.
    """
    if game_state is None:
        return ACTIONS

    _, _, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    valid = []

    # Check movement actions
    moves = {
        'UP': (x, y - 1),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y),
        'RIGHT': (x + 1, y)
    }

    for action, (nx, ny) in moves.items():
        # Check boundaries
        if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
            continue
        # Check if free tile
        if field[nx, ny] != 0:
            continue
        # Check if not in explosion
        if explosion_map[nx, ny] > 0:
            continue
        # Check if not a bomb position
        bomb_positions = [pos for pos, _ in bombs]
        if (nx, ny) in bomb_positions:
            continue
        # Check if not moving into immediate danger
        if is_in_danger(nx, ny, bombs, explosion_map, field):
            continue
        valid.append(action)

    # WAIT is valid if current position is safe
    if not is_in_danger(x, y, bombs, explosion_map, field):
        valid.append('WAIT')

    # BOMB is valid if available AND has safe escape
    # NOTE: With CRATE_DENSITY=0, bombing is rarely useful, but allow for edge cases
    if bomb_available:
        # Simulate bomb placement
        test_bombs = bombs + [((x, y), 4)]

        # Check if we have safe escape route
        can_escape = False
        for action, (nx, ny) in moves.items():
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0:  # Free tile
                    if not is_in_danger(nx, ny, test_bombs, explosion_map, field):
                        can_escape = True
                        break

        # Only allow bomb if safe escape exists
        # (Agent will learn bombing is not rewarding without crates via -0.5 penalty)
        if can_escape:
            valid.append('BOMB')

    return valid if len(valid) > 0 else ['WAIT']


# ======================================================
# REWARD SHAPING
# ======================================================

import numpy as np
import events as e

def reward_from_events(self, events: list) -> float:
    """
    EVALUATION MODE: Using standardized evaluation rewards for fair comparison.

    This uses the universal reward structure from evaluation_rewards.py
    to ensure all agents are evaluated with identical reward signals.
    """
    # Import evaluation rewards
    try:
        from evaluation_rewards import EVALUATION_REWARDS
        use_evaluation = True
    except ImportError:
        self.logger.warning("evaluation_rewards.py not found, using default rewards")
        use_evaluation = False

    if use_evaluation:
        # Use standardized evaluation rewards
        reward_sum = 0.0
        for ev in events:
            reward_sum += EVALUATION_REWARDS.get(ev, 0.0)

        self.logger.debug(f"Evaluation Reward: {reward_sum:.2f} for {events}")
        return reward_sum

    else:
        # Fallback to original training rewards
        game_rewards = {
            e.COIN_COLLECTED: 25.0,
            e.KILLED_OPPONENT: 100.0,
            "MOVED_TOWARDS_COIN": 3.0,
            "MOVED_AWAY_FROM_COIN": -1.0,
            "ESCAPED_DANGER": 10.0,
            e.SURVIVED_ROUND: 20.0,
            e.BOMB_DROPPED: -0.5,
            e.KILLED_SELF: -100.0,
            e.GOT_KILLED: -50.0,
            e.INVALID_ACTION: -10.0,
            "MOVED_INTO_DANGER": -20.0,
            "WAITED_UNNECESSARILY": -2.0,
            e.WAITED: -1.0,
            "SAFE_MOVE": 0.3,
        }

        reward_sum = 0.0
        for ev in events:
            reward_sum += game_rewards.get(ev, 0.0)

        reward_sum = float(np.clip(reward_sum, -150, 300))
        self.logger.debug(f"Training Reward: {reward_sum:.2f} for {events}")
        return reward_sum