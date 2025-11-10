"""
Sequential Q-Learning → Rule-Based Agent

Architecture Flow:
1. Q-Learning Model: Analyzes game state and suggests an action
2. Rule-Based Agent: Makes final decision using hard-coded rules
3. Final Decision: Rule-based agent's action is executed

This sequential approach allows Q-learning to learn from rule-based agent's decisions.
"""

import os
import pickle
import random
import numpy as np
from typing import Optional
from metrics.metrics_tracker import MetricsTracker

# Import rule-based agent's act function
from agent_code.rule_based_agent import callbacks as rule_based_callbacks

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# ===================================================================
# SETUP
# ===================================================================

def setup(self):
    """
    Initialize Q-learning model and rule-based agent for sequential decision making.
    """
    self.name = "Q→Rule-Based Sequential Agent"

    # Load Q-learning model
    model_paths = [
        "agent_code/q_learning_rule_based/my-saved-model.pt",  # Preferred
        "agent_code/q_learning/my-saved-model.pt",             # Fallback
        "my-saved-model.pt",                                    # Local
    ]

    loaded_model = None
    model_source = None

    for model_path in model_paths:
        if os.path.isfile(model_path):
            print(f"📂 Found model at: {model_path}")
            try:
                with open(model_path, "rb") as file:
                    saved = pickle.load(file)
                    if isinstance(saved, dict) and "model" in saved:
                        loaded_model = saved["model"]
                        self.epsilon = saved.get("epsilon", 0.05)
                        self.total_rounds_trained = saved.get("total_rounds_trained", 0)
                    else:
                        loaded_model = saved if isinstance(saved, dict) else {}
                        self.epsilon = 0.05
                        self.total_rounds_trained = 0
                    model_source = model_path
                    print(f"✅ Loaded Q-learning model from: {model_path}")
                    break
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue

    if loaded_model is not None:
        self.q_model = loaded_model
    else:
        print("⚠️  No existing model found. Starting new Q-table.")
        self.q_model = {}
        self.epsilon = 0.05
        self.total_rounds_trained = 0

    # Q-learning hyperparameters
    self.alpha = 0.1
    self.gamma = 0.99
    self.epsilon_decay = 0.998
    self.min_epsilon = 0.01

    # Rule-based agent configuration
    self.use_rule_based = False  # Set to False to use pure Q-learning

    # Initialize rule-based agent's internal state
    rule_based_callbacks.setup(self)

    # State tracking
    self.q_suggestions = []       # Track Q-learning suggestions
    self.rule_based_decisions = [] # Track rule-based decisions

    # Metrics tracking
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="metrics")
    self.episode_counter = 0
    self.episode_active = False
    self.current_step = 0

    print(f"🎮 Q-table size: {len(self.q_model)}")
    print(f"🤖 Rule-based enabled: {self.use_rule_based}")
    print(f"🎯 Strategy: Q-learning suggests → Rule-based decides")


# ===================================================================
# MAIN ACTION SELECTION (Sequential Flow)
# ===================================================================

def act(self, game_state) -> str:
    """
    Sequential decision making:
    1. Q-learning analyzes state and suggests action
    2. Rule-based agent makes final decision
    """
    round_num = game_state.get('round', 0)
    step = game_state.get('step', 0)

    # print(f"\n{'='*60}")
    # print(f"Round {round_num}, Step {step} - Sequential Q→Rule-Based Agent")
    # print(f"{'='*60}")

    # ===================================================================
    # PHASE 1: Q-LEARNING SUGGESTION
    # ===================================================================
    q_action, q_values, q_features = get_q_learning_suggestion(self, game_state)
    # print(f"[Q-LEARNING] Suggested action: {q_action}")
    # print(f"[Q-LEARNING] Q-values: {q_values}")

    # ===================================================================
    # PHASE 2: RULE-BASED FINAL DECISION
    # ===================================================================
    if self.use_rule_based:
        try:
            final_action = rule_based_callbacks.act(self, game_state)
            # print(f"[RULE-BASED] Final decision: {final_action}")

            # Track override vs acceptance
            if final_action == q_action:
                # print(f"[RULE-BASED] ✅ Agreed with Q-learning suggestion")
                ...
            else:
                # print(f"[RULE-BASED] 🔄 Overrode Q-learning: {q_action} → {final_action}")
                ...
        except Exception as e:
            # print(f"[RULE-BASED] ❌ Error, falling back to Q-learning: {e}")
            final_action = q_action
    else:
        final_action = q_action
        # print(f"[Q-LEARNING] Rule-based disabled, using Q-suggestion: {final_action}")

    # ===================================================================
    # PHASE 3: SAFETY VALIDATION
    # ===================================================================
    valid_actions = get_valid_actions(game_state)
    if final_action not in valid_actions:
        # print(f"[SAFETY] ⚠️  Action {final_action} not valid, choosing from: {valid_actions}")
        final_action = valid_actions[0] if valid_actions else 'WAIT'

    # ===================================================================
    # PHASE 4: METRICS TRACKING
    # ===================================================================
    track_episode_metrics(self, game_state, final_action)

    # ===================================================================
    # PHASE 5: STATE TRACKING
    # ===================================================================
    self.q_suggestions.append({'step': step, 'action': q_action, 'q_values': q_values})
    self.rule_based_decisions.append({'step': step, 'action': final_action})

    # print(f"✅ Final action: {final_action}")
    # print(f"{'='*60}\n")

    return final_action


# ===================================================================
# Q-LEARNING SUGGESTION
# ===================================================================

def get_q_learning_suggestion(self, game_state) -> tuple:
    """
    Get Q-learning's suggested action based on learned Q-values.

    Returns:
        (action, q_values_dict, features): Q-learning's suggestion with context
    """
    features = state_to_features(game_state)

    if features is None:
        return ('WAIT', {}, None)

    # Initialize unseen state
    if features not in self.q_model:
        self.q_model[features] = {action: 0.0 for action in ACTIONS}

    q_values = self.q_model[features]

    # Get valid actions
    valid_actions = get_valid_actions(game_state)
    if len(valid_actions) == 0:
        valid_actions = ACTIONS

    # Epsilon-greedy for exploration (only during training)
    if self.train and random.random() < self.epsilon:
        action = np.random.choice(valid_actions)
        # print(f"[Q-LEARNING] Exploring (ε={self.epsilon:.3f})")
    else:
        # Exploitation: choose best valid action
        valid_q = {a: q_values.get(a, 0.0) for a in valid_actions}
        action = max(valid_q, key=valid_q.get) if valid_q else valid_actions[0]
        # print(f"[Q-LEARNING] Exploiting (best Q={valid_q.get(action, 0.0):.2f})")

    return (action, q_values, features)


# ===================================================================
# FEATURE ENGINEERING (from Q-learning agent)
# ===================================================================

def state_to_features(game_state: dict) -> Optional[tuple]:
    """
    Convert game state to feature tuple for Q-learning.
    Uses same features as pure Q-learning agent for compatibility.
    """
    if game_state is None:
        return None

    _, score, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']

    features = []

    # Movement safety (4 directions)
    moves = {
        'UP': (x, y - 1),
        'RIGHT': (x + 1, y),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y)
    }

    for direction in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        nx, ny = moves[direction]
        safety = get_move_safety(nx, ny, field, bombs, explosion_map)
        features.append(safety)

    # Danger level at current position
    danger_level = get_danger_level(x, y, bombs, explosion_map, field)
    features.append(danger_level)

    # Coin navigation
    if len(coins) > 0:
        coin_direction = get_direction_to_nearest(x, y, coins)
        coin_distance = get_distance_bucket(x, y, coins)
        features.append(coin_direction)
        features.append(coin_distance)
    else:
        features.append('no_coin')
        features.append('no_distance')

    # Escape routes available
    escape_count = count_escape_routes(x, y, field, bombs, explosion_map)
    features.append(escape_count)

    # Bomb placement value
    if bomb_available:
        bomb_value = evaluate_bomb_placement(x, y, field, bombs, explosion_map)
        features.append(bomb_value)
    else:
        features.append('no_bomb')

    # Crate proximity
    crate_nearby = count_nearby_crates(x, y, field)
    features.append(crate_nearby)

    return tuple(features)


# ===================================================================
# HELPER FUNCTIONS (Feature extraction - from Q-learning agent)
# ===================================================================

def get_move_safety(x, y, field, bombs, explosion_map):
    """Evaluate safety of moving to position (x, y)."""
    if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1]:
        return 'blocked'
    if field[x, y] == -1 or field[x, y] == 1:
        return 'blocked'
    if explosion_map[x, y] > 0:
        return 'blocked'
    bomb_positions = [pos for pos, _ in bombs]
    if (x, y) in bomb_positions:
        return 'blocked'
    if is_in_danger(x, y, bombs, explosion_map, field):
        return 'risky'
    return 'safe'


def get_danger_level(x, y, bombs, explosion_map, field):
    """Evaluate danger level at current position."""
    if explosion_map[x, y] > 0:
        return 'critical'

    min_bomb_timer = 999
    in_blast_radius = False
    bomb_power = 3

    for (bx, by), timer in bombs:
        if (bx == x and abs(by - y) <= bomb_power) or (by == y and abs(bx - x) <= bomb_power):
            blocked = False
            if bx == x:
                for check_y in range(min(y, by), max(y, by) + 1):
                    if field[x, check_y] == -1:
                        blocked = True
                        break
            else:
                for check_x in range(min(x, bx), max(x, bx) + 1):
                    if field[check_x, y] == -1:
                        blocked = True
                        break

            if not blocked:
                in_blast_radius = True
                min_bomb_timer = min(min_bomb_timer, timer)

    if not in_blast_radius:
        return 'none'
    if min_bomb_timer <= 1:
        return 'critical'
    elif min_bomb_timer <= 2:
        return 'high'
    else:
        return 'low'


def get_distance_bucket(x, y, targets):
    """Categorize distance to nearest target."""
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
    """Count number of safe adjacent tiles."""
    safe_count = 0
    adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    for nx, ny in adjacent:
        if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
            continue
        if field[nx, ny] != 0:
            continue
        if explosion_map[nx, ny] > 0:
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
    """Evaluate if placing a bomb here is valuable."""
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

    crates_hit = 0
    bomb_power = 3

    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        dx, dy = direction
        for dist in range(1, bomb_power + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
                break
            if field[nx, ny] == -1:
                break
            if field[nx, ny] == 1:
                crates_hit += 1
                break

    if crates_hit >= 2:
        return 'good'
    elif crates_hit == 1:
        return 'neutral'
    else:
        return 'bad'


def count_nearby_crates(x, y, field):
    """Count crates in surrounding area."""
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
    """Get direction to nearest target."""
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
    """Check if position is in danger from bombs/explosions."""
    if explosion_map[x, y] > 0:
        return True

    danger_tiles = set()
    bomb_power = 3

    for (bx, by), timer in bombs:
        if timer <= 3:
            danger_tiles.add((bx, by))
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in directions:
                for dist in range(1, bomb_power + 1):
                    nx, ny = bx + dx * dist, by + dy * dist
                    if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
                        break
                    if field[nx, ny] == -1:
                        break
                    danger_tiles.add((nx, ny))
                    if field[nx, ny] == 1:
                        break

    if (x, y) in danger_tiles:
        safe_dirs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in safe_dirs:
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0 and (nx, ny) not in danger_tiles:
                    return False
        return True
    return False


def get_valid_actions(game_state):
    """Returns list of valid actions that don't lead to immediate danger."""
    if game_state is None:
        return ACTIONS

    _, _, bomb_available, (x, y) = game_state['self']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    valid = []

    moves = {
        'UP': (x, y - 1),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y),
        'RIGHT': (x + 1, y)
    }

    for action, (nx, ny) in moves.items():
        if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
            continue
        if field[nx, ny] != 0:
            continue
        if explosion_map[nx, ny] > 0:
            continue
        bomb_positions = [pos for pos, _ in bombs]
        if (nx, ny) in bomb_positions:
            continue
        if is_in_danger(nx, ny, bombs, explosion_map, field):
            continue
        valid.append(action)

    if not is_in_danger(x, y, bombs, explosion_map, field):
        valid.append('WAIT')

    if bomb_available:
        test_bombs = bombs + [((x, y), 4)]
        can_escape = False
        for action, (nx, ny) in moves.items():
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 0:
                    if not is_in_danger(nx, ny, test_bombs, explosion_map, field):
                        can_escape = True
                        break
        if can_escape:
            valid.append('BOMB')

    return valid if len(valid) > 0 else ['WAIT']


# ===================================================================
# METRICS TRACKING
# ===================================================================

def track_episode_metrics(self, game_state, action):
    """Track metrics for both training and play modes."""
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="metrics")

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
        metadata={"mode": "sequential_q_rule_based"}
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
