# callbacks.py - WITH GAMEPLAY METRICS TRACKING

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import settings as s
from metrics.metrics_tracker import MetricsTracker

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
MODEL_PATH = "models/ppo_agent.pth"

BOARD_SHAPE = (s.COLS, s.ROWS)
BOARD_SIZE = BOARD_SHAPE[0] * BOARD_SHAPE[1]
SPATIAL_PLANES = 9
GLOBAL_FEATURE_DIM = 9  # INCREASED: was 6, now 9 (added 3 escape/crate features)
PLANE_VECTOR_SIZE = BOARD_SIZE * SPATIAL_PLANES
FEATURE_VECTOR_SIZE = BOARD_SIZE * SPATIAL_PLANES + GLOBAL_FEATURE_DIM

DIRECTIONS = {
    'UP': (0, -1),
    'RIGHT': (1, 0),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
}

DANGER_THRESHOLD = 0.75  # Increased - was too conservative at 0.6, causing too many masked actions
EPS = 1e-6
MAX_COIN_COUNT = max(1, (s.COLS * s.ROWS) // 4)
MAX_BOMB_COUNT = max(1, s.MAX_AGENTS)


def compute_danger_map(game_state: Optional[dict]) -> np.ndarray:
    if not game_state:
        return np.zeros(BOARD_SHAPE, dtype=np.float32)

    field = np.array(game_state["field"], dtype=np.float32)
    danger = np.zeros_like(field, dtype=np.float32)
    for (x, y), timer in game_state.get("bombs") or []:
        norm_timer = 1.0 / (1.0 + max(0, timer))
        danger[x, y] = max(danger[x, y], norm_timer)
        for dx, dy in DIRECTIONS.values():
            for step in range(1, s.BOMB_POWER + 1):
                nx, ny = x + dx * step, y + dy * step
                if nx < 0 or ny < 0 or nx >= field.shape[0] or ny >= field.shape[1]:
                    break
                danger[nx, ny] = max(danger[nx, ny], norm_timer)
                if field[nx, ny] == -1:
                    break

    explosion_map = game_state.get("explosion_map")
    if explosion_map is not None:
        normalized = np.array(explosion_map, dtype=np.float32) / max(1.0, float(s.EXPLOSION_TIMER))
        danger = np.maximum(danger, np.clip(normalized, 0.0, 1.0))

    return danger


def count_safe_neighbors(
    game_state: dict,
    position: tuple,
    danger_map: np.ndarray,
    danger_threshold: float
) -> int:
    """Count number of immediately adjacent safe tiles."""
    field = np.array(game_state["field"])
    explosion_map = np.array(game_state.get("explosion_map", np.zeros_like(field)), dtype=np.float32)
    bomb_positions = {tuple(pos) for (pos, _) in game_state.get("bombs") or []}

    x, y = position
    safe_count = 0

    for dx, dy in DIRECTIONS.values():
        nx, ny = x + dx, y + dy
        if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
            continue
        if field[nx, ny] != 0:
            continue
        if (nx, ny) in bomb_positions:
            continue
        if danger_map[nx, ny] < danger_threshold and explosion_map[nx, ny] == 0:
            safe_count += 1

    return safe_count


def compute_min_safety_distance(
    game_state: dict,
    position: tuple,
    danger_map: np.ndarray,
    danger_threshold: float,
    max_distance: int = 10
) -> int:
    """
    Compute minimum distance to a safe tile using BFS.
    Returns max_distance if no safe tile found.
    """
    from collections import deque

    field = np.array(game_state["field"])
    explosion_map = np.array(game_state.get("explosion_map", np.zeros_like(field)), dtype=np.float32)
    bomb_positions = {tuple(pos) for (pos, _) in game_state.get("bombs") or []}

    x, y = position

    # Already safe
    if danger_map[x, y] < danger_threshold and explosion_map[x, y] == 0:
        return 0

    queue = deque([(x, y, 0)])
    visited = {(x, y)}

    while queue:
        cx, cy, dist = queue.popleft()

        if dist >= max_distance:
            continue

        for dx, dy in DIRECTIONS.values():
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0:
                continue
            if (nx, ny) in visited:
                continue
            if (nx, ny) in bomb_positions:
                continue

            # Found safe tile?
            if danger_map[nx, ny] < danger_threshold and explosion_map[nx, ny] == 0:
                return dist + 1

            visited.add((nx, ny))
            queue.append((nx, ny, dist + 1))

    return max_distance


def count_crates_in_bomb_range_at_pos(game_state: dict, position: tuple) -> int:
    """Count how many crates would be destroyed by a bomb at given position."""
    field = np.array(game_state["field"])
    x, y = position
    crate_count = 0

    # Check all four directions
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        for step in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * step, y + dy * step
            if nx < 0 or ny < 0 or nx >= field.shape[0] or ny >= field.shape[1]:
                break
            if field[nx, ny] == -1:  # Wall blocks
                break
            if field[nx, ny] == 1:  # Crate
                crate_count += 1

    return crate_count


def can_safely_escape_bomb(
    game_state: dict,
    position: tuple,
    danger_map: np.ndarray,
    danger_threshold: float,
    max_steps: int = 5
) -> bool:
    """
    Check if agent can reach a safe position before bomb explodes.
    Uses BFS to find path to safety within bomb timer.

    CRITICAL: Simulates NEW BOMB danger at position before checking escape!

    Args:
        game_state: Current game state
        position: (x, y) position to check escape from
        danger_map: Pre-computed danger map (existing bombs only)
        danger_threshold: Threshold for what counts as "safe"
        max_steps: Maximum steps to search (bomb timer is 4)

    Returns:
        True if agent can reach safety, False otherwise
    """
    from collections import deque

    field = np.array(game_state["field"])
    bomb_positions = {tuple(pos) for (pos, _) in game_state.get("bombs") or []}
    explosion_map = np.array(game_state.get("explosion_map", np.zeros_like(field)), dtype=np.float32)

    # CRITICAL FIX: Simulate new bomb danger at position
    # Create updated danger map with hypothetical bomb
    simulated_danger = danger_map.copy()
    x, y = position

    # Add danger from new bomb at position
    # Bomb will explode in 4 steps, so danger level = 1.0
    simulated_danger[x, y] = max(simulated_danger[x, y], 1.0)

    # Add blast zone danger in all 4 directions
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        for step in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * step, y + dy * step
            if nx < 0 or ny < 0 or nx >= field.shape[0] or ny >= field.shape[1]:
                break
            simulated_danger[nx, ny] = max(simulated_danger[nx, ny], 1.0)
            if field[nx, ny] == -1:  # Wall blocks blast
                break

    # Now search for escape route using SIMULATED danger
    queue = deque([(x, y, 0)])  # (x, y, steps)
    visited = {(x, y)}

    while queue:
        cx, cy, steps = queue.popleft()

        # Skip starting position (will be dangerous after bomb drop)
        if steps > 0:
            # Found safe tile within time limit?
            # NUCLEAR OPTION: Moderately safe (70% - balance safety and learning)
            if simulated_danger[cx, cy] < (danger_threshold * 0.70) and explosion_map[cx, cy] == 0:
                return True

        # Don't search beyond bomb timer
        if steps >= max_steps:
            continue

        # Explore neighbors
        for dx, dy in DIRECTIONS.values():
            nx, ny = cx + dx, cy + dy

            # Check bounds
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue

            # Check walkable
            if field[nx, ny] != 0:
                continue

            # Check not already visited
            if (nx, ny) in visited:
                continue

            # Check not a bomb position
            if (nx, ny) in bomb_positions:
                continue

            visited.add((nx, ny))
            queue.append((nx, ny, steps + 1))

    return False


def build_action_mask(
    game_state: Optional[dict],
    danger_map: Optional[np.ndarray] = None,
    danger_threshold: float = DANGER_THRESHOLD,
) -> np.ndarray:
    if not game_state or not game_state.get("self"):
        return np.ones(len(ACTIONS), dtype=np.float32)

    if danger_map is None:
        danger_map = compute_danger_map(game_state)

    mask = np.ones(len(ACTIONS), dtype=np.float32)
    field = np.array(game_state["field"], dtype=np.float32)
    explosion_map = np.array(game_state.get("explosion_map", np.zeros_like(field)), dtype=np.float32)
    _, _, bombs_left, (x, y) = game_state["self"]
    bombs_left = int(bombs_left)
    bomb_positions = {tuple(pos) for (pos, _) in game_state.get("bombs") or []}
    others = {
        tuple(other[3])
        for other in (game_state.get("others") or [])
        if other and other[3] is not None
    }

    for action, (dx, dy) in DIRECTIONS.items():
        idx = ACTION_IDX[action]
        nx, ny = x + dx, y + dy
        if nx < 0 or ny < 0 or nx >= field.shape[0] or ny >= field.shape[1]:
            mask[idx] = 0.0
            continue
        if field[nx, ny] != 0:
            mask[idx] = 0.0
            continue
        if (nx, ny) in bomb_positions or (nx, ny) in others:
            mask[idx] = 0.0
            continue
        # Mask both explosions AND high danger zones (but with higher threshold)
        if explosion_map[nx, ny] > 0 or danger_map[nx, ny] >= danger_threshold:
            mask[idx] = 0.0

    # Mask WAIT if in danger zone or actively exploding
    if explosion_map[x, y] > 0 or danger_map[x, y] >= danger_threshold:
        mask[ACTION_IDX['WAIT']] = 0.0

    # Basic bomb masking
    if bombs_left <= 0 or (x, y) in bomb_positions:
        mask[ACTION_IDX['BOMB']] = 0.0
    else:
        # IMPROVED: Check if agent can actually reach safety before bomb explodes
        # Uses BFS to find path to safe zone within bomb timer
        has_escape = can_safely_escape_bomb(
            game_state,
            (x, y),
            danger_map,
            danger_threshold,
            max_steps=4  # Bomb timer in Bomberman
        )
        if not has_escape:
            mask[ACTION_IDX['BOMB']] = 0.0

    if mask.sum() == 0:
        mask.fill(1.0)

    return mask.astype(np.float32)


def build_feature_vector(game_state: Optional[dict]) -> np.ndarray:
    if not game_state or not game_state.get("self"):
        return np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)

    field = np.array(game_state["field"], dtype=np.float32)
    _, _, bombs_left, (x, y) = game_state["self"]
    danger_map = compute_danger_map(game_state)

    walls = (field == -1).astype(np.float32)
    crates = (field == 1).astype(np.float32)
    free = (field == 0).astype(np.float32)

    coins = np.zeros_like(field, dtype=np.float32)
    for cx, cy in game_state.get("coins", []):
        coins[cx, cy] = 1.0

    others = np.zeros_like(field, dtype=np.float32)
    for other in game_state.get("others", []):
        if other and other[3] is not None:
            ox, oy = other[3]
            others[ox, oy] = 1.0

    bombs_map = np.zeros_like(field, dtype=np.float32)
    for (bx, by), timer in game_state.get("bombs") or []:
        bombs_map[bx, by] = max(bombs_map[bx, by], 1.0 / (1.0 + max(0, timer)))

    explosion_map = np.array(game_state.get("explosion_map", np.zeros_like(field)), dtype=np.float32)
    explosion_map = np.clip(explosion_map / max(1.0, float(s.EXPLOSION_TIMER)), 0.0, 1.0)

    self_map = np.zeros_like(field, dtype=np.float32)
    self_map[x, y] = 1.0

    planes = np.stack(
        [walls, crates, free, coins, others, bombs_map, explosion_map, danger_map, self_map],
        axis=0,
    )

    total_coins = len(game_state.get("coins", []))
    bomb_count = len(game_state.get("bombs") or [])

    # NEW: Escape route features (critical for learning safe bombing)
    safe_neighbors = count_safe_neighbors(game_state, (x, y), danger_map, DANGER_THRESHOLD)
    min_safety_dist = compute_min_safety_distance(game_state, (x, y), danger_map, DANGER_THRESHOLD)
    crates_in_range = count_crates_in_bomb_range_at_pos(game_state, (x, y))

    global_features = np.array(
        [
            float(bombs_left > 0),
            danger_map[x, y],
            min(1.0, total_coins / MAX_COIN_COUNT),
            min(1.0, bomb_count / MAX_BOMB_COUNT),
            min(1.0, game_state.get("step", 0) / max(1.0, float(s.MAX_STEPS))),
            min(1.0, game_state.get("round", 0) / 1000.0),
            # NEW FEATURES:
            min(1.0, safe_neighbors / 4.0),  # Normalized by max possible (4 directions)
            min(1.0, min_safety_dist / 10.0),  # Normalized by max search distance
            min(1.0, crates_in_range / 8.0),  # Normalized by reasonable max (2 per direction)
        ],
        dtype=np.float32,
    )

    return np.concatenate([planes.reshape(-1), global_features], axis=0)


# ---------------------------------------------------------------------
# PPO Network (IMPROVED ARCHITECTURE)
# ---------------------------------------------------------------------
class PPONetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=512):  # Increased from 256
        super(PPONetwork, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.spatial_channels = SPATIAL_PLANES
        self.global_dim = GLOBAL_FEATURE_DIM
        assert self.input_dim == PLANE_VECTOR_SIZE + self.global_dim, (
            f"Expected input_dim {PLANE_VECTOR_SIZE + self.global_dim}, got {input_dim}"
        )

        # Improved convolutional encoder with less aggressive pooling
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(self.spatial_channels, 64, kernel_size=3, padding=1),  # More filters
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # More filters
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # Only one pooling layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU(),
        )

        self.cnn_output_dim = self._infer_cnn_out_dim()
        trunk_input_dim = self.cnn_output_dim + self.global_dim

        # Shared trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Value head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _infer_cnn_out_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.spatial_channels, *BOARD_SHAPE)
            out = self.conv_encoder(dummy)
        return out.view(1, -1).size(1)

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")

        spatial = x[:, :PLANE_VECTOR_SIZE]
        global_feats = x[:, PLANE_VECTOR_SIZE:]

        spatial = spatial.view(-1, self.spatial_channels, *BOARD_SHAPE)
        conv_out = self.conv_encoder(spatial)
        conv_out = conv_out.view(conv_out.size(0), -1)

        joint = torch.cat([conv_out, global_feats], dim=1)
        hidden = self.shared_trunk(joint)

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value


# ---------------------------------------------------------------------
# PPO Agent (unchanged - keeping your exact implementation)
# ---------------------------------------------------------------------
class PPOAgent:
    def __init__(
        self,
        input_dim,
        action_dim,
        lr=3e-4,                    # Conservative learning rate
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        update_epochs=4,
        batch_size=1024,            # Increased - better stability
        minibatch_size=256,         # Increased proportionally
        entropy_coef=0.04,          # Increased for better exploration
        max_grad_norm=0.5,
        target_kl=0.015,            # Slightly more conservative
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.minibatch_size = max(1, min(minibatch_size, batch_size))
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.eps = EPS
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
            
        self.model = PPONetwork(input_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.memory = {
            "obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "action_masks": [],
        }

    def featurize(self, game_state: dict) -> np.ndarray:
        features = build_feature_vector(game_state)
        if features.shape[0] != self.input_dim:
            raise ValueError(f"Feature dimension {features.shape[0]} does not match expected {self.input_dim}")
        return features

    def select_action(self, obs, action_mask=None):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if obs_tensor.shape[1] != self.input_dim:
            raise ValueError(f"Observation dimension {obs_tensor.shape[1]} doesn't match model input_dim {self.input_dim}")
        
        mask_np = self._ensure_action_mask(action_mask)
        mask_tensor = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)

        logits, value = self.model(obs_tensor)
        masked_logits = self._apply_action_mask(logits, mask_tensor)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return int(action.item()), float(log_prob.item()), float(value.item()), mask_np

    def store_transition(self, obs, action, log_prob, reward, value, done, mask):
        self.memory["obs"].append(obs)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(reward)
        self.memory["values"].append(value)
        self.memory["dones"].append(done)
        self.memory["action_masks"].append(self._ensure_action_mask(mask))

    def compute_advantages(self, next_value=0):
        rewards = np.array(self.memory["rewards"], dtype=np.float32)
        values = np.array(self.memory["values"] + [next_value], dtype=np.float32)
        dones = np.array(self.memory["dones"], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self):
        num_samples = len(self.memory["obs"])
        if num_samples == 0:
            return {"updated": False, "num_samples": 0}
        if num_samples < self.batch_size:
            return {"updated": False, "num_samples": num_samples}

        obs = torch.tensor(np.vstack(self.memory["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32, device=self.device)
        action_masks = torch.tensor(np.vstack(self.memory["action_masks"]), dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_advantages()
        advantages = np.array(advantages, dtype=np.float32)
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std < 1e-8:
            advantages = advantages - adv_mean
        else:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        indices = np.arange(num_samples)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0
        kl_exceeded = False

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, num_samples)
                batch_idx = indices[start:end]
                batch_idx_tensor = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)

                mb_obs = obs[batch_idx_tensor]
                mb_actions = actions[batch_idx_tensor]
                mb_old_log_probs = old_log_probs[batch_idx_tensor]
                mb_advantages = advantages_tensor[batch_idx_tensor]
                mb_returns = returns[batch_idx_tensor]
                mb_masks = action_masks[batch_idx_tensor]

                logits, values = self.model(mb_obs)
                masked_logits = self._apply_action_mask(logits, mb_masks)
                dist = Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (mb_returns - values.squeeze(-1)).pow(2).mean()
                loss = actor_loss + critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kl = torch.mean(mb_old_log_probs - new_log_probs).abs().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl
                num_updates += 1

                if approx_kl > self.target_kl:
                    kl_exceeded = True
                    break
            if kl_exceeded:
                break

        self.clear_memory()
        if num_updates == 0:
            return {"updated": False, "num_samples": num_samples}

        return {
            "updated": True,
            "num_samples": num_samples,
            "avg_actor_loss": total_actor_loss / num_updates,
            "avg_critic_loss": total_critic_loss / num_updates,
            "avg_entropy": total_entropy / num_updates,
            "avg_kl": total_kl / num_updates,
        }

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'batch_size': self.batch_size,
            'minibatch_size': self.minibatch_size,
        }
        torch.save(checkpoint, path)

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                saved_input_dim = checkpoint.get('input_dim', None)
                
                if saved_input_dim is not None and saved_input_dim != self.input_dim:
                    print(f"Warning: Saved model input_dim ({saved_input_dim}) != current input_dim ({self.input_dim})")
                    print("Cannot load model with mismatched dimensions. Using random initialization.")
                    return False
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.batch_size = checkpoint.get('batch_size', self.batch_size)
                self.minibatch_size = checkpoint.get('minibatch_size', self.minibatch_size)
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"Successfully loaded model from {path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []

    def _ensure_action_mask(self, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return np.ones(self.action_dim, dtype=np.float32)
        mask_np = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask_np.shape[0] != self.action_dim:
            raise ValueError(f"Mask dimension {mask_np.shape[0]} does not match action_dim {self.action_dim}")
        if mask_np.sum() <= 0:
            mask_np = np.ones_like(mask_np)
        return mask_np

    def _apply_action_mask(self, logits: torch.Tensor, mask_tensor: torch.Tensor) -> torch.Tensor:
        # Apply mask by setting invalid actions to very large negative value
        # This ensures they're never selected (unlike log(eps) which is only -13.8)
        masked_logits = torch.where(
            mask_tensor > 0,
            logits,
            torch.full_like(logits, -1e10)  # Effectively impossible to select
        )
        return masked_logits


# ---------------------------------------------------------------------
# Bomberman Callbacks — WITH GAMEPLAY METRICS
# ---------------------------------------------------------------------

def setup(self):
    """Called once before the first game starts."""
    FEATURE_DIM = FEATURE_VECTOR_SIZE
    self.name = "PPO"
    self.train_agent = PPOAgent(
        input_dim=FEATURE_DIM,
        action_dim=len(ACTIONS),
        lr=3e-4,                    # Conservative learning rate
        batch_size=1024,            # Increased - need more samples for stable updates
        minibatch_size=256,         # Increased proportionally
        update_epochs=4,
        clip_eps=0.2,
        entropy_coef=0.04,          # Increased for more exploration early on
        target_kl=0.015,            # Slightly more conservative
    )
    
    # Load model
    if os.path.exists(MODEL_PATH):
        success = self.train_agent.load(MODEL_PATH)
        if success:
            self.logger.info(f"Loaded pretrained PPO model from {MODEL_PATH}")
        else:
            self.logger.info(f"Starting with fresh PPO model (input_dim={FEATURE_DIM})")
    else:
        self.logger.info(f"No pretrained model found. Initialized new PPOAgent with input_dim={FEATURE_DIM}")
    
    # Initialize state tracking
    self.last_obs = None
    self.last_action = None
    self.last_log_prob = None
    self.last_value = None
    self.last_action_mask = np.ones(len(ACTIONS), dtype=np.float32)
    self.post_bomb_origin = None
    self.post_bomb_timer = 0
    
    # Initialize metrics tracker
    self.metrics_tracker = MetricsTracker(
        agent_name=self.name,
        save_dir="evaluation_metrics"  # Use separate folder for evaluation
    )
    self.episode_counter = 0

    # Track if episode is active
    self.episode_active = False
    self.current_step = 0
    
    self.logger.info("PPO callbacks.setup() finished with metrics tracking.")


def act(self, game_state):
    """Called each game step to select an action."""
    
    # Safety check
    if not hasattr(self, 'train_agent') or self.train_agent is None:
        self.logger.error("train_agent not initialized! Calling setup()...")
        setup(self)
    
    # Initialize metrics tracker if missing
    if not hasattr(self, 'metrics_tracker'):
        self.metrics_tracker = MetricsTracker(
            agent_name=self.name,
            save_dir="evaluation_metrics"  # Use separate folder for evaluation
        )
        self.episode_counter = 0
        self.episode_active = False
        self.current_step = 0
    
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
    
    # =========================================================================
    # CHECK FOR EPISODE END (AGENT ELIMINATED OR GAME OVER)
    # =========================================================================
    if game_state and self.episode_active:
        # Check if this agent is dead/eliminated
        agent_alive = True
        
        # Method 1: Check if 'self' exists in game_state
        if 'self' not in game_state or game_state['self'] is None:
            agent_alive = False
        
        # Method 2: Check for explicit dead flag (if your framework has it)
        if 'dead' in game_state and game_state.get('dead', False):
            agent_alive = False
        
        # Method 3: Check game state for end conditions
        # The game might set 'round_finished' or similar
        if game_state.get('round_finished', False):
            agent_alive = False  # Treat as end
        
        # If agent died, end the episode
        if not agent_alive and self.episode_active:
            _end_gameplay_episode(self, game_state, events=[], died=True)
    
    # Increment step counter
    if game_state:
        self.current_step = game_state.get('step', self.current_step + 1)
    
    # =========================================================================
    # SELECT ACTION
    # =========================================================================
    obs = self.train_agent.featurize(game_state)
    danger_map = compute_danger_map(game_state)
    action_mask = build_action_mask(game_state, danger_map)
    
    try:
        action_idx, log_prob, value, used_mask = self.train_agent.select_action(obs, action_mask=action_mask)
    except Exception as e:
        self.logger.error(f"Error selecting action: {e}")
        action_idx = np.random.randint(0, len(ACTIONS))
        log_prob = 0.0
        value = 0.0
        used_mask = np.ones(len(ACTIONS), dtype=np.float32)
    
    # Store for potential training use
    self.last_obs = obs
    self.last_action = action_idx
    self.last_log_prob = log_prob
    self.last_value = value
    self.last_action_mask = used_mask
    
    action = ACTIONS[action_idx]
    
    # =========================================================================
    # TRACK ACTION
    # =========================================================================
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_action(
            action,
            is_valid=used_mask[ACTION_IDX[action]] > 0.0
        )
    
    return action


def _end_gameplay_episode(self, game_state, events, died=False):
    """
    Helper function to end episode during gameplay.

    Args:
        game_state: Final game state
        events: List of events that occurred (should be from end_of_round)
        died: Whether agent died/was eliminated
    """
    if not hasattr(self, 'metrics_tracker') or not self.episode_active:
        return

    if not self.metrics_tracker.current_episode:
        return

    # Track final events (these only happen once at episode end)
    # Use simplified rewards for gameplay tracking
    GAMEPLAY_EVENT_REWARDS = {
        'COIN_COLLECTED': 10,
        'KILLED_OPPONENT': 50,
        'KILLED_SELF': -100,
        'GOT_KILLED': -50,
        'CRATE_DESTROYED': 5,
        'INVALID_ACTION': -5,
        'SURVIVED_ROUND': 30,
    }

    for event in events:
        reward = GAMEPLAY_EVENT_REWARDS.get(event, 0)
        self.metrics_tracker.record_event(event, reward=reward)
        
    # Determine outcome
    won = False
    rank = 4  # Default to last place
    
    if not died:
        # Agent survived - check if won
        if 'others' in game_state and game_state['others']:
            alive_opponents = sum(1 for o in game_state['others'] if o is not None)
            won = (alive_opponents == 0)
            rank = 1 if won else 2
        else:
            # No opponent info, assume won if survived
            won = True
            rank = 1
    else:
        # Agent died
        won = False
        if 'others' in game_state and game_state['others']:
            alive_opponents = sum(1 for o in game_state['others'] if o is not None)
            rank = alive_opponents + 2  # Finished below all alive opponents
        else:
            rank = 4
    
    survival_steps = self.current_step
    total_steps = game_state.get('step', survival_steps) if game_state else survival_steps
    
    # End episode
    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=survival_steps,
        total_steps=total_steps,
        metadata={
            'mode': 'gameplay',
            'died': died,
            'episode': self.episode_counter
        }
    )
    
    self.episode_active = False
    self.episode_counter += 1

    # Note: end_episode already called above at line 716 - DO NOT call again!
    # Metrics are saved automatically
    print(f"Ended gameplay episode: won={won}, rank={rank}, steps={survival_steps}")


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each round during gameplay.
    This is ONLY called in play mode, not during training.
    """
    # If episode is still active, end it now
    if hasattr(self, 'episode_active') and self.episode_active:
        _end_gameplay_episode(self, last_game_state, events, died=False)
    
    # Log final events if you want to track them
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        # This shouldn't happen since we ended the episode above,
        # but just in case there's a timing issue
        self.logger.warning("Episode still active in end_of_round - ending now")
        _end_gameplay_episode(self, last_game_state, events, died=False)


# =============================================================================
# OPTIONAL: Add event tracking during gameplay
# =============================================================================

def game_events_occurred(self, old_game_state: dict, self_action: str,
                        new_game_state: dict, events: list):
    """
    Optional: Track events during gameplay (not just training).
    Only available if you're running with --scenario or certain modes.

    Note: This callback is typically NOT called during regular gameplay,
    only during training mode. During training, train.py handles event tracking.
    This function is here for completeness but may not be used.

    WARNING: If this gets called during training, it will cause DOUBLE COUNTING
    because train.py already tracks events!
    """
    # DO NOTHING - event tracking is handled in train.py during training
    # and in _end_gameplay_episode() for gameplay mode
    pass
