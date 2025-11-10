import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_agent.pth")

class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = x.reshape(1)

        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        self.var = (self.var * self.count + batch_var * batch_count +
                    delta**2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class SimplifiedCNN_Network(nn.Module):
    def __init__(self, action_dim, hidden_dim=512):  # Increased from 128 to 512
        super(SimplifiedCNN_Network, self).__init__()

        # IMPROVED CNN: Better spatial preservation
        # 17x17 -> 9x9 -> 5x5 (keeps more detail than 4x4)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1)),   # 17->9
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),  # 9->5
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)), # 5->5 (preserve!)
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_feature_dim = 128 * 5 * 5  # 3200 features (was 1024)
        scalar_dim = 13
        combined_dim = cnn_feature_dim + scalar_dim

        # IMPROVED SHARED LAYERS: Deeper network with more capacity
        self.fc_shared = nn.Sequential(
            layer_init(nn.Linear(combined_dim, hidden_dim), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim), std=np.sqrt(2)),  # Additional layer
            nn.ReLU(),
        )

        # IMPROVED ACTOR: Larger hidden layer for better policy expressiveness
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 256), std=np.sqrt(2)),  # 128->256
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01)
        )

        # IMPROVED CRITIC: Larger hidden layer for better value estimates
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 256), std=np.sqrt(2)),  # 128->256
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, x_spatial, x_scalar, action_mask=None):
        cnn_features = self.cnn(x_spatial)
        combined = torch.cat([cnn_features, x_scalar], dim=1)
        features = self.fc_shared(combined)

        logits = self.actor(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        probs = torch.softmax(logits, dim=-1)

        value = self.critic(features)
        return probs, value


class PPOAgent:
    def __init__(self, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.1,
                 update_epochs=10, entropy_coef_start=0.15, entropy_coef_end=0.01,
                 entropy_decay=0.999, episodes_per_update=10, target_kl=0.04,
                 value_coef_start=0.8, value_coef_end=0.8, value_coef_decay=1,
                 aux_loss_coef_start=1.0, aux_loss_coef_end=0.0, aux_loss_decay=0.999):

        import logging
        self.logger = logging.getLogger(__name__)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.target_kl = target_kl

        self.entropy_coef = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay = entropy_decay

        self.value_coef = value_coef_start
        self.value_coef_end = value_coef_end
        self.value_coef_decay = value_coef_decay

        self.aux_loss_coef = aux_loss_coef_start
        self.aux_loss_coef_end = aux_loss_coef_end
        self.aux_loss_coef_decay = aux_loss_decay

        self.training_steps = 0
        self.episodes_per_update = episodes_per_update
        self.episodes_since_update = 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # CRITICAL FIX: Use 512 hidden dims (was hardcoded to 128, overriding default!)
        self.model = SimplifiedCNN_Network(action_dim, hidden_dim=512).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        self.initial_lr = lr

        self.reward_normalizer = RunningMeanStd()

        self.memory = {
            "spatial_obs": [],
            "scalar_obs": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "action_masks": [],
            "expert_actions": []
        }

        self.episode_metrics = {
            'returns': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'kl_divs': [],
            'grad_norms': [],
            'learning_rates': [],
            'aux_losses': [],
        }

        self.own_bomb_positions = set()

    def _compute_bomb_value(self, game_state, x, y, field, explosion_map, own_bomb_map):
        try:
            if game_state["self"][2] == 0:
                return -1.0

            bomb_positions = set([(bx, by) for (bx, by), _ in game_state["bombs"]])
            if (x, y) in bomb_positions:
                return -1.0

            crates_count = 0
            for dx in range(-3, 4):
                nx = x + dx
                if 0 <= nx < 17 and field[nx, y] != -1:
                    if field[nx, y] == 1:
                        crates_count += 1
                    if field[nx, y] == -1:
                        break

            for dy in range(-3, 4):
                ny = y + dy
                if 0 <= ny < 17 and field[x, ny] != -1:
                    if field[x, ny] == 1:
                        crates_count += 1
                    if field[x, ny] == -1:
                        break

            opponent_bonus = 0
            for _, _, _, (ox, oy) in game_state["others"]:
                if abs(ox - x) + abs(oy - y) <= 3:
                    opponent_bonus = 5
                    break

            try:
                from . import train as train_module
                escape_routes = train_module.count_escape_routes(game_state, x, y)
                if escape_routes == 0:
                    return -1.0
            except:
                free_adjacent = sum([1 for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]
                                   if (0 <= x+dx < 17 and 0 <= y+dy < 17 and
                                       field[x+dx, y+dy] == 0 and
                                       explosion_map[x+dx, y+dy] == 0 and
                                       own_bomb_map[x+dx, y+dy] == 0)])
                if free_adjacent == 0:
                    return -1.0

            if crates_count > 0 or opponent_bonus > 0:
                return min(1.0, (crates_count + opponent_bonus) / 10.0)

            return 0.0

        except Exception:
            return 0.0

    def featurize(self, game_state: dict):
        if game_state is None:
            return np.zeros((9, 17, 17), dtype=np.float32), np.zeros(13, dtype=np.float32)

        try:
            _, score, bombs_left, (x, y) = game_state["self"]

            field = np.array(game_state["field"], dtype=np.float32)
            explosion_map = np.array(game_state["explosion_map"], dtype=np.float32)
            own_bomb_map = np.zeros((17, 17), dtype=np.float32)
            enemy_bomb_map = np.zeros((17, 17), dtype=np.float32)

            if not hasattr(self, 'own_bomb_positions'):
                self.own_bomb_positions = set()

            for (bx, by), timer in game_state["bombs"]:
                is_own_bomb = (bx, by) in self.own_bomb_positions
                target_map = own_bomb_map if is_own_bomb else enemy_bomb_map
                urgency = np.exp(-timer / 2.0)
                target_map[bx, by] = urgency

                for dx in range(1, 4):
                    nx = bx + dx
                    if not (0 <= nx < 17) or field[nx, by] == -1:
                        break
                    target_map[nx, by] = max(target_map[nx, by], urgency)

                for dx in range(1, 4):
                    nx = bx - dx
                    if not (0 <= nx < 17) or field[nx, by] == -1:
                        break
                    target_map[nx, by] = max(target_map[nx, by], urgency)

                for dy in range(1, 4):
                    ny = by + dy
                    if not (0 <= ny < 17) or field[bx, ny] == -1:
                        break
                    target_map[bx, ny] = max(target_map[bx, ny], urgency)

                for dy in range(1, 4):
                    ny = by - dy
                    if not (0 <= ny < 17) or field[bx, ny] == -1:
                        break
                    target_map[bx, ny] = max(target_map[bx, ny], urgency)

            coin_map = np.zeros((17, 17), dtype=np.float32)
            for cx, cy in game_state["coins"]:
                coin_map[cx, cy] = 1.0

            agents_map = np.zeros((17, 17), dtype=np.float32)
            for _, _, _, (ox, oy) in game_state["others"]:
                agents_map[ox, oy] = 1.0

            self_map = np.zeros((17, 17), dtype=np.float32)
            self_map[x, y] = 1.0

            escape_quality_map = np.zeros((17, 17), dtype=np.float32)
            # Only compute for nearby positions
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 17 and 0 <= ny < 17 and field[nx, ny] == 0:
                        free_adjacent = 0
                        for ddx, ddy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nnx, nny = nx + ddx, ny + ddy
                            if (0 <= nnx < 17 and 0 <= nny < 17 and
                                field[nnx, nny] == 0 and
                                explosion_map[nnx, nny] == 0 and
                                own_bomb_map[nnx, nny] < 0.25 and
                                enemy_bomb_map[nnx, nny] < 0.25):
                                free_adjacent += 1
                        escape_quality_map[nx, ny] = free_adjacent / 4.0

            crate_destruction_map = np.zeros((17, 17), dtype=np.float32)
            for px in range(17):
                for py in range(17):
                    if field[px, py] == 0:
                        crates_count = 0
                        for dx in range(1, 4):
                            nx = px + dx
                            if not (0 <= nx < 17) or field[nx, py] == -1:
                                break
                            if field[nx, py] == 1:
                                crates_count += 1
                        for dx in range(1, 4):
                            nx = px - dx
                            if not (0 <= nx < 17) or field[nx, py] == -1:
                                break
                            if field[nx, py] == 1:
                                crates_count += 1
                        for dy in range(1, 4):
                            ny = py + dy
                            if not (0 <= ny < 17) or field[px, ny] == -1:
                                break
                            if field[px, ny] == 1:
                                crates_count += 1
                        for dy in range(1, 4):
                            ny = py - dy
                            if not (0 <= ny < 17) or field[px, ny] == -1:
                                break
                            if field[px, ny] == 1:
                                crates_count += 1
                        crate_destruction_map[px, py] = min(1.0, crates_count / 10.0)

            spatial = np.stack([field, explosion_map, own_bomb_map, enemy_bomb_map,
                               coin_map, agents_map, self_map, escape_quality_map,
                               crate_destruction_map])

            pos_x = x / 16.0
            pos_y = y / 16.0
            bombs_available = float(bombs_left)
            normalized_score = score / 100.0
            in_danger = 1.0 if explosion_map[x, y] > 0 or own_bomb_map[x, y] > 0 or enemy_bomb_map[x, y] > 0 else 0.0

            if len(game_state["coins"]) > 0:
                coin_distances = [abs(cx - x) + abs(cy - y) for cx, cy in game_state["coins"]]
                min_coin_dist = min(coin_distances) / 32.0
            else:
                min_coin_dist = 1.0

            if len(game_state["others"]) > 0:
                opponent_distances = [abs(ox - x) + abs(oy - y) for _, _, _, (ox, oy) in game_state["others"]]
                min_opponent_dist = min(opponent_distances) / 32.0
            else:
                min_opponent_dist = 1.0

            if len(game_state["bombs"]) > 0:
                bomb_distances = [abs(bx - x) + abs(by - y) for (bx, by), _ in game_state["bombs"]]
                min_bomb_dist = min(bomb_distances) / 32.0 if bomb_distances else 1.0
            else:
                min_bomb_dist = 1.0

            adjacent_safe = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < 17 and 0 <= ny < 17 and
                    field[nx, ny] == 0 and
                    explosion_map[nx, ny] == 0 and
                    own_bomb_map[nx, ny] == 0):
                    adjacent_safe += 1
            adjacent_safe_norm = adjacent_safe / 4.0

            num_coins = len(game_state["coins"]) / 10.0
            num_opponents = len(game_state["others"]) / 3.0
            bomb_value = self._compute_bomb_value(game_state, x, y, field, explosion_map, own_bomb_map)
            num_crates = np.sum(field == 1)
            remaining_objectives = (num_coins * 10.0 + num_crates / 3.0) / 100.0

            scalar = np.array([
                pos_x, pos_y, bombs_available, normalized_score, in_danger,
                min_coin_dist, min_opponent_dist, min_bomb_dist, adjacent_safe_norm,
                num_coins, num_opponents, bomb_value, remaining_objectives
            ], dtype=np.float32)

            return spatial, scalar

        except Exception as e:
            print(f"Error in featurize: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((9, 17, 17), dtype=np.float32), np.zeros(13, dtype=np.float32)

    def get_action_mask(self, game_state: dict) -> torch.Tensor:
        if game_state is None:
            return torch.ones(6, dtype=torch.bool)

        try:
            _, _, bombs_left, (x, y) = game_state["self"]
            field = game_state["field"]
            explosion_map = game_state["explosion_map"]
            bomb_positions = set([(bx, by) for (bx, by), _ in game_state["bombs"]])
            other_positions = set([(ox, oy) for _, _, _, (ox, oy) in game_state["others"]])

            mask = torch.ones(6, dtype=torch.bool)

            def is_valid_move(nx, ny):
                if not (0 <= nx < 17 and 0 <= ny < 17):
                    return False
                if field[nx, ny] != 0 or explosion_map[nx, ny] > 0:
                    return False
                if (nx, ny) in bomb_positions or (nx, ny) in other_positions:
                    return False
                return True

            mask[0] = is_valid_move(x, y-1)
            mask[1] = is_valid_move(x+1, y)
            mask[2] = is_valid_move(x, y+1)
            mask[3] = is_valid_move(x-1, y)
            mask[4] = True

            mask[5] = False
            if bombs_left > 0 and (x, y) not in bomb_positions:
                from . import train as train_module
                escape_routes = train_module.count_escape_routes(game_state, x, y)
                if escape_routes >= 1:
                    mask[5] = True

            if not mask.any():
                mask[4] = True

            return mask

        except Exception:
            return torch.ones(6, dtype=torch.bool)

    def select_action(self, spatial_obs, scalar_obs, game_state=None):
        spatial_tensor = torch.tensor(spatial_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        scalar_tensor = torch.tensor(scalar_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        if game_state is not None:
            action_mask = self.get_action_mask(game_state).unsqueeze(0).to(self.device)
        else:
            action_mask = None

        probs, value = self.model(spatial_tensor, scalar_tensor, action_mask=action_mask)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value.item()), action_mask

    def store_transition(self, spatial_obs, scalar_obs, action, log_prob, reward, value, done, action_mask=None, expert_action=None):
        self.reward_normalizer.update([reward])
        normalized_reward = self.reward_normalizer.normalize(reward)

        self.memory["spatial_obs"].append(spatial_obs)
        self.memory["scalar_obs"].append(scalar_obs)
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        self.memory["rewards"].append(normalized_reward)
        self.memory["values"].append(value)
        self.memory["dones"].append(done)
        if action_mask is not None:
            self.memory["action_masks"].append(action_mask.cpu().numpy())
        self.memory["expert_actions"].append(expert_action if expert_action is not None else -1)

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
        if len(self.memory["spatial_obs"]) < 32:
            # Clear memory even if we skip the update to prevent stale data accumulation
            for k in self.memory.keys():
                self.memory[k] = []
            return

        spatial_obs = torch.tensor(np.array(self.memory["spatial_obs"]), dtype=torch.float32).to(self.device)
        scalar_obs = torch.tensor(np.array(self.memory["scalar_obs"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.memory["actions"], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.memory["log_probs"], dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.memory["values"], dtype=torch.float32).to(self.device)
        expert_actions = torch.tensor(self.memory["expert_actions"], dtype=torch.long).to(self.device)

        if len(self.memory["action_masks"]) > 0:
            action_masks = torch.tensor(np.array(self.memory["action_masks"]), dtype=torch.bool).to(self.device)
            action_masks = action_masks.squeeze(1)
        else:
            action_masks = None

        advantages, returns = self.compute_advantages()
        # Improved advantage normalization with stability checks
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        # Only normalize if std is meaningful (avoid division by near-zero)
        if adv_std > 1e-4:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # If advantages are nearly constant, just center them
            advantages = advantages - adv_mean
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        all_epoch_metrics = {
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'kl_divs': [],
            'aux_losses': []
        }

        batch_size = 32
        num_samples = len(spatial_obs)

        for epoch in range(self.update_epochs):
            epoch_kl_divs = []
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_entropies = []
            epoch_aux_losses = []

            indices = np.random.permutation(num_samples)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_spatial = spatial_obs[batch_indices]
                batch_scalar = scalar_obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_action_masks = action_masks[batch_indices] if action_masks is not None else None
                batch_expert_actions = expert_actions[batch_indices]

                probs, values = self.model(batch_spatial, batch_scalar, action_mask=batch_action_masks)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                values_clipped = batch_old_values + torch.clamp(
                    values.squeeze() - batch_old_values, -self.clip_eps, self.clip_eps)
                value_loss_unclipped = (values.squeeze() - batch_returns).pow(2)
                value_loss_clipped = (values_clipped - batch_returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                aux_loss = 0.0
                if self.aux_loss_coef > 0:
                    valid_expert_mask = batch_expert_actions >= 0
                    if valid_expert_mask.any():
                        filtered_probs = probs[valid_expert_mask]
                        filtered_expert_actions = batch_expert_actions[valid_expert_mask]
                        filtered_dist = Categorical(filtered_probs)
                        expert_log_probs = filtered_dist.log_prob(filtered_expert_actions)
                        aux_loss = -expert_log_probs.mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy + self.aux_loss_coef * aux_loss

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean().item()
                    epoch_kl_divs.append(kl_div)
                    epoch_actor_losses.append(actor_loss.item())
                    epoch_critic_losses.append(critic_loss.item())
                    epoch_entropies.append(entropy.item())
                    epoch_aux_losses.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)

            epoch_avg_kl = np.mean(epoch_kl_divs)
            all_epoch_metrics['kl_divs'].append(epoch_avg_kl)
            all_epoch_metrics['actor_losses'].extend(epoch_actor_losses)
            all_epoch_metrics['critic_losses'].extend(epoch_critic_losses)
            all_epoch_metrics['entropies'].extend(epoch_entropies)
            all_epoch_metrics['aux_losses'].extend(epoch_aux_losses)

            # KL divergence early stopping
            if abs(epoch_avg_kl) > self.target_kl:
                self.logger.info(f"Early stopping at epoch {epoch+1}/{self.update_epochs} (KL={epoch_avg_kl:.4f} > target={self.target_kl:.4f})")
                break

        self.episode_metrics['actor_losses'].append(np.mean(all_epoch_metrics['actor_losses']))
        self.episode_metrics['critic_losses'].append(np.mean(all_epoch_metrics['critic_losses']))
        self.episode_metrics['entropies'].append(np.mean(all_epoch_metrics['entropies']))
        self.episode_metrics['kl_divs'].append(np.mean(all_epoch_metrics['kl_divs']))
        self.episode_metrics['aux_losses'].append(np.mean(all_epoch_metrics['aux_losses']))
        self.episode_metrics['grad_norms'].append(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)
        self.episode_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        self.episode_metrics['returns'].append(float(np.mean(returns)))

        # Limit memory growth: keep only last 1000 metrics entries
        max_metrics = 1000
        for key in self.episode_metrics.keys():
            if len(self.episode_metrics[key]) > max_metrics:
                self.episode_metrics[key] = self.episode_metrics[key][-max_metrics:]

        for k in self.memory.keys():
            self.memory[k] = []

        self.entropy_coef = max(self.entropy_coef_end, self.entropy_coef * self.entropy_decay)
        self.value_coef = max(self.value_coef_end, self.value_coef * self.value_coef_decay)
        self.aux_loss_coef = max(self.aux_loss_coef_end, self.aux_loss_coef * self.aux_loss_coef_decay)
        self.training_steps += 1

    def save(self, path=MODEL_PATH, round_counter=None, metrics_counters=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'aux_loss_coef': self.aux_loss_coef,
            'training_steps': self.training_steps,
            'episodes_since_update': self.episodes_since_update,
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_var': self.reward_normalizer.var,
            'reward_normalizer_count': self.reward_normalizer.count,
            'episode_metrics': self.episode_metrics,
            'round_counter': round_counter if round_counter is not None else 0,
            'metrics_counters': metrics_counters if metrics_counters is not None else {},
        }
        torch.save(checkpoint, path)

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False, 0, {}

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            round_counter = 0
            metrics_counters = {}

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # No scheduler loading - keeping LR constant
                if 'scheduler_state_dict' in checkpoint:
                    print("Note: Scheduler state found in checkpoint but ignored (LR kept constant)")

                if 'entropy_coef' in checkpoint:
                    self.entropy_coef = checkpoint['entropy_coef']
                if 'value_coef' in checkpoint:
                    self.value_coef = checkpoint['value_coef']
                if 'aux_loss_coef' in checkpoint:
                    self.aux_loss_coef = checkpoint['aux_loss_coef']
                if 'training_steps' in checkpoint:
                    self.training_steps = checkpoint['training_steps']
                if 'episodes_since_update' in checkpoint:
                    self.episodes_since_update = checkpoint['episodes_since_update']

                if 'reward_normalizer_mean' in checkpoint:
                    self.reward_normalizer.mean = checkpoint['reward_normalizer_mean']
                    self.reward_normalizer.var = checkpoint['reward_normalizer_var']
                    self.reward_normalizer.count = checkpoint['reward_normalizer_count']

                if 'episode_metrics' in checkpoint:
                    self.episode_metrics = checkpoint['episode_metrics']

                if 'round_counter' in checkpoint:
                    round_counter = checkpoint['round_counter'] + 1

                if 'metrics_counters' in checkpoint:
                    metrics_counters = checkpoint['metrics_counters']
            else:
                self.model.load_state_dict(checkpoint)

            print(f"Loaded model from {path} (entropy={self.entropy_coef:.4f}, aux_coef={self.aux_loss_coef:.4f}, steps={self.training_steps}, round={round_counter})")
            return True, round_counter, metrics_counters

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False, 0, {}


def setup(self):
    self.name = "PPO"
    self.train_agent = PPOAgent(action_dim=len(ACTIONS))

    # Load checkpoint and restore round_counter and metrics counters if available
    self.round_counter = 0
    if os.path.exists(MODEL_PATH):
        success, restored_round, restored_counters = self.train_agent.load(MODEL_PATH)
        if success:
            self.round_counter = restored_round
            # Store restored counters for use by setup_training
            self.restored_metrics_counters = restored_counters

    self.last_spatial_obs = None
    self.last_scalar_obs = None
    self.last_action = None
    self.last_log_prob = None
    self.last_value = None
    self.last_action_mask = None
    self.episode_active = False
    self.current_step = 0
    self.total_training_rounds = 1000
    self.rule_based_exploration_threshold = int(0.3 * self.total_training_rounds)


def look_for_targets(free_space, start, targets):
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def _rule_based_heuristic_action(self, game_state):
    from collections import deque

    try:
        # Initialize history tracking if not exists
        if not hasattr(self, 'heuristic_bomb_history'):
            self.heuristic_bomb_history = deque([], 5)
            self.heuristic_coordinate_history = deque([], 20)
            self.heuristic_ignore_others_timer = 0
            self.heuristic_current_round = 0

        # Check if we are in a different round
        if game_state["round"] != self.heuristic_current_round:
            self.heuristic_bomb_history = deque([], 5)
            self.heuristic_coordinate_history = deque([], 20)
            self.heuristic_ignore_others_timer = 0
            self.heuristic_current_round = game_state["round"]

        # Gather information about the game state
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)

        # If agent has been in the same location three times recently, it's a loop
        if self.heuristic_coordinate_history.count((x, y)) > 2:
            self.heuristic_ignore_others_timer = 5
        else:
            self.heuristic_ignore_others_timer -= 1
        self.heuristic_coordinate_history.append((x, y))

        # Check which moves make sense at all
        directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_tiles, valid_actions = [], []
        for d in directions:
            if ((arena[d] == 0) and
                    (game_state['explosion_map'][d] <= 1) and
                    (bomb_map[d] > 0) and
                    (not d in others) and
                    (not d in bomb_xys)):
                valid_tiles.append(d)
        if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
        if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
        if (x, y - 1) in valid_tiles: valid_actions.append('UP')
        if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
        if (x, y) in valid_tiles: valid_actions.append('WAIT')
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if (bombs_left > 0) and (x, y) not in self.heuristic_bomb_history:
            valid_actions.append('BOMB')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        random.shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 0)
                     and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in range(1, 16) for y in range(1, 16) if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        # Add other agents as targets if in hunting mode or no crates/coins left
        if self.heuristic_ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)

        # Exclude targets that are currently occupied by a bomb
        targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        if self.heuristic_ignore_others_timer > 0:
            for o in others:
                free_space[o] = False
        d = look_for_targets(free_space, (x, y), targets)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if touching an opponent
        if len(others) > 0:
            if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
                action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) < 4):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) < 4):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                # Keep track of chosen action for cycle detection
                if a == 'BOMB':
                    self.heuristic_bomb_history.append((x, y))
                return a

        return 'WAIT'

    except Exception as e:
        self.logger.error(f"Error in rule-based heuristic: {e}")
        import traceback
        traceback.print_exc()
        return 'WAIT'

def act(self, game_state):
    """Called each game step to select an action."""

    # Safety check
    if not hasattr(self, 'train_agent') or self.train_agent is None:
        self.logger.error("train_agent not initialized! Calling setup()...")
        setup(self)

    # Episode tracking
    if game_state and game_state.get('step', 0) == 1:
        self.episode_active = True
        self.current_step = 0

    # Increment step counter
    if game_state:
        self.current_step = game_state.get('step', self.current_step + 1)

    # Select action with action masking (using spatial + scalar features)
    spatial_obs, scalar_obs = self.train_agent.featurize(game_state)

    # Collect expert action for auxiliary loss (doesn't affect which action is taken)
    in_training_mode = hasattr(self, 'round_counter')
    expert_action_idx = None
    if in_training_mode:
        try:
            expert_action_str = _rule_based_heuristic_action(self, game_state)
            expert_action_idx = ACTION_IDX[expert_action_str]
        except Exception:
            expert_action_idx = None

    # Always use PPO policy to select action
    try:
        action_idx, log_prob, value, action_mask = self.train_agent.select_action(
            spatial_obs, scalar_obs, game_state=game_state
        )
        action = ACTIONS[action_idx]

    except Exception as e:
        self.logger.error(f"Error selecting action: {e}")
        import traceback
        traceback.print_exc()
        action_idx = 4  # WAIT fallback
        log_prob = 0.0
        value = 0.0
        action_mask = None
        action = 'WAIT'

    # Store for training use (including expert action for auxiliary loss)
    self.last_spatial_obs = spatial_obs
    self.last_scalar_obs = scalar_obs
    self.last_action = action_idx
    self.last_log_prob = log_prob
    self.last_value = value
    self.last_action_mask = action_mask
    self.last_expert_action = expert_action_idx

    # Track own bomb positions for featurization
    if action == 'BOMB' and game_state:
        _, _, _, (x, y) = game_state["self"]
        if not hasattr(self.train_agent, 'own_bomb_positions'):
            self.train_agent.own_bomb_positions = set()
        self.train_agent.own_bomb_positions.add((x, y))

    return action


def _end_gameplay_episode(self, game_state, events, died=False):
    """Helper function to end episode during gameplay (metrics disabled)."""
    if hasattr(self, 'episode_active'):
        self.episode_active = False


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Called at the end of each round during gameplay."""
    if hasattr(self, 'episode_active') and self.episode_active:
        _end_gameplay_episode(self, last_game_state, events, died=False)


def game_events_occurred(self, old_game_state: dict, self_action: str,
                        new_game_state: dict, events: list):
    """Optional: Track events during gameplay (metrics disabled to save disk space)."""
    pass  # Metrics tracking disabled
