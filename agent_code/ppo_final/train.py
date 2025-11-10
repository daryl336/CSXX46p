import os
import numpy as np
from typing import List

try:
    from . import callbacks as ppo_callbacks
except Exception:
    import callbacks as ppo_callbacks

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ppo_agent.pth")

BASE_REWARDS = {
    # Core objectives - these are what matter
    "COIN_COLLECTED": 0.5,        # PRIMARY GOAL - increased from 1.0
    "KILLED_OPPONENT": 1.0,        # SECONDARY GOAL
    "CRATE_DESTROYED": 0.05,        # ENABLES COINS - increased from 0.1

    # Death penalties
    "KILLED_SELF": -0.5,
    "GOT_KILLED": -0.5,            # Moderate penalty for death

    # Minor events
    "SURVIVED_ROUND": 0.0,         # NO reward for just surviving (will add terminal reward based on score)
    "COIN_FOUND": 0.0,
    "OPPONENT_ELIMINATED": 0.0,    # Covered by KILLED_OPPONENT
    "BOMB_EXPLODED": 0.0,

    # Movement
    "MOVED_LEFT": -0.001,
    "MOVED_RIGHT": -0.001,
    "MOVED_UP": -0.001,
    "MOVED_DOWN": -0.001,
    "WAITED": -0.001,
    "BOMB_DROPPED": -0.001,

    # Invalid actions
    "INVALID_ACTION": -0.2,        # Small penalty
}


def is_in_danger(game_state, x, y, look_ahead_timer=2):
    if game_state is None:
        return False

    try:
        if game_state["explosion_map"][x, y] > 0:
            return True

        for (bx, by), timer in game_state["bombs"]:
            if timer <= look_ahead_timer:
                if by == y and abs(bx - x) <= 3:
                    blocked = False
                    step = 1 if bx > x else -1
                    for check_x in range(x, bx + step, step):
                        if game_state["field"][check_x, y] == -1:
                            blocked = True
                            break
                    if not blocked:
                        return True

                if bx == x and abs(by - y) <= 3:
                    blocked = False
                    step = 1 if by > y else -1
                    for check_y in range(y, by + step, step):
                        if game_state["field"][x, check_y] == -1:
                            blocked = True
                            break
                    if not blocked:
                        return True
        return False
    except Exception:
        return False


def has_path_to_safety(game_state, start_x, start_y, max_depth=5):
    if game_state is None:
        return False

    try:
        from collections import deque

        visited = set()
        queue = deque([(start_x, start_y, 0)])
        visited.add((start_x, start_y))

        while queue:
            cx, cy, depth = queue.popleft()

            if not is_in_danger(game_state, cx, cy) and game_state["explosion_map"][cx, cy] == 0:
                min_bomb_dist = 999
                for (bx, by), timer in game_state["bombs"]:
                    dist = abs(bx - cx) + abs(by - cy)
                    min_bomb_dist = min(min_bomb_dist, dist)

                if min_bomb_dist >= 4:
                    return True

            if depth >= max_depth:
                continue

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if ((0 <= nx < 17 and 0 <= ny < 17) and (nx, ny) not in visited and
                    game_state["field"][nx, ny] == 0 and game_state["explosion_map"][nx, ny] == 0):
                    visited.add((nx, ny))
                    queue.append((nx, ny, depth + 1))

        return False
    except Exception:
        return False


def count_escape_routes(game_state, x, y):
    if game_state is None:
        return 0

    try:
        count = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < 17 and 0 <= ny < 17 and game_state["field"][nx, ny] == 0 and
                game_state["explosion_map"][nx, ny] == 0 and has_path_to_safety(game_state, nx, ny, max_depth=6)):
                count += 1
        return count
    except Exception:
        return 0


def count_destructible_in_range(game_state, x, y, range_val=3):
    if game_state is None:
        return 0

    try:
        count = 0
        for dx in range(-range_val, range_val + 1):
            nx = x + dx
            if 0 <= nx < 17:
                if game_state["field"][nx, y] == 1:
                    count += 1
                if game_state["field"][nx, y] == -1:
                    break

        for dy in range(-range_val, range_val + 1):
            ny = y + dy
            if 0 <= ny < 17:
                if game_state["field"][x, ny] == 1:
                    count += 1
                if game_state["field"][x, ny] == -1:
                    break
        return count
    except Exception:
        return 0


def compute_custom_rewards(self, old_state, action, new_state, events):
    custom_reward = 0.0

    if old_state is None or new_state is None:
        return custom_reward

    if not hasattr(self, 'own_bomb_tracker'):
        self.own_bomb_tracker = {}

    if not hasattr(self, 'closest_coin_distance'):
        self.closest_coin_distance = {}

    if not hasattr(self, 'danger_escape_cooldown'):
        self.danger_escape_cooldown = 0

    if not hasattr(self, 'own_bomb_max_dist'):
        self.own_bomb_max_dist = {}

    if not hasattr(self, 'rewarded_bomb_survivals'):
        self.rewarded_bomb_survivals = set()

    try:
        old_x, old_y = old_state["self"][3]
        new_x, new_y = new_state["self"][3]

        # Track progress toward coins (only reward if we beat our historical best distance)
        if len(new_state["coins"]) > 0:
            new_coin_distances = [abs(cx - new_x) + abs(cy - new_y) for cx, cy in new_state["coins"]]
            new_min_dist = min(new_coin_distances)

            # Find which coin we're closest to
            closest_coin_idx = new_coin_distances.index(new_min_dist)
            closest_coin = new_state["coins"][closest_coin_idx]

            # Check if we've improved our best distance to this specific coin
            if closest_coin in self.closest_coin_distance:
                best_dist = self.closest_coin_distance[closest_coin]
                if new_min_dist < best_dist:
                    # Reward for achieving new best distance to this coin
                    custom_reward += 0.05  # Increased from 0.03 to incentivize coin pursuit
                    self.closest_coin_distance[closest_coin] = new_min_dist
            else:
                # First time seeing this coin, just record the distance
                self.closest_coin_distance[closest_coin] = new_min_dist

        # Clean up tracking for collected coins
        if "COIN_COLLECTED" in events:
            # Remove collected coins from tracking
            coins_to_remove = [coin for coin in self.closest_coin_distance.keys()
                              if coin not in new_state["coins"]]
            for coin in coins_to_remove:
                del self.closest_coin_distance[coin]

        if "BOMB_DROPPED" in events:
            self.own_bomb_tracker[(new_x, new_y)] = new_state["step"]
            crates_in_range = count_destructible_in_range(new_state, new_x, new_y, range_val=3)

            if crates_in_range > 0:
                custom_reward += 0.066 * crates_in_range

            if len(new_state["others"]) > 0:
                opponent_in_range = any([abs(ox - new_x) + abs(oy - new_y) <= 3
                                       for _, _, _, (ox, oy) in new_state["others"]])
                if opponent_in_range:
                    custom_reward += 0.1

            escape_options = count_escape_routes(new_state, new_x, new_y)
            if escape_options == 0:
                custom_reward -= 0.1

        old_in_danger = is_in_danger(old_state, old_x, old_y, look_ahead_timer=4)
        new_in_danger = is_in_danger(new_state, new_x, new_y, look_ahead_timer=4)

        bombs_to_remove = []
        for bomb_pos, placed_step in list(self.own_bomb_tracker.items()):
            bx, by = bomb_pos
            steps_since_placed = new_state["step"] - placed_step

            if steps_since_placed <= 4:
                old_dist = abs(old_x - bx) + abs(old_y - by)
                new_dist = abs(new_x - bx) + abs(new_y - by)

                # Track maximum distance achieved from each bomb to prevent oscillation
                max_dist = self.own_bomb_max_dist.get(bomb_pos, old_dist)
                if new_in_danger:
                    if new_dist > max_dist:
                        custom_reward += 0.01
                        self.own_bomb_max_dist[bomb_pos] = new_dist
                    elif new_dist <= old_dist or not old_in_danger:
                        custom_reward -= 0.05
                elif old_in_danger:
                    # Escaped bomb danger early
                    custom_reward += 0.05

            elif steps_since_placed > 4:
                if not is_in_danger(new_state, new_x, new_y):
                    custom_reward += 0.02
                    if hasattr(self, 'bomb_survival_count'):
                        self.bomb_survival_count += 1
                bombs_to_remove.append(bomb_pos)

        for bomb_pos in bombs_to_remove:
            del self.own_bomb_tracker[bomb_pos]
            # Clean up tracking dicts
            if bomb_pos in self.own_bomb_max_dist:
                del self.own_bomb_max_dist[bomb_pos]

        return custom_reward

    except Exception as e:
        print(f"Error in compute_custom_rewards: {e}")
        return 0.0


def setup_training(self):
    self.name = "PPO"

    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir != "":
        os.makedirs(model_dir, exist_ok=True)

    if hasattr(self, "train_agent") and self.train_agent is not None:
        self.logger.info("Using existing train_agent")
        # Restore round_counter from existing train_agent
        self.round_counter = getattr(self, "round_counter", 0)
    else:
        try:
            ppo_callbacks.setup(self)
            if not (hasattr(self, "train_agent") and self.train_agent is not None):
                raise RuntimeError("callbacks.setup() did not create train_agent")
            # Get round_counter from setup (which loads from checkpoint)
            self.round_counter = getattr(self, "round_counter", 0)
        except Exception as e:
            self.logger.error(f"Failed to initialize train_agent: {e}")
            raise

    try:
        from .visualization import TrainingVisualizer
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        # Load existing metrics when resuming training
        self.visualizer = TrainingVisualizer(log_dir=log_dir, load_existing=True)
    except Exception as e:
        self.logger.warning(f"Failed to initialize visualizer: {e}")
        self.visualizer = None

    # Restore metrics counters from checkpoint if available
    restored_counters = getattr(self, 'restored_metrics_counters', {})
    self.metrics_tracker = None
    self.own_bomb_tracker = {}
    self.bomb_usage_count = restored_counters.get('bomb_usage_count', 0)
    self.bomb_survival_count = restored_counters.get('bomb_survival_count', 0)
    self.bomb_death_count = restored_counters.get('bomb_death_count', 0)
    self.total_actions = restored_counters.get('total_actions', 0)
    self.current_round_score = 0
    self.current_round_return = 0

    if restored_counters:
        self.logger.info(f"Restored metrics counters: actions={self.total_actions}, "
                        f"bombs={self.bomb_usage_count}, survived={self.bomb_survival_count}, "
                        f"deaths={self.bomb_death_count}")

    # Initialize coin distance tracking for reward shaping
    self.closest_coin_distance = {}
    self.danger_escape_cooldown = 0
    self.own_bomb_max_dist = {}
    self.rewarded_bomb_survivals = set()

    # Initialize best model tracking (based on moving average score)
    self.best_avg_score = restored_counters.get('best_avg_score', -float('inf'))
    self.best_avg_score_round = restored_counters.get('best_avg_score_round', 0)
    self.recent_scores = []  # Track last 100 scores for moving average

    if self.best_avg_score > -float('inf'):
        self.logger.info(f"Restored best model tracking: best_avg_score={self.best_avg_score:.2f} "
                        f"(round {self.best_avg_score_round})")


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    if old_game_state is None or new_game_state is None:
        return

    if not hasattr(self, "train_agent") or self.train_agent is None:
        return

    if not hasattr(self, "last_spatial_obs") or self.last_spatial_obs is None:
        return

    reward = sum([BASE_REWARDS.get(event, 0) for event in events])
    reward += compute_custom_rewards(self, old_game_state, self_action, new_game_state, events)

    if hasattr(self, 'current_round_score'):
        if "COIN_COLLECTED" in events:
            self.current_round_score += 1
        if "KILLED_OPPONENT" in events:
            self.current_round_score += 5

    if hasattr(self, 'current_round_return'):
        self.current_round_return += reward

    if hasattr(self, 'total_actions'):
        self.total_actions += 1
    if "BOMB_DROPPED" in events and hasattr(self, 'bomb_usage_count'):
        self.bomb_usage_count += 1

    # Check if agent died mid-round (terminal state)
    agent_died = "GOT_KILLED" in events or "KILLED_SELF" in events

    try:
        expert_action = getattr(self, 'last_expert_action', None)
        self.train_agent.store_transition(
            spatial_obs=self.last_spatial_obs,
            scalar_obs=self.last_scalar_obs,
            action=self.last_action,
            log_prob=self.last_log_prob,
            reward=reward,
            value=self.last_value,
            done=agent_died,  # Set done=True if agent died
            action_mask=self.last_action_mask,
            expert_action=expert_action
        )
    except Exception as e:
        self.logger.error(f"Failed to store transition: {e}")


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if not hasattr(self, "train_agent") or self.train_agent is None:
        return

    final_reward = sum([BASE_REWARDS.get(event, 0) for event in events])
    final_reward += compute_custom_rewards(self, None, last_action, last_game_state, events)

    # FIX #1: Terminal rewards based on performance (not rank)
    # This makes scoring valuable and penalizes passive survival
    try:
        survived = "SURVIVED_ROUND" in events

        # Get score this round (coins + kills)
        round_score = getattr(self, 'current_round_score', 0)

        if survived:
            if round_score > 0:
                final_reward += 0.5
            else:
                final_reward -= 0.5

    except Exception as e:
        self.logger.warning(f"Failed to compute terminal reward: {e}")

    if hasattr(self, "last_spatial_obs") and self.last_spatial_obs is not None:
        try:
            expert_action = getattr(self, 'last_expert_action', None)
            self.train_agent.store_transition(
                spatial_obs=self.last_spatial_obs,
                scalar_obs=self.last_scalar_obs,
                action=self.last_action,
                log_prob=self.last_log_prob,
                reward=final_reward,
                value=self.last_value,
                done=True,
                action_mask=self.last_action_mask,
                expert_action=expert_action
            )
        except Exception as e:
            self.logger.error(f"Failed to store final transition: {e}")

    self.train_agent.episodes_since_update += 1

    if self.train_agent.episodes_since_update >= self.train_agent.episodes_per_update:
        try:
            self.train_agent.update()
            self.train_agent.episodes_since_update = 0

            if len(self.train_agent.episode_metrics['actor_losses']) > 0:
                metrics = self.train_agent.episode_metrics
                self.logger.info(f"Update: loss={metrics['actor_losses'][-1]:.3f}, "
                               f"return={metrics['returns'][-1]:.1f}, KL={metrics['kl_divs'][-1]:.4f}")
        except Exception as e:
            self.logger.error(f"PPO update failed: {e}")

    try:
        # Save metrics counters along with model
        metrics_counters = {
            'bomb_usage_count': self.bomb_usage_count,
            'bomb_survival_count': self.bomb_survival_count,
            'bomb_death_count': self.bomb_death_count,
            'total_actions': self.total_actions,
        }
        self.train_agent.save(MODEL_PATH, round_counter=self.round_counter,
                             metrics_counters=metrics_counters)
    except Exception as e:
        self.logger.error(f"Failed to save model: {e}")

    # Track bomb deaths here because dead agents don't receive game_events_occurred()
    if "KILLED_SELF" in events and hasattr(self, 'bomb_death_count'):
        self.bomb_death_count += 1
        self.logger.debug(f"KILLED_SELF in end_of_round! bomb_death_count: {self.bomb_death_count}")

    if hasattr(self, 'own_bomb_tracker'):
        self.own_bomb_tracker = {}
    if hasattr(self.train_agent, 'own_bomb_positions'):
        self.train_agent.own_bomb_positions = set()
    if hasattr(self, 'closest_coin_distance'):
        self.closest_coin_distance = {}
    if hasattr(self, 'danger_escape_cooldown'):
        self.danger_escape_cooldown = 0
    if hasattr(self, 'own_bomb_max_dist'):
        self.own_bomb_max_dist = {}
    if hasattr(self, 'rewarded_bomb_survivals'):
        self.rewarded_bomb_survivals = set()

    self.round_counter = getattr(self, "round_counter", 0) + 1

    bomb_usage_rate = self.bomb_usage_count / max(self.total_actions, 1)
    survival_rate = self.bomb_survival_count / max(self.bomb_usage_count, 1)
    death_rate = self.bomb_death_count / max(self.bomb_usage_count, 1)

    actor_loss = self.train_agent.episode_metrics['actor_losses'][-1] if len(self.train_agent.episode_metrics['actor_losses']) > 0 else 0.0
    critic_loss = self.train_agent.episode_metrics['critic_losses'][-1] if len(self.train_agent.episode_metrics['critic_losses']) > 0 else 0.0

    if hasattr(self, 'visualizer') and self.visualizer is not None:
        try:
            self.visualizer.record_round(
                round_num=self.round_counter,
                score=self.current_round_score,
                episode_return=self.current_round_return,
                bomb_usage=bomb_usage_rate,
                survival_rate=survival_rate,
                death_rate=death_rate,
                entropy=self.train_agent.entropy_coef,
                lr=self.train_agent.optimizer.param_groups[0]['lr'],
                actor_loss=actor_loss,
                critic_loss=critic_loss
            )
            csv_path = self.visualizer.save_csv(append_mode=True)
        except Exception as e:
            self.logger.warning(f"Failed to record visualization metrics: {e}")

    # Track best model based on moving average score
    current_score = self.current_round_score

    # Update recent scores for moving average (last 100 rounds)
    if hasattr(self, 'recent_scores'):
        self.recent_scores.append(current_score)

        # Check for new best moving average after we have at least 100 rounds
        if len(self.recent_scores) >= 100:
            avg_score = sum(self.recent_scores) / len(self.recent_scores)
            self.recent_scores.clear()  # Reset for next evaluation window
            if avg_score > self.best_avg_score:
                old_best_avg = self.best_avg_score
                self.best_avg_score = avg_score
                self.best_avg_score_round = self.round_counter

                # Save best model
                best_model_path = MODEL_PATH.replace('.pth', '_best.pth')
                try:
                    metrics_counters_best = {
                        'bomb_usage_count': self.bomb_usage_count,
                        'bomb_survival_count': self.bomb_survival_count,
                        'bomb_death_count': self.bomb_death_count,
                        'total_actions': self.total_actions,
                        'best_avg_score': self.best_avg_score,
                        'best_avg_score_round': self.best_avg_score_round,
                    }
                    self.train_agent.save(best_model_path, round_counter=self.round_counter,
                                         metrics_counters=metrics_counters_best)
                    self.logger.info(f"NEW BEST MODEL: Avg score {self.best_avg_score:.2f} "
                                   f"(last 100 rounds, round {self.round_counter}, previous: {old_best_avg:.2f}) "
                                   f"- Saved to {best_model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save best model: {e}")

    self.current_round_score = 0
    self.current_round_return = 0

    if self.round_counter % 100 == 0:
        self.logger.info(f"=== Round {self.round_counter} ===")
        self.logger.info(f"Training step: {self.train_agent.training_steps}")
        self.logger.info(f"Entropy: {self.train_agent.entropy_coef:.4f}, LR: {self.train_agent.optimizer.param_groups[0]['lr']:.6f}")
        self.logger.info(f"Bomb usage: {bomb_usage_rate:.1%}, Survival: {survival_rate:.1%}, Deaths: {death_rate:.1%}")

        if hasattr(self, 'visualizer') and self.visualizer is not None:
            try:
                self.visualizer.load_csv()
                plot_path = self.visualizer.plot_training_progress(smooth=True)
                summary = self.visualizer.generate_summary_report()
                self.logger.info(f"Plot: {plot_path}")
                self.logger.info(f"\n{summary}")
            except Exception as e:
                self.logger.warning(f"Failed to generate visualizations: {e}")
