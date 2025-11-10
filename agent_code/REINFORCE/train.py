import pickle
import numpy as np
from typing import List
import torch
import torch.optim as optim
import os
from datetime import datetime
import events as e
from .callbacks import state_to_features, device
from collections import deque
import json

def setup_training(self):
    """
    Initialize training-specific variables for REINFORCE.

    REINFORCE is a Monte Carlo policy gradient method:
    - Collect full episode (all states, actions, rewards)
    - Calculate returns (sum of future rewards)
    - Update policy to increase probability of good actions

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.info("Setting up REINFORCE training mode.")

    # Episode storage (cleared after each episode)
    self.saved_log_probs = []  # Log probabilities of actions
    self.rewards = []  # Rewards at each step

    # Training hyperparameters
    self.gamma = 0.99  # Discount factor

    # Statistics tracking
    self.episode_rewards = []


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called after each step.

    For REINFORCE: Just collect data, don't update yet!

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state before action
    :param self_action: The action taken
    :param new_game_state: The state after action
    :param events: Events that occurred
    """
    self.logger.debug(f'Events: {", ".join(events)}')
     # Add episode tracking like PPO
    if old_game_state:
        if old_game_state.get('step', 0) == 1:
            # Extract opponent information
            opponent_names = []
            if 'others' in old_game_state and old_game_state['others']:
                for other in old_game_state['others']:
                    if other is not None:
                        opponent_names.append(other[0])
            
            self.episode_counter = old_game_state.get('round')
            # Start episode tracking
            self.metrics_tracker.start_episode(
                episode_id=self.episode_counter,
                opponent_types=opponent_names,
                scenario="training"
            )

            # Reset exploration tracking
            self.visited_positions = set()
            self.last_coin_distance = None
            
            self.logger.debug(f"Started episode {self.episode_counter} with opponents: {opponent_names}")

    # Add custom events based on state changes
    events = add_custom_events(old_game_state, new_game_state, events)
    for event in events:
        event_name = event if isinstance(event, str) else getattr(event, 'name', str(event))
        self.metrics_tracker.record_event(event_name)
        self.logger.debug(f"Recorded event: {event_name}")

    # Calculate reward
    reward = reward_from_events(events)

    # Store the log_prob (set in callbacks.act()) and reward
    if hasattr(self, 'last_log_prob'):
        self.saved_log_probs.append(self.last_log_prob)
        self.rewards.append(reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at end of episode.

    THIS IS WHERE WE UPDATE THE POLICY!

    REINFORCE algorithm:
    1. Calculate returns: G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    2. Calculate loss: -Σ log π(a_t|s_t) * G_t
    3. Backpropagate and update policy

    :param self: Agent object
    :param last_game_state: Final state
    :param last_action: Final action
    :param events: Final events
    """
    self.logger.debug(f'Final events: {", ".join(events)}')

    # Get final reward
    final_reward = reward_from_events(events)

    # Store final transition
    if hasattr(self, 'last_log_prob'):
        self.saved_log_probs.append(self.last_log_prob)
        self.rewards.append(final_reward)

    # Calculate total episode reward (for logging)
    total_reward = sum(self.rewards)
    self.episode_rewards.append(total_reward)

    # ============================================
    # REINFORCE UPDATE (the main algorithm!)
    # ============================================

    if len(self.saved_log_probs) > 0:
        # Step 1: Calculate returns (discounted cumulative rewards)
        returns = deque()
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G  # G_t = r_t + γ*G_{t+1}
            returns.appendleft(G)

        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize returns (helps training stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Step 2: Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)  # -log π(a|s) * G

        policy_loss = torch.stack(policy_loss).mean()

        # Step 3: Backpropagate
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.logger.info(f'Policy loss: {policy_loss.item():.4f}')

        round_num = last_game_state["round"]

        # Save loss data
        if policy_loss.item() is not None:
            save_loss_data(
                round_num=round_num,
                loss=policy_loss.item(),
                reward=total_reward,
                episode_length=len(self.rewards),
                log_dir="training_logs"
            )

    # ============================================
    # Logging and cleanup
    # ============================================

    self.logger.info(f'Round {last_game_state["round"]} finished')
    self.logger.info(f'Episode reward: {total_reward:.2f}')
    self.logger.info(f'Episode length: {len(self.rewards)} steps')

    # Log average over last 100 episodes
    if len(self.episode_rewards) >= 100:
        avg_reward = np.mean(self.episode_rewards[-100:])
        self.logger.info(f'Avg reward (last 100): {avg_reward:.2f}')

    # Clear episode data for next round
    self.saved_log_probs = []
    self.rewards = []
    # Delete the reference to the last log_prob tensor to prevent it from
    # being used in the next episode after its graph has been freed.
    if hasattr(self, 'last_log_prob'):
        del self.last_log_prob

    # Save model every 100 rounds
    if last_game_state["round"] % 100 == 0:
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "my-saved-model.pth")
        self.logger.info("Model saved.")
    rank = 4
    if e.SURVIVED_ROUND in events and "others" in last_game_state:
        alive = sum(1 for o in last_game_state["others"] if o)
        won = alive == 0
        rank = 1 if won else 2
    # === Record episode end ===
    current_step = last_game_state.get("step", 0)
    self.metrics_tracker.end_episode(
        won="WON" in events,
        rank=rank,
        survival_steps=current_step,
        total_steps=400
    )
    self.episode_active = False
    self.episode_counter += 1

    self.metrics_tracker.save()


def add_custom_events(old_game_state, new_game_state, events):
    """
    Add custom events to provide more detailed feedback.

    Custom events:
    - MOVED_TOWARDS_COIN: Agent moved closer to nearest coin
    - MOVED_AWAY_FROM_COIN: Agent moved farther from nearest coin
    - ESCAPED_DANGER: Agent moved from dangerous area to safe area
    - MOVED_INTO_DANGER: Agent moved from safe area to dangerous area
    - BOMB_NEAR_CRATE: Agent placed bomb next to crate(s)

    :param old_game_state: Previous state
    :param new_game_state: Current state
    :param events: List of events that occurred
    :return: Updated events list
    """
    if old_game_state is None or new_game_state is None:
        return events

    old_x, old_y = old_game_state['self'][3]
    new_x, new_y = new_game_state['self'][3]

    # Custom event 1: Moved towards/away from coin
    old_coins = old_game_state['coins']
    new_coins = new_game_state['coins']

    if len(old_coins) > 0 and len(new_coins) > 0:
        # Calculate distance to nearest coin in both states
        old_min_dist = min(abs(cx - old_x) + abs(cy - old_y) for cx, cy in old_coins)
        new_min_dist = min(abs(cx - new_x) + abs(cy - new_y) for cx, cy in new_coins)

        if new_min_dist < old_min_dist:
            events.append('MOVED_TOWARDS_COIN')
        elif new_min_dist > old_min_dist:
            events.append('MOVED_AWAY_FROM_COIN')

    # Custom event 2: Danger management
    old_explosion_map = old_game_state['explosion_map']
    new_explosion_map = new_game_state['explosion_map']
    old_bombs = old_game_state['bombs']
    new_bombs = new_game_state['bombs']

    old_danger = is_in_danger(old_x, old_y, old_bombs, old_explosion_map)
    new_danger = is_in_danger(new_x, new_y, new_bombs, new_explosion_map)

    if old_danger and not new_danger:
        events.append('ESCAPED_DANGER')
    elif not old_danger and new_danger:
        events.append('MOVED_INTO_DANGER')

    # Custom event 3: Bomb placement near crates
    if e.BOMB_DROPPED in events:
        old_field = old_game_state['field']
        crates_nearby = count_crates_nearby(old_x, old_y, old_field)

        if crates_nearby > 0:
            events.append('BOMB_NEAR_CRATE')

            # Extra reward for multiple crates
            if crates_nearby >= 2:
                events.append('BOMB_MULTI_CRATE')

    return events


def is_in_danger(x, y, bombs, explosion_map):
    """
    Check if position is in danger zone.

    :param x: X position
    :param y: Y position
    :param bombs: List of bombs
    :param explosion_map: Current explosion map
    :return: True if in danger, False otherwise
    """
    # Currently exploding
    if explosion_map[x, y] > 0:
        return True

    # In bomb blast radius
    for (bx, by), timer in bombs:
        if bx == x and abs(by - y) <= 3:
            return True
        if by == y and abs(bx - x) <= 3:
            return True

    return False


def count_crates_nearby(x, y, field):
    """
    Count number of crates within bomb blast radius.

    :param x: X position
    :param y: Y position
    :param field: Game field
    :return: Number of crates
    """
    crate_count = 0

    # Check in 4 directions (up, right, down, left) within range 3
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for distance in range(1, 4):  # Blast radius of 3
            nx, ny = x + dx * distance, y + dy * distance

            # Check bounds
            if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1]:
                break

            # Wall blocks blast
            if field[nx, ny] == -1:
                break

            # Count crate
            if field[nx, ny] == 1:
                crate_count += 1

    return crate_count

def reward_from_events(events: List[str]) -> float:
    """
    Simple reward function.

    :param events: List of game events
    :return: Total reward
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -30,
        e.INVALID_ACTION: -0.5,
        e.WAITED: -0.2,
        e.CRATE_DESTROYED: 2,
        e.SURVIVED_ROUND: 1,

        # Official game events (medium impact)
        e.CRATE_DESTROYED: 3.0,
        e.BOMB_DROPPED: 0.0,  # Neutral, context matters

        # Custom events (movement)
        'MOVED_TOWARDS_COIN': 0.01,
        'MOVED_AWAY_FROM_COIN': -0.01,

        # Custom events (danger management)
        'ESCAPED_DANGER': 5.0,
        'MOVED_INTO_DANGER': -3.0,

        # Custom events (strategic bombing)
        'BOMB_NEAR_CRATE': 2.0,
        'BOMB_MULTI_CRATE': 3.0,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum


def save_loss_data(round_num, loss, reward, episode_length, log_dir="training_logs"):
    """
    Save loss and metrics data to JSON file.

    Args:
        round_num: Current round/episode number
        loss: Policy loss value
        reward: Total episode reward
        episode_length: Number of steps in episode
        log_dir: Directory to save logs
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create filename with timestamp (one file per training session)
    if not hasattr(save_loss_data, 'filename'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_loss_data.filename = os.path.join(log_dir, f"training_losses_{timestamp}.json")
        save_loss_data.data = []

    # Append new data point
    data_point = {
        "round": round_num,
        "loss": float(loss),
        "reward": float(reward),
        "episode_length": int(episode_length)
    }
    save_loss_data.data.append(data_point)

    # Save to file (overwrite with all data)
    with open(save_loss_data.filename, 'w') as f:
        json.dump(save_loss_data.data, f, indent=2)