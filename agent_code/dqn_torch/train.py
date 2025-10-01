import random
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -------------------------
# DQN Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Training Setup (updated)
# -------------------------
def setup_training(self):
    """Initialize everything for training mode."""
    self.gamma = 0.99
    self.batch_size = 64
    self.lr = 1e-3
    self.epsilon = 1.0
    self.epsilon_min = 0.05
    self.epsilon_decay = 0.995
    self.target_update = 50
    self.memory_size = 100_000
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.memory = deque(maxlen=self.memory_size)

    # IMPORTANT: match this to state_to_features output size.
    # Current feature extractor returns a single 17x17 channel => 289.
    self.input_dim = 17 * 17  # 289

    self.output_dim = 6  # [UP, DOWN, LEFT, RIGHT, BOMB, WAIT]

    self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
    self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    self.steps_done = 0

    print("✅ Training setup complete.")


# -------------------------
# Core callback
# -------------------------
def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called once per step to inform the agent about what just happened.
    Here we store transitions in replay memory.
    """
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    action_map = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]
    # action_idx = action_map.index(self_action)
    if self_action is None or self_action not in action_map:
        action_idx = 5  # default to WAIT if no valid action
    else:
        action_idx = action_map.index(self_action)

    reward = reward_from_events(events)

    self.memory.append((old_features, action_idx, reward, new_features, 0))
    optimize_model(self)

# -------------------------
# End of Round callback
# -------------------------
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each round. Good place to finalize updates and save the model.
    """
    last_features = state_to_features(last_game_state)
    action_map = ["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"]
    if last_action is None or last_action not in action_map:
        action_idx = 5  # default to WAIT if no valid action
    else:
        action_idx = action_map.index(last_action)

    reward = reward_from_events(events)
    self.memory.append((last_features, action_idx, reward, np.zeros_like(last_features), 1))

    # Run a few extra updates at end of episode
    for _ in range(10):
        optimize_model(self)

    # Save model checkpoint
    if not hasattr(self, "save_counter"):
        self.save_counter = 0
    self.save_counter += 1
    if self.save_counter % 50 == 0:
        torch.save(self.policy_net.state_dict(), f"dqn_checkpoint_{self.save_counter}.pth")
        print(f"💾 Saved model at episode {self.save_counter}")

# -------------------------
# Reward shaping
# -------------------------
def reward_from_events(events: List[str]) -> int:
    reward = 0
    if "COIN_COLLECTED" in events:
        reward += 10
    if "KILLED_OPPONENT" in events:
        reward += 20
    if "INVALID_ACTION" in events:
        reward -= 5
    if "GOT_KILLED" in events:
        reward -= 20
    return reward

# -------------------------
# Optimize step (updated)
# -------------------------
def optimize_model(self):
    if len(self.memory) < self.batch_size:
        return

    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert lists of arrays to properly stacked NumPy arrays first
    states = np.stack(states).astype(np.float32)
    next_states = np.stack(next_states).astype(np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    # Then convert to tensors
    states = torch.from_numpy(states).to(self.device)
    next_states = torch.from_numpy(next_states).to(self.device)
    actions = torch.from_numpy(actions).to(self.device)
    rewards = torch.from_numpy(rewards).to(self.device)
    dones = torch.from_numpy(dones).to(self.device)

    # Q-learning targets
    q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
    next_q_values = self.target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

    loss = nn.MSELoss()(q_values.squeeze(-1), expected_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Epsilon decay
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

# -------------------------
# Helper: state -> features (updated)
# -------------------------
def state_to_features(game_state: dict):
    """
    Convert the game_state dict into a 1D float32 feature vector.
    Current implementation: single 17x17 channel -> length 289.
    """
    if game_state is None:
        return np.zeros(17 * 17, dtype=np.float32)

    # field shape: (17, 17); flatten to length 289 vector
    field = game_state["field"].astype(np.float32).flatten()
    return field
