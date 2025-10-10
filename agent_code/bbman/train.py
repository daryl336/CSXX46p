import random
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os
from .callbacks import DuelingDQN, PrioritizedReplayBuffer
from .helper import state_to_features, update_epsilon, reward_from_events
from .helper import ACTIONS
# -------------------------
# Enhanced CNN-LSTM Model Components
# -------------------------

# CNN-LSTM specific parameters
CNN_CHANNELS = [32, 64, 128, 256]
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 2
SEQUENCE_LENGTH = 8  # Number of frames to consider for temporal patterns
DROPOUT_RATE = 0.3

# -------------------------
# Enhanced Training Setup
# -------------------------
def setup_training(self):
    """Initialize everything for enhanced training mode with CNN-LSTM architecture."""
    # Enhanced hyperparameters
    self.gamma = 0.99
    self.batch_size = 32  # Reduced for more stable training with larger model
    self.lr = 1e-4  # Lower learning rate for better stability
    self.epsilon = 1.0
    self.epsilon_min = 0.05
    self.epsilon_decay_steps = 300_000  # Linear decay over more steps
    self.epsilon_step = 0
    self.target_update_freq = 1000  # Less frequent hard updates since we use soft updates
    self.tau = 0.005  # Soft target update parameter
    self.memory_size = 100_000
    self.min_memory_size = 2_000  # Start learning only after enough samples
    self.save_path = "models"
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced replay buffer with prioritized experience replay
    self.memory = PrioritizedReplayBuffer(self.memory_size, alpha=0.6)
    self.beta = 0.4  # Importance sampling parameter
    self.beta_increment = 0.001

    # Input dimensions for CNN-LSTM: (sequence_length, channels, height, width)
    self.sequence_length = SEQUENCE_LENGTH
    self.input_channels = 4  # Multi-channel state representation
    self.output_dim = 6  # [UP, DOWN, LEFT, RIGHT, BOMB, WAIT]

    # Enhanced CNN-LSTM DQN models
    self.policy_net = DuelingDQN(
        input_channels=self.input_channels, 
        output_dim=self.output_dim, 
        sequence_length=self.sequence_length
    ).to(self.device)
    
    self.target_net = DuelingDQN(
        input_channels=self.input_channels, 
        output_dim=self.output_dim, 
        sequence_length=self.sequence_length
    ).to(self.device)
    
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    # Enhanced optimizer with weight decay
    self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.9)
    
    self.steps_done = 0
    
    # Initialize frame buffer for temporal sequences
    self.frame_buffer = deque(maxlen=self.sequence_length)
    
    # Create models directory
    os.makedirs(self.save_path, exist_ok=True)

    print("✅ Enhanced CNN-LSTM training setup complete.")
    print(f"   Device: {self.device}")
    print(f"   Model parameters: ~{sum(p.numel() for p in self.policy_net.parameters()):,}")
    print(f"   Sequence length: {self.sequence_length}")
    print(f"   Input channels: {self.input_channels}")



# -------------------------
# Core callback with enhanced features
# -------------------------
def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Enhanced version to handle CNN-LSTM architecture with temporal sequences.
    """
    # Convert to enhanced multi-channel features
    old_features = state_to_features(old_game_state, self.frame_buffer.copy() if hasattr(self, 'frame_buffer') else deque(maxlen=self.sequence_length))
    new_features = state_to_features(new_game_state, self.frame_buffer if hasattr(self, 'frame_buffer') else deque(maxlen=self.sequence_length))

    if self_action is None or self_action not in ACTIONS:
        action_idx = 4  # default to WAIT if no valid action
    else:
        action_idx = ACTIONS.index(self_action)
    
    reward = reward_from_events(events)

    # Store in prioritized replay buffer
    self.memory.add(old_features, action_idx, reward, new_features, 0)
    
    # Update epsilon
    update_epsilon(self)
    
    # Optimize model
    optimize_model(self)

# -------------------------
# End of Round callback with enhanced features
# -------------------------
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Enhanced end of round with better terminal handling and saving.
    """
    last_features = state_to_features(last_game_state, self.frame_buffer.copy() if hasattr(self, 'frame_buffer') else deque(maxlen=self.sequence_length))
    if last_action is None or last_action not in ACTIONS:
        action_idx = 4  # default to WAIT if no valid action
    else:
        action_idx = ACTIONS.index(last_action)

    reward = reward_from_events(events)
    
    # Terminal state (done=1)
    terminal_features = np.zeros_like(last_features, dtype=np.float32)
    self.memory.add(last_features, action_idx, reward, terminal_features, 1)

    # Run extra updates at end of episode for better terminal learning
    for _ in range(5):
        optimize_model(self)
    
    # Clear frame buffer for next episode
    if hasattr(self, 'frame_buffer'):
        self.frame_buffer.clear()

    # Enhanced model saving with metadata
    if not hasattr(self, "save_counter"):
        self.save_counter = 0
    self.save_counter += 1
    
    if self.save_counter % 50 == 0:
        # Save comprehensive checkpoint
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "episode": self.save_counter,
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "config": {
                "sequence_length": self.sequence_length,
                "input_channels": self.input_channels,
                "architecture": "cnn-lstm-dueling-dqn"
            }
        }
        save_filename = f"{self.save_path}/bbman_enhanced_checkpoint_{self.save_counter}.pth"
        torch.save(checkpoint, save_filename)
        print(f"💾 Saved enhanced model at episode {self.save_counter}")

# -------------------------
# Enhanced Optimize step with Double DQN and Prioritized Replay
# -------------------------
def optimize_model(self):
    """Enhanced optimization with Double DQN and prioritized experience replay"""
    if len(self.memory) < max(self.min_memory_size, self.batch_size):
        return

    try:
        # Sample from prioritized buffer
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(
            self.batch_size, self.beta
        )

        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().to(self.device)
        weights = torch.from_numpy(np.array(weights)).float().to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use policy network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Evaluate actions using target network
            target_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_q = rewards + (self.gamma * target_q * (1 - dones))

        # Compute TD errors for priority update
        td_errors = torch.abs(current_q - expected_q).detach().cpu().numpy()
        
        # Weighted MSE loss (prioritized experience replay)
        loss = (weights * F.mse_loss(current_q, expected_q, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Soft target network update (more frequent, smaller updates)
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * policy_param.data)

        self.steps_done += 1
        
    except Exception as e:
        print(f"Error in optimize_model: {e}")
        # Continue training even if one optimization step fails

