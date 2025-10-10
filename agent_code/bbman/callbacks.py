import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import events as e  # bomberman_rl event enums provided by the framework

from .helper import state_to_features, update_epsilon, reward_from_events
from .helper import ACTIONS
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception as ex:
    raise RuntimeError(
        "PyTorch is required for the bbman agent. "
        "Install it with: pip install 'torch>=2.3'"
    ) from ex

# Action space used by the framework (string labels)

ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

# -------------------------
# Enhanced Hyperparameters
# -------------------------
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
REPLAY_SIZE = 100_000
MIN_REPLAY_TO_LEARN = 2_000
TAU = 0.005                  # soft target update
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 300_000    # linear decay steps
TRAIN_EVERY_K_STEPS = 4
SAVE_EVERY_K_ROUNDS = 100

# CNN-LSTM specific parameters
CNN_CHANNELS = [32, 64, 128, 256]
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 2
SEQUENCE_LENGTH = 8  # Number of frames to consider for temporal patterns
DROPOUT_RATE = 0.3

Transition = namedtuple("Transition", ("s", "a", "r", "ns", "done"))

# -------------------------
# Enhanced CNN-LSTM Model
# -------------------------
class CNNLSTMBlock(nn.Module):
    """Combined CNN-LSTM feature extractor"""
    
    def __init__(self, input_channels=4, sequence_length=8):
        super(CNNLSTMBlock, self).__init__()
        self.sequence_length = sequence_length
        
        # Spatial Feature Extractor (CNN)
        self.conv_layers = nn.ModuleList([
            # First conv block
            nn.Conv2d(input_channels, CNN_CHANNELS[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_CHANNELS[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block  
            nn.Conv2d(CNN_CHANNELS[0], CNN_CHANNELS[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_CHANNELS[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(CNN_CHANNELS[1], CNN_CHANNELS[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_CHANNELS[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(CNN_CHANNELS[2], CNN_CHANNELS[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_CHANNELS[3]),
            nn.ReLU(inplace=True),
        ])
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(CNN_CHANNELS[3], CNN_CHANNELS[3] // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS[3] // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Calculate the flattened size after convolutions
        # For 17x17 input: 17 -> 8 -> 4 -> 2 -> 2 after pooling
        self.spatial_feat_size = CNN_CHANNELS[3] * 2 * 2
        
        # Temporal Feature Extractor (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.spatial_feat_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT_RATE if LSTM_NUM_LAYERS > 1 else 0,
            bidirectional=True
        )
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE // 4),
            nn.Tanh(),
            nn.Linear(LSTM_HIDDEN_SIZE // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE * 2, LSTM_HIDDEN_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE // 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each frame through CNN
        spatial_features = []
        for t in range(seq_len):
            frame = x[:, t]  # (batch_size, channels, height, width)
            
            # Apply convolutional layers
            feat = frame
            for i, layer in enumerate(self.conv_layers):
                feat = layer(feat)
            
            # Apply spatial attention
            attention_weights = self.spatial_attention(feat)
            feat = feat * attention_weights
            
            # Flatten spatial features
            feat = feat.view(batch_size, -1)  # (batch_size, spatial_feat_size)
            spatial_features.append(feat)
        
        # Stack temporal features
        temporal_input = torch.stack(spatial_features, dim=1)  # (batch_size, seq_len, spatial_feat_size)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(temporal_input)  # (batch_size, seq_len, lstm_hidden_size * 2)
        
        # Apply temporal attention
        attention_weights = self.temporal_attention(lstm_out)  # (batch_size, seq_len, 1)
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, lstm_hidden_size * 2)
        
        # Feature fusion
        fused_features = self.feature_fusion(attended_features)  # (batch_size, lstm_hidden_size // 2)
        
        return fused_features

class DuelingDQN(nn.Module):
    """Enhanced DQN with Dueling Architecture and CNN-LSTM backbone"""
    
    def __init__(self, input_channels=4, output_dim=6, sequence_length=8):
        super(DuelingDQN, self).__init__()
        
        # CNN-LSTM feature extractor
        self.cnn_lstm_block = CNNLSTMBlock(input_channels, sequence_length)
        feature_dim = LSTM_HIDDEN_SIZE // 2
        
        # Dueling DQN architecture
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # State value V(s)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)  # Advantage A(s,a)
        )
        
    def forward(self, x):
        # Extract features using CNN-LSTM
        features = self.cnn_lstm_block(x)
        
        # Compute value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# -------------------------
# Enhanced replay buffer with prioritized experience replay
# -------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def __len__(self):
        return len(self.buffer)
    
    def add(self, *args, priority=None):
        transition = Transition(*args)
        
        if priority is None:
            # Set max priority for new experiences
            max_priority = self.priorities.max() if self.buffer else 1.0
        else:
            max_priority = priority
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
            
        # Compute sampling probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Unpack samples
        s, a, r, ns, done = zip(*samples)
        s = np.stack(s).astype(np.float32)
        ns = np.stack(ns).astype(np.float32)
        a = np.asarray(a, dtype=np.int64)
        r = np.asarray(r, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        
        return s, a, r, ns, done, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# -------------------------
# Agent state container
# -------------------------
@dataclass
class AgentState:
    device: str = "cpu"
    model: Optional[nn.Module] = None
    target_model: Optional[nn.Module] = None
    opt: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    rb: Optional[PrioritizedReplayBuffer] = None
    steps: int = 0
    eps_step: int = 0
    epsilon: float = EPS_START
    rounds: int = 0
    save_dir: str = "models"
    save_path: str = "bbman_cnn_lstm_checkpoint.pth"
    frame_buffer: Optional[deque] = None
    beta: float = 0.4  # Importance sampling parameter
    beta_increment: float = 0.001

# -------------------------
# Enhanced optimization with prioritized experience replay
# -------------------------
def optimize(self):
    st = self.state
    if st.rb is None or len(st.rb) < max(MIN_REPLAY_TO_LEARN, BATCH_SIZE):
        return

    if st.steps % TRAIN_EVERY_K_STEPS != 0:
        return

    # Sample from prioritized buffer
    s, a, r, ns, done, weights, indices = st.rb.sample(BATCH_SIZE, st.beta)
    device = st.device

    s = torch.from_numpy(s).to(device)
    ns = torch.from_numpy(ns).to(device)
    a = torch.from_numpy(a).to(device)
    r = torch.from_numpy(r).to(device)
    done = torch.from_numpy(done).to(device)
    weights = torch.from_numpy(weights).to(device)

    # Current Q values
    current_q = st.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Target Q values (Double DQN)
    with torch.no_grad():
        # Use main network to select actions
        next_actions = st.model(ns).argmax(1)
        # Use target network to evaluate actions
        target_q = st.target_model(ns).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_values = r + GAMMA * target_q * (1.0 - done)

    # Compute TD errors for priority update
    td_errors = torch.abs(current_q - target_values)
    
    # Weighted loss
    loss = (weights * F.mse_loss(current_q, target_values, reduction='none')).mean()

    # Optimize
    st.opt.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(st.model.parameters(), max_norm=10.0)
    st.opt.step()
    
    # Update learning rate
    if st.scheduler:
        st.scheduler.step()

    # Update priorities in replay buffer
    st.rb.update_priorities(indices, td_errors.detach().cpu().numpy())
    
    # Update beta for importance sampling
    st.beta = min(1.0, st.beta + st.beta_increment)

    # Soft target update
    with torch.no_grad():
        for p, tp in zip(st.model.parameters(), st.target_model.parameters()):
            tp.data.mul_(1.0 - TAU).add_(TAU * p.data)

# -------------------------
# Required callbacks
# -------------------------
def setup(self):
    """
    Called once before the first game (in both train and eval modes).
    Initializes the enhanced CNN-LSTM DQN model.
    """
    self.state = AgentState()
    # Enhanced hyperparameters
    self.state.gamma = 0.99
    self.state.batch_size = 32  # Reduced for more stable training with larger model
    self.state.lr = 1e-4  # Lower learning rate for better stability
    self.state.epsilon = 1.0
    self.state.epsilon_min = 0.05
    self.state.epsilon_decay_steps = 300_000  # Linear decay over more steps
    self.state.epsilon_step = 0
    self.state.target_update_freq = 1000  # Less frequent hard updates since we use soft updates
    self.state.tau = 0.005  # Soft target update parameter
    self.state.memory_size = 100_000
    self.state.min_memory_size = 2_000  # Start learning only after enough samples
    self.state.save_path = "" #"models/bbman_enhanced_checkpoint_100.pth"
    # Check if MPS is available
    if torch.backends.mps.is_available():
        self.state.device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        self.state.device = torch.device("cpu")
        print("Using CPU")
    # self.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[bbman] Using device: {self.state.device}")

    n_actions = len(ACTIONS)
    model = DuelingDQN(input_channels=4, output_dim=n_actions, sequence_length=SEQUENCE_LENGTH)
    target = DuelingDQN(input_channels=4, output_dim=n_actions, sequence_length=SEQUENCE_LENGTH)
    target.load_state_dict(model.state_dict())

    self.state.model = model.to(self.state.device)
    self.state.target_model = target.to(self.state.device)
    
    # Initialize frame buffer for temporal sequences
    self.state.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

    # Create checkpoint directory
    os.makedirs(self.state.save_dir, exist_ok=True)

    # Load existing weights if available
    if os.path.exists(self.state.save_path):
        try:
            payload = torch.load(self.state.save_path, map_location=self.state.device)

            if isinstance(payload, dict):
                if "policy_net" in payload:
                    # Load legacy checkpoint format from train.py
                    self.state.model.load_state_dict(payload["policy_net"])
                    self.state.target_model.load_state_dict(payload["target_net"])
                    print(f"[bbman] Loaded legacy checkpoint from {self.state.save_path}")
                elif "model" in payload:
                    # Load newer callback format
                    self.state.model.load_state_dict(payload["model"])
                    self.state.target_model.load_state_dict(payload.get("target", payload["model"]))
                    print(f"[bbman] Loaded model checkpoint from {self.state.save_path}")
                else:
                    # Direct state_dict
                    self.state.model.load_state_dict(payload)
                    self.state.target_model.load_state_dict(payload)
                    print(f"[bbman] Loaded raw state_dict from {self.state.save_path}")
            else:
                self.state.model.load_state_dict(payload)
                self.state.target_model.load_state_dict(payload)
                print(f"[bbman] Loaded raw state_dict")
        except Exception as ex:
            print(f"[bbman] Could not load existing weights: {ex}")

def act(self, game_state: dict) -> str:
    """
    Enhanced action selection using CNN-LSTM model with epsilon-greedy exploration.
    """
    st = self.state
    st.steps += 1
    
    update_epsilon(st)

    # Feature extraction with temporal sequences
    feat = state_to_features(game_state, st.frame_buffer)  # (seq_len, channels, h, w)
    x = torch.from_numpy(feat[None, :]).to(st.device)  # (1, seq_len, channels, h, w)

    # Epsilon-greedy with improved exploration
    if random.random() < st.epsilon and st.rb is not None:
        # Improved exploration: avoid obviously bad actions
        valid_actions = []
        if game_state and game_state.get('self'):
            self_pos = game_state['self'][:2]
            field = game_state.get('field', np.zeros((17, 17)))
            
            for i, action in enumerate(ACTIONS):
                if action == "WAIT" or action == "BOMB":
                    valid_actions.append(i)
                elif action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    # Check if movement is valid
                    dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[action]
                    new_x, new_y = self_pos[0] + dx, self_pos[1] + dy
                    if (0 <= new_x < 17 and 0 <= new_y < 17 and 
                        field[new_x, new_y] != -1):  # Not a wall
                        valid_actions.append(i)
            
            if valid_actions:
                a_idx = random.choice(valid_actions)
            else:
                a_idx = random.randrange(len(ACTIONS))
        else:
            a_idx = random.randrange(len(ACTIONS))
    else:
        with torch.no_grad():
            qvals = st.model(x)  # (1, n_actions)
            a_idx = int(torch.argmax(qvals, dim=1).item())
    # print(IDX_TO_ACTION[a_idx])
    return IDX_TO_ACTION[a_idx]

def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called once per step to store transitions and train.
    """
    # Convert to sequential features
    s = state_to_features(old_game_state, deque(list(self.state.frame_buffer)[:-1], maxlen=SEQUENCE_LENGTH))
    ns = state_to_features(new_game_state, self.state.frame_buffer.copy())

    # Action index
    a_idx = ACTION_TO_IDX.get(self_action, ACTION_TO_IDX["WAIT"])

    # Enhanced reward shaping
    r = reward_from_events(events, old_game_state)

    # Not terminal inside step callback
    done = 0.0

    # Store transition if training
    if self.state.rb is not None:
        self.state.rb.add(s, a_idx, r, ns, done)
        optimize(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at end of each round to finalize updates and save.
    """
    self.state.rounds += 1
    
    # Terminal transition
    s = state_to_features(last_game_state, deque(list(self.state.frame_buffer)[:-1], maxlen=SEQUENCE_LENGTH))
    a_idx = ACTION_TO_IDX.get(last_action, ACTION_TO_IDX["WAIT"])
    r = reward_from_events(events, last_game_state)
    ns = np.zeros_like(s, dtype=np.float32)
    done = 1.0
    
    if self.state.rb is not None:
        self.state.rb.add(s, a_idx, r, ns, done)
        # Extra training at episode end
        for _ in range(5):
            optimize(self)

    # Reset frame buffer for next round
    self.state.frame_buffer.clear()

    # Save checkpoint
    if self.state.rounds % SAVE_EVERY_K_ROUNDS == 0:
        payload = {
            "model": self.state.model.state_dict(),
            "target": self.state.target_model.state_dict(),
            "optimizer": self.state.opt.state_dict(),
            "scheduler": self.state.scheduler.state_dict() if self.state.scheduler else None,
            "meta": {
                "episode": self.state.rounds,
                "arch": "cnn-lstm-dueling-dqn",
                "lr": LR,
                "sequence_length": SEQUENCE_LENGTH,
                "cnn_channels": CNN_CHANNELS,
                "lstm_hidden": LSTM_HIDDEN_SIZE,
            },
        }
        torch.save(payload, self.state.save_path)
        print(f"[bbman] Saved enhanced checkpoint at round {self.state.rounds} -> {self.state.save_path}")