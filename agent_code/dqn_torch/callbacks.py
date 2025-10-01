import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import events as e  # bomberman_rl event enums provided by the framework

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as ex:
    raise RuntimeError(
        "PyTorch is required for the dqn_torch agent. "
        "Install it with: pip install 'torch>=2.3'"
    ) from ex

# Action space used by the framework (string labels)
ACTIONS: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

# -------------------------
# Hyperparameters
# -------------------------
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 64
REPLAY_SIZE = 50_000
MIN_REPLAY_TO_LEARN = 1_000
TAU = 0.005                  # soft target update
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000    # linear decay steps
TRAIN_EVERY_K_STEPS = 4
SAVE_EVERY_K_ROUNDS = 10_000

Transition = namedtuple("Transition", ("s", "a", "r", "ns", "done"))

# -------------------------
# Model: MLP DQN
# -------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, input_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # (B, n_actions)

# -------------------------
# Simple replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, done = zip(*batch)
        # Stack into contiguous numpy arrays for fast torch.from_numpy
        s = np.stack(s).astype(np.float32)
        ns = np.stack(ns).astype(np.float32)
        a = np.asarray(a, dtype=np.int64)
        r = np.asarray(r, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32)
        return s, a, r, ns, done

# -------------------------
# Agent state container
# -------------------------
@dataclass
class AgentState:
    device: str = "cpu"
    model: Optional[nn.Module] = None
    target_model: Optional[nn.Module] = None
    opt: Optional[optim.Optimizer] = None
    rb: Optional[ReplayBuffer] = None
    steps: int = 0
    eps_step: int = 0
    epsilon: float = EPS_START
    rounds: int = 0
    save_dir: str = "models"
    save_path: str = "dqn_checkpoint_50000.pth"

# -------------------------
# Features: match MLP input
# -------------------------
INPUT_DIM = 17 * 17  # 289, matches training with flat field

def state_to_features(game_state: Optional[dict]) -> np.ndarray:
    """
    Convert the game_state dict into a flat float32 vector of length 289.
    Uses only 'field' to match the MLP DQN you trained.
    """
    if game_state is None:
        return np.zeros(INPUT_DIM, dtype=np.float32)
    field = game_state["field"].astype(np.float32).flatten()  # (289,)
    return field

# -------------------------
# Epsilon schedule (linear)
# -------------------------
def update_epsilon(st: AgentState):
    st.eps_step += 1
    frac = min(1.0, st.eps_step / EPS_DECAY_STEPS)
    st.epsilon = EPS_START + frac * (EPS_END - EPS_START)

# -------------------------
# Reward shaping
# -------------------------
def reward_from_events(events: List[str]) -> float:
    reward = 0.0
    if e.COIN_COLLECTED in events:
        reward += 10.0
    if e.KILLED_OPPONENT in events:
        reward += 20.0
    if e.INVALID_ACTION in events:
        reward -= 5.0
    if e.GOT_KILLED in events or e.KILLED_SELF in events:
        reward -= 20.0
    return reward

# -------------------------
# Optimization step
# -------------------------
def optimize(self):
    st = self.state
    if st.rb is None or len(st.rb) < max(MIN_REPLAY_TO_LEARN, BATCH_SIZE):
        return

    if st.steps % TRAIN_EVERY_K_STEPS != 0:
        return

    s, a, r, ns, done = st.rb.sample(BATCH_SIZE)
    device = st.device

    s = torch.from_numpy(s).to(device)
    ns = torch.from_numpy(ns).to(device)
    a = torch.from_numpy(a).to(device)
    r = torch.from_numpy(r).to(device)
    done = torch.from_numpy(done).to(device)

    # Q(s,a)
    q = st.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(ns,a')
    with torch.no_grad():
        nq = st.target_model(ns).max(1)[0]
        target = r + GAMMA * nq * (1.0 - done)

    loss = nn.MSELoss()(q, target)

    st.opt.zero_grad()
    loss.backward()
    st.opt.step()

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
    Initializes the MLP DQN policy/target with INPUT_DIM=289 to match training.
    """
    self.state = AgentState()
    self.state.device = "cuda" if torch.cuda.is_available() else "cpu"

    n_actions = len(ACTIONS)
    model = DQN(INPUT_DIM, n_actions)
    target = DQN(INPUT_DIM, n_actions)
    target.load_state_dict(model.state_dict())

    self.state.model = model.to(self.state.device)
    self.state.target_model = target.to(self.state.device)

    # Create checkpoint directory
    os.makedirs(self.state.save_dir, exist_ok=True)

    # Best-effort weight loading: try payload dict first, then raw state_dict
    if os.path.exists(self.state.save_path):
        try:
            payload = torch.load(self.state.save_dir + "/" + self.state.save_path, map_location=self.state.device)
            if isinstance(payload, dict) and "model" in payload:
                self.state.model.load_state_dict(payload["model"])
                self.state.target_model.load_state_dict(payload.get("target", payload["model"]))
                meta = payload.get("meta", {})
                self.state.rounds = int(meta.get("episode", 0))
                print("[dqn_torch] Loaded payload from", self.state.save_path)
            else:
                self.state.model.load_state_dict(payload)
                self.state.target_model.load_state_dict(payload)
                print("[dqn_torch] Loaded state_dict from", self.state.save_path)
        except Exception as ex:
            print("[dqn_torch] Could not load existing weights:", ex)

def setup_training(self):
    """
    Called once before the first game in training mode: sets optimizer and replay buffer.
    """
    # Optimizer and buffer
    self.state.opt = optim.Adam(self.state.model.parameters(), lr=LR)
    self.state.rb = ReplayBuffer(REPLAY_SIZE)

def act(self, game_state: dict) -> str:
    """
    Epsilon-greedy action selection on the MLP DQN with INPUT_DIM=289.
    """
    st = self.state
    st.steps += 1
    update_epsilon(st)

    # Feature extraction
    feat = state_to_features(game_state)  # (289,)
    x = torch.from_numpy(feat[None, :]).to(st.device)  # (1, 289)

    # Epsilon-greedy
    if random.random() < st.epsilon and st.rb is not None:
        a_idx = random.randrange(len(ACTIONS))
    else:
        with torch.no_grad():
            qvals = st.model(x)  # (1, n_actions)
            a_idx = int(torch.argmax(qvals, dim=1).item())
    return IDX_TO_ACTION[a_idx]

def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called once per step to store transitions and train.
    """
    # Convert to features
    s = state_to_features(old_game_state)
    ns = state_to_features(new_game_state)

    # Action index (default WAIT if missing)
    a_idx = ACTION_TO_IDX.get(self_action, ACTION_TO_IDX["WAIT"])

    # Reward shaping
    r = reward_from_events(events)

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
    s = state_to_features(last_game_state)
    a_idx = ACTION_TO_IDX.get(last_action, ACTION_TO_IDX["WAIT"])
    r = reward_from_events(events)
    ns = np.zeros_like(s, dtype=np.float32)
    done = 1.0
    if self.state.rb is not None:
        self.state.rb.add(s, a_idx, r, ns, done)
        # A few extra updates at episode end
        for _ in range(10):
            optimize(self)

    # Save every N rounds
    if self.state.rounds % SAVE_EVERY_K_ROUNDS == 0:
        # Save a robust payload containing both policy and target
        payload = {
            "model": self.state.model.state_dict(),
            "target": self.state.target_model.state_dict(),
            "meta": {
                "episode": self.state.rounds,
                "arch": "mlp-289",
                "lr": LR,
            },
        }
        torch.save(payload, self.state.save_path)
        print(f"[dqn_torch] Saved checkpoint at round {self.state.rounds} -> {self.state.save_path}")
