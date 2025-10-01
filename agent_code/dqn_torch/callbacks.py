import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import events as e  # provided by the bomberman_rl framework
# The framework will import this module and call the hooks defined below.

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as ex:
    raise RuntimeError(
        "PyTorch is required for the dqn_torch agent. "
        "Install it with: pip install 'torch>=2.3'"
    ) from ex

# Bomberman action space used by the framework (string labels):
ACTIONS: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}

# ---- Hyperparameters you can tweak ----
GAMMA = 0.99
LR = 3e-4
BATCH_SIZE = 64
REPLAY_SIZE = 50000
MIN_REPLAY_TO_LEARN = 1000
TAU = 0.005  # soft update for target network
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 200_000  # steps until epsilon reaches EPS_END
TRAIN_EVERY_K_STEPS = 4
TARGET_SYNC_EVERY_K_STEPS = 1  # we use soft updates each step
SAVE_EVERY_K_ROUNDS = 25

Transition = namedtuple("Transition", ("s", "a", "r", "ns", "done"))


class DuelingDQN(nn.Module):
    """
    Lightweight dueling DQN for grid observations.
    Input is (C, H, W). We build compact conv layers so it runs fast on CPU.
    """
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # infer flatten size lazily
        self._feat_dim = None

        # Will be initialized after first forward pass
        self.adv_head = None
        self.val_head = None
        self.n_actions = n_actions

    def _init_heads(self, feat_dim: int):
        self.adv_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self.n_actions)
        )
        self.val_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        z = self.feature_extractor(x)
        if self._feat_dim is None:
            self._feat_dim = z.shape[1]
            self._init_heads(self._feat_dim)
        adv = self.adv_head(z)
        val = self.val_head(z)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        idxs = np.random.randint(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        s = torch.from_numpy(np.stack([b.s for b in batch])).float()
        a = torch.tensor([b.a for b in batch], dtype=torch.long)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32)
        ns = torch.from_numpy(np.stack([b.ns for b in batch])).float()
        done = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return Transition(s, a, r, ns, done)


@dataclass
class AgentState:
    step: int = 0
    round_id: int = 0
    epsilon: float = EPS_START
    last_state: Optional[np.ndarray] = None
    last_action_idx: Optional[int] = None
    last_game_state: Optional[dict] = None
    model: Optional[DuelingDQN] = None
    target_model: Optional[DuelingDQN] = None
    opt: Optional[optim.Optimizer] = None
    rb: Optional[ReplayBuffer] = None
    device: str = "cpu"
    save_dir: str = "models"
    save_path: str = "models/dqn_torch.pt"


# ----- Framework hooks -----

def setup(self):
    """
    Called once before the first game. Initialize model, buffer, etc.
    """
    self.state = AgentState()
    self.state.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 8 channels by default; see state_to_tensor() below. Adjust if you change features.
    in_channels = 8
    n_actions = len(ACTIONS)
    model = DuelingDQN(in_channels, n_actions)
    target = DuelingDQN(in_channels, n_actions)
    target.load_state_dict(model.state_dict())

    self.state.model = model.to(self.state.device)
    self.state.target_model = target.to(self.state.device)
    self.state.opt = optim.Adam(self.state.model.parameters(), lr=LR)
    self.state.rb = ReplayBuffer(REPLAY_SIZE)

    os.makedirs(self.state.save_dir, exist_ok=True)

    # Load weights if present
    if os.path.exists(self.state.save_path):
        try:
            payload = torch.load(self.state.save_path, map_location=self.state.device)
            self.state.model.load_state_dict(payload["model"])
            self.state.target_model.load_state_dict(payload["target"])
            self.state.state_dict = payload.get("meta", {})
            print("[dqn_torch] Loaded weights from", self.state.save_path)
        except Exception as ex:
            print("[dqn_torch] Could not load existing weights:", ex)


def act(self, game_state: dict) -> str:
    """
    Choose an action given the current game_state.
    We use epsilon-greedy on Q-values. Illegal actions are masked out.
    """
    if game_state is None:
        return "WAIT"

    obs = state_to_tensor(game_state)  # (C,H,W) float32
    legal_mask = legal_actions_mask(game_state)  # (A,) bool
    q = None

    # epsilon schedule
    s = self.state
    s.epsilon = EPS_END + max(0.0, (EPS_START - EPS_END) * (1.0 - min(1.0, s.step / EPS_DECAY_STEPS)))

    if random.random() < s.epsilon or not legal_mask.any():
        # random legal action
        legal_idxs = np.where(legal_mask)[0]
        a_idx = int(np.random.choice(legal_idxs)) if len(legal_idxs) else np.random.randint(len(ACTIONS))
    else:
        with torch.no_grad():
            t = torch.from_numpy(obs[None, ...]).float().to(s.device)  # (1,C,H,W)
            q = s.model(t)[0].cpu().numpy()  # (A,)
        # mask illegal moves by -inf
        q_masked = np.where(legal_mask, q, -1e9)
        a_idx = int(np.argmax(q_masked))

    # stash last state/action so game_events_occurred can build transition
    s.last_state = obs
    s.last_action_idx = a_idx
    s.last_game_state = game_state
    s.step += 1

    return IDX_TO_ACTION[a_idx]


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: List[str]):
    """
    Called by the framework after each step with transition info.
    We push to replay buffer and (periodically) learn.
    """
    s = self.state
    if old_game_state is None or new_game_state is None:
        return

    # Build transition
    old_obs = state_to_tensor(old_game_state)
    new_obs = state_to_tensor(new_game_state)
    a_idx = ACTION_TO_IDX.get(self_action, 4)  # default to WAIT if unknown

    # Compute reward from events (dense shaping helps Bomberman)
    r = reward_from_events(events)

    # If you want additional shaping, you can add it here:
    # r += small_potential_based_shaping(old_game_state, new_game_state)

    done = bool(e.ROUND_END in events)  # end of round signal
    s.rb.push(old_obs, a_idx, float(r), new_obs, done)

    # Learn periodically
    if len(s.rb) >= MIN_REPLAY_TO_LEARN and (s.step % TRAIN_EVERY_K_STEPS == 0):
        dqn_learn_step(s)

    # Soft-update target network every step
    if s.target_model is not None and s.model is not None:
        with torch.no_grad():
            for p, tp in zip(s.model.parameters(), s.target_model.parameters()):
                tp.data.lerp_(p.data, TAU)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called by the framework at the end of each round. We can save checkpoints here.
    """
    s = self.state
    s.round_id += 1

    # final transition already inserted in game_events_occurred, but we can
    # run a few extra gradient steps here if you want:
    for _ in range(4):
        if len(s.rb) >= MIN_REPLAY_TO_LEARN:
            dqn_learn_step(s)

    if s.round_id % SAVE_EVERY_K_ROUNDS == 0:
        payload = {
            "model": s.model.state_dict(),
            "target": s.target_model.state_dict(),
            "meta": {"round_id": s.round_id, "step": s.step},
        }
        torch.save(payload, s.save_path)
        print(f"[dqn_torch] Saved checkpoint after round {s.round_id} to {s.save_path}")


# ----- DQN update -----

def dqn_learn_step(s: AgentState):
    batch = s.rb.sample(BATCH_SIZE)
    s.model.train()

    device = s.device
    obs = batch.s.to(device)           # (B,C,H,W)
    next_obs = batch.ns.to(device)     # (B,C,H,W)
    actions = batch.a.to(device)       # (B,)
    rewards = batch.r.to(device)       # (B,)
    dones = batch.done.to(device)      # (B,)

    q = s.model(obs).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

    with torch.no_grad():
        # Double DQN: choose action via online net, evaluate via target net
        next_q_online = s.model(next_obs)
        next_actions = next_q_online.argmax(dim=1)  # (B,)
        next_q_target = s.target_model(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (1.0 - dones) * GAMMA * next_q_target

    loss = torch.nn.functional.smooth_l1_loss(q, target)

    s.opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(s.model.parameters(), 10.0)
    s.opt.step()


# ----- Reward shaping -----

def reward_from_events(events: List[str]) -> float:
    """
    Map framework events to scalar reward. Adjust to taste.
    """
    # Default sparse rewards from framework (if any) are not relied upon.
    # We define our own dense shaping consistent with common student solutions.
    mapping = {
        e.COIN_COLLECTED: +1.0,
        e.KILLED_OPPONENT: +5.0,
        e.SURVIVED_ROUND: +2.0,
        e.BOMB_DROPPED: +0.1,
        e.INVALID_ACTION: -0.3,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -2.0,
        e.WAITED: -0.02,         # discourage idling
        e.CRATE_DESTROYED: +0.2, # encourage opening space
        e.MOVED_TOWARDS_COIN: +0.03,
        e.MOVED_AWAY_FROM_COIN: -0.02,
        e.RUN_INTO_BOMB: -0.3,
    }
    return float(sum(mapping.get(ev, 0.0) for ev in events))


# ----- Feature extraction -----

def state_to_tensor(game_state: dict) -> np.ndarray:
    """
    Convert game_state (dict supplied by framework) to a fixed (C,H,W) tensor.
    This function is robust to small differences across forks.

    Channels (8):
      0: free tiles
      1: walls
      2: crates
      3: explosions now
      4: bombs (normalized timer)
      5: coins
      6: self position
      7: opponents
    """
    # Fallback grid size; framework default is usually 17x17
    H = W = 17
    field = np.array(game_state.get("field", np.zeros((H, W), dtype=np.int8)), dtype=np.int16)
    if field.ndim != 2:
        field = np.zeros((H, W), dtype=np.int16)

    H, W = field.shape
    free = (field == 0).astype(np.float32)
    # Some forks use -1 for walls / 1 for crates; some invert. Be permissive:
    walls = ((field == -1) | (field == 9)).astype(np.float32)  # 9 just in case
    crates = (field == 1).astype(np.float32)

    explosion_map = np.array(game_state.get("explosion_map", np.zeros((H, W), dtype=np.int8)), dtype=np.float32)
    explosions = (explosion_map > 0).astype(np.float32)

    bombs = np.zeros((H, W), dtype=np.float32)
    for b in game_state.get("bombs", []):
        # formats seen: ((x,y), timer) or (x,y,timer)
        if isinstance(b, tuple) and len(b) == 2 and isinstance(b[0], tuple):
            (x, y), t = b
        elif isinstance(b, tuple) and len(b) == 3:
            x, y, t = b
        else:
            continue
        if 0 <= x < W and 0 <= y < H:
            bombs[y, x] = max(0.0, float(t)) / 4.0  # normalize by default timer=4

    coins = np.zeros((H, W), dtype=np.float32)
    for c in game_state.get("coins", []):
        x, y = c
        if 0 <= x < W and 0 <= y < H:
            coins[y, x] = 1.0

    me = np.zeros((H, W), dtype=np.float32)
    self_info = game_state.get("self")
    if isinstance(self_info, tuple) and len(self_info) >= 2:
        _, _, _, (sx, sy) = self_info if len(self_info) == 4 else (None, None, None, (0, 0))
        if 0 <= sx < W and 0 <= sy < H:
            me[sy, sx] = 1.0

    opponents = np.zeros((H, W), dtype=np.float32)
    for other in game_state.get("others", []):
        # formats seen: (name, score, bomb_avail, (x,y))
        if isinstance(other, tuple) and len(other) >= 4 and isinstance(other[3], tuple):
            ox, oy = other[3]
            if 0 <= ox < W and 0 <= oy < H:
                opponents[oy, ox] = 1.0

    stacked = np.stack([free, walls, crates, explosions, bombs, coins, me, opponents], axis=0)  # (C,H,W)
    return stacked.astype(np.float32)


def legal_actions_mask(game_state: dict) -> np.ndarray:
    """
    Heuristic legality filter to mask obviously invalid moves.
    The framework also gives INVALID_ACTION events which we penalize,
    but masking helps learning.
    """
    H = W = 17
    field = np.array(game_state.get("field", np.zeros((H, W), dtype=np.int8)), dtype=np.int16)
    if field.ndim != 2:
        field = np.zeros((H, W), dtype=np.int16)
    H, W = field.shape

    # default: all legal
    mask = np.ones(len(ACTIONS), dtype=bool)

    # get self position
    self_info = game_state.get("self")
    if isinstance(self_info, tuple) and len(self_info) >= 4:
        _, _, bomb_avail, (sx, sy) = self_info
    else:
        bomb_avail, (sx, sy) = False, (0, 0)

    # block movement into walls/crates
    deltas = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
    for a in ["UP", "RIGHT", "DOWN", "LEFT"]:
        dx, dy = deltas[a]
        nx, ny = sx + dx, sy + dy
        if nx < 0 or nx >= W or ny < 0 or ny >= H:
            mask[ACTION_TO_IDX[a]] = False
        else:
            tile = field[ny, nx]
            # block if wall or crate (common encodings: -1=wall, 1=crate)
            if tile == -1 or tile == 1 or tile == 9:
                mask[ACTION_TO_IDX[a]] = False

    # forbid BOMB if not available
    if not bomb_avail:
        mask[ACTION_TO_IDX["BOMB"]] = False

    return mask
