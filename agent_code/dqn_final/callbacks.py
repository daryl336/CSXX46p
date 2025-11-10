from __future__ import annotations
import math, time, random
from pathlib import Path
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 128
REPLAY_CAP = 120_000
WARMUP_STEPS = 2_000
TARGET_SYNC = 2_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 1_000_000  
EPS_EVAL = 0.05 
GRAD_CLIP = 10.0

SAVE_EVERY_ROUNDS = 100
KEEP_LAST = 5

SAFE_TTB_LIMIT       = 2

# Policy biases
WAIT_DISCOURAGE      = 0.25  
REVERSE_DISCOURAGE   = 0.15  
TWO_CYCLE_DISCOURAGE = 0.20 
BOMB_CRATE_BONUS     = 0.20 
COIN_PULL            = 0.05 

# Bomb loop control
BOMB_COOLDOWN_STEPS        = 10   
BOMB_NEAR_PENALTY_R        = 2 
BOMB_NEAR_DISCOURAGE       = 0.20
REPEAT_BOMB_TILE_COOLDOWN  = 60  
BOMB_LOOP_DISCOURAGE       = 0.35  

# Safe-bomb learning shaping
SHAPE_TO_SAFETY_GAIN = 0.08 
SHAPE_TO_SAFETY_COST = 0.08 
SHAPE_IMMINENT_PEN   = 0.80  
SHAPE_SHELTER_BONUS  = 0.10  
SHAPE_TTB_IMPROVE    = 0.03  

ACTIONS = ["UP","RIGHT","DOWN","LEFT","WAIT","BOMB"]
DIR = {"UP":(0,-1),"RIGHT":(1,0),"DOWN":(0,1),"LEFT":(-1,0)}
REVERSE = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}
Transition = namedtuple("Transition", ("s","a","r","sp","done"))

# =========================
# Network
# =========================
class ConvQNet(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )
    def forward(self, x): return self.head(self.conv(x))

# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, cap: int): self.buf = deque(maxlen=cap)
    def push(self, *args): self.buf.append(Transition(*args))
    def sample(self, n: int):
        idx = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        s  = torch.stack([b.s for b in batch]).to(DEVICE)
        a  = torch.tensor([b.a for b in batch], dtype=torch.long, device=DEVICE)
        r  = torch.tensor([b.r for b in batch], dtype=torch.float32, device=DEVICE)
        sp = torch.stack([b.sp for b in batch]).to(DEVICE)
        done = torch.tensor([b.done for b in batch], dtype=torch.bool, device=DEVICE)
        return s, a, r, sp, done
    def __len__(self): return len(self.buf)

# =========================
# Agent State
# =========================
class AgentState:
    def __init__(self, q, tgt, opt):
        self.q, self.tgt, self.opt = q, tgt, opt
        self.rb = ReplayBuffer(REPLAY_CAP)
        self.step = 0
        self.episode = 0
        self.last_s = None
        self.last_a = None

        self.my_bomb_pos = None
        self.last_bomb_step = -10**9
        self.last_bomb_pos = None
        self.bomb_site_last_step = {} 

        self.last_move = None
        self.pos_hist = deque(maxlen=6)

        self.prev_dist_to_safe = None
        self.prev_ttb_here = None
        self.took_shelter = False

# =========================
# Setup / Training
# =========================
def _ensure_dirs():
    Path(__file__).parent.joinpath("models").mkdir(parents=True, exist_ok=True)

def _latest_model_path():
    mdir = Path(__file__).parent / "models"
    if not mdir.exists(): return None
    patterns = ["dqn_final_ep*.pth", "dqn_final_ep*.pt", "dqn_final_*.pth", "dqn_final_*.pt",
                "dqn_works_ep*.pth", "dqn_works_ep*.pt", "dqn_works_*.pth", "dqn_works_*.pt"]
    cands = []
    for pat in patterns: cands.extend(mdir.glob(pat))
    if not cands: return None
    return max(cands, key=lambda p: p.stat().st_mtime)

def setup(self):
    _ensure_dirs()
    model_path = _latest_model_path()
    q = ConvQNet(in_ch=9, n_actions=len(ACTIONS)).to(DEVICE)
    self.q_infer = q

    self.my_bomb_pos = None
    self.last_bomb_pos = None
    self.last_bomb_step = -10**9

    if model_path:
        state = torch.load(model_path, map_location=DEVICE)
        q.load_state_dict(state["q"])
        msg = f"[dqn_final] Loaded model for inference: {model_path.resolve()}"
    else:
        msg = "[dqn_final] No model found; running untrained."
    print(msg, flush=True)

    # Metrics tracking for evaluation
    from metrics.metrics_tracker import MetricsTracker
    self.name = "DQN Final"
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.episode_active = False
    self.current_step = 0

def setup_training(self):
    _ensure_dirs()
    q   = ConvQNet(9, len(ACTIONS)).to(DEVICE)
    tgt = ConvQNet(9, len(ACTIONS)).to(DEVICE); tgt.load_state_dict(q.state_dict())
    opt = Adam(q.parameters(), lr=LR)
    self.state = AgentState(q, tgt, opt)

    model_path = _latest_model_path()
    if model_path:
        try:
            chk = torch.load(model_path, map_location=DEVICE)
            q.load_state_dict(chk["q"]); tgt.load_state_dict(chk["tgt"]); opt.load_state_dict(chk["opt"])
            self.state.step = chk.get("step", 0)
            self.state.episode = chk.get("episode", 0)
            msg = f"[dqn_final] Resumed from {model_path.name}"
        except Exception as e:
            msg = f"[dqn_final] Could not resume: {e}"
    else:
        msg = "[dqn_final] Starting fresh training."
    print(msg, flush=True)

    # Metrics tracking for evaluation
    from metrics.metrics_tracker import MetricsTracker
    self.name = "DQN Final"
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.episode_active = False
    self.current_step = 0

# =========================
# --- Helpers ---
# =========================
def _iter_bombs(bombs):
    """Yield bombs as (bx, by, t) supporting formats: ((x,y),t) and (x,y,t)."""
    for b in bombs:
        if isinstance(b, (list, tuple)):
            if len(b) == 2 and isinstance(b[0], (list, tuple)) and len(b[0]) == 2:
                (bx, by), t = b
                yield int(bx), int(by), int(t)
            elif len(b) == 3:
                bx, by, t = b
                yield int(bx), int(by), int(t)

def _blast_range(gs, default=3):
    """Prefer radius from state; fall back to 3 (±3)."""
    return int(gs.get("bomb_power",
               gs.get("blast_range",
               gs.get("explosion_range", default))))

def time_to_blast_grid(gs):
    field = np.asarray(gs["field"])
    bombs = gs.get("bombs", [])
    H, W = field.shape
    INF = 10**9
    ttb = np.full((H, W), INF, dtype=np.int32)

    R = _blast_range(gs, default=3)

    def mark_cross(cx, cy, timer):
        ttb[cy, cx] = min(ttb[cy, cx], timer)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx, cy
            for _ in range(R): 
                nx += dx; ny += dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    break
                if field[ny, nx] == -1: 
                    break
                ttb[ny, nx] = min(ttb[ny, nx], timer)
                if field[ny, nx] == 1:
                    break

    for bx, by, t in _iter_bombs(bombs):
        if 0 <= by < H and 0 <= bx < W:
            mark_cross(bx, by, int(t))

    ex = np.asarray(gs.get("explosion_map", np.zeros_like(field)))
    ttb = np.minimum(ttb, np.where(ex > 0, 0, ttb))
    return ttb

def compute_danger_maps(gs):
    ttb = time_to_blast_grid(gs)
    danger_now = (ttb == 0).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(ttb < 10**9, 1.0 / np.maximum(ttb, 1), 0.0).astype(np.float32)
    danger_next = inv.copy()
    danger_next[ttb > SAFE_TTB_LIMIT] *= 0.25
    danger_next = np.clip(danger_next, 0.0, 1.0)
    return danger_now, danger_next

def _in_blast_line(gs, bomb_xy, pos_xy):
    """True if pos is aligned with bomb with no blocking obstacle between."""
    field = np.asarray(gs["field"])
    (bx, by) = bomb_xy
    (x, y)   = pos_xy
    if x == bx:
        step = 1 if y > by else -1
        for yy in range(by + step, y, step):
            if field[yy, x] in (-1, 1): 
                return False
        return True
    if y == by:
        step = 1 if x > bx else -1
        for xx in range(bx + step, x, step):
            if field[y, xx] in (-1, 1):
                return False
        return True
    return False

def _is_sheltered_by_wall(gs, bomb_xy):
    (sx, sy) = gs["self"][3]
    return not _in_blast_line(gs, bomb_xy, (sx, sy))

def _timeaware_distance_to_safety(gs, bomb_xy):
    """
    Return the minimum number of steps to a tile that will still be safe
    upon arrival (ttb > arrival time and beyond SAFE_TTB_LIMIT).
    If unreachable, return 8 (bounded neutral).
    """
    field = np.asarray(gs["field"])
    H, W = field.shape
    (sx, sy) = gs["self"][3]
    ttb = time_to_blast_grid(gs)
    INF = 10**9

    from collections import deque
    q = deque()
    seen = set()
    q.append((sx, sy, 0)) 
    seen.add((sx, sy))
    best = INF

    def passable(x,y): return 0 <= x < W and 0 <= y < H and field[y, x] == 0

    while q:
        x, y, d = q.popleft()
        if ttb[y, x] > max(d, SAFE_TTB_LIMIT):
            best = d
            break
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            nd = d+1
            if passable(nx, ny) and (nx, ny) not in seen and ttb[ny, nx] > nd:
                seen.add((nx, ny))
                q.append((nx, ny, nd))
    return best if best < INF else 8

def _bomb_still_present(gs, pos):
    (bx, by) = pos
    for px, py, _ in _iter_bombs(gs.get("bombs", [])):
        if px == bx and py == by:
            return True
    return False

def _best_dir_toward_coin(gs):
    coins = gs.get("coins", [])
    if not coins: return None
    sx, sy = gs["self"][3]
    cx, cy = min(coins, key=lambda c: abs(c[0]-sx) + abs(c[1]-sy))
    best = None
    best_d = abs(cx - sx) + abs(cy - sy)
    for a,(dx,dy) in DIR.items():
        nx, ny = sx+dx, sy+dy
        d = abs(cx - nx) + abs(cy - ny)
        if d < best_d:
            best = a; best_d = d
    return best

def _has_escape_route(gs):
    field = np.asarray(gs["field"])
    H, W = field.shape
    sx, sy = gs["self"][3]
    ttb = time_to_blast_grid(gs)

    from collections import deque
    q = deque()
    seen = set([(sx, sy, 0)])
    q.append((sx, sy, 0)) 

    def passable(x,y):
        return 0 <= x < W and 0 <= y < H and field[y, x] == 0

    while q:
        x, y, d = q.popleft()
        if ttb[y, x] > max(d, SAFE_TTB_LIMIT):
            return True
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            nd = d+1
            if passable(nx, ny) and (nx, ny, nd) not in seen:
                if ttb[ny, nx] > nd:
                    seen.add((nx, ny, nd))
                    q.append((nx, ny, nd))
    return False

def _targets_in_blast(gs, origin_xy):
    """
    True if at least one crate or enemy is in the bomb's blast lines (±R), unblocked.
    """
    field  = np.asarray(gs["field"])
    others = {tuple(ag[3]) for ag in gs.get("others", [])}
    H, W   = field.shape
    R      = _blast_range(gs, default=3)
    bx, by = origin_xy

    def scan(dx, dy):
        x, y = bx, by
        for _ in range(R):
            x += dx; y += dy
            if x < 0 or x >= W or y < 0 or y >= H: return False
            if field[y, x] == -1: return False 
            if (x, y) in others: return True  
            if field[y, x] == 1:  return True 
        return False

    return scan(1,0) or scan(-1,0) or scan(0,1) or scan(0,-1)

def _should_bomb_here(gs):
    x, y = gs["self"][3]
    return _targets_in_blast(gs, (x, y))

def _looks_like_two_cycle(self, gs):
    if not hasattr(self, "state"): return False
    ph = self.state.pos_hist
    return len(ph) >= 3 and ph[-1] == ph[-3]

def _is_strong_two_cycle(self):
    if not hasattr(self, "state"): return False
    ph = self.state.pos_hist
    return len(ph) >= 4 and ph[-1] == ph[-3] and ph[-2] == ph[-4]

def _mask_invalid_and_dangerous_actions(gs):
    field = np.asarray(gs["field"])
    x, y = gs["self"][3]
    H, W = field.shape

    ttb = time_to_blast_grid(gs)
    bomb_tiles = {(bx, by) for bx, by, _ in _iter_bombs(gs.get("bombs", []))}

    mask = torch.zeros(len(ACTIONS), dtype=torch.float32, device=DEVICE)

    if (x, y) in bomb_tiles or ttb[y, x] <= SAFE_TTB_LIMIT:
        mask[ACTIONS.index("WAIT")] = -1e9

    for a, (dx,dy) in DIR.items():
        idx = ACTIONS.index(a)
        nx, ny = x+dx, y+dy
        if nx < 0 or nx >= W or ny < 0 or ny >= H:
            mask[idx] = -1e9; continue
        if field[ny, nx] != 0:           
            mask[idx] = -1e9; continue
        if (nx, ny) in bomb_tiles:       
            mask[idx] = -1e9; continue
        if ttb[ny, nx] <= SAFE_TTB_LIMIT: 
            mask[idx] = -1e9

    # BOMB availability
    if not bool(gs["self"][2]):
        mask[ACTIONS.index("BOMB")] = -1e9

    return mask

def _emergency_evac_action(gs):
    """
    If everything is masked and we're on a hot tile, pick the passable neighbor
    with the largest time-to-blast to avoid waiting to die.
    """
    field = np.asarray(gs["field"])
    H, W   = field.shape
    (x, y) = gs["self"][3]
    ttb    = time_to_blast_grid(gs)

    best_a, best_score = None, -1_000_000
    for a, (dx, dy) in DIR.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and field[ny, nx] == 0:
            score = int(ttb[ny, nx])
            if score > best_score:
                best_score, best_a = score, a
    return best_a

def _two_step_best_evac(gs):
    """
    Pick the first move whose best reachable tile after two steps has the largest
    time-to-blast. Helps when a 1-step evac still dies on the next tick in corridors.
    """
    field = np.asarray(gs["field"])
    H, W = field.shape
    (x, y) = gs["self"][3]
    ttb = time_to_blast_grid(gs)

    def passable(a, b): return 0 <= a < W and 0 <= b < H and field[b, a] == 0

    best_a, best_score = None, -1_000_000
    for a, (dx1, dy1) in DIR.items():
        x1, y1 = x + dx1, y + dy1
        if not passable(x1, y1):
            continue
        best_after_two = int(ttb[y1, x1])
        for dx2, dy2 in DIR.values():
            x2, y2 = x1 + dx2, y1 + dy2
            if passable(x2, y2):
                best_after_two = max(best_after_two, int(ttb[y2, x2]))
        if best_after_two > best_score:
            best_score, best_a = best_after_two, a
    return best_a

# =========================
# Policy (uses helpers above)
# =========================
def act(self, game_state: dict) -> str:
    feat = state_to_features(self, game_state)
    if feat is None:
        return random.choice(ACTIONS)

    feat = feat.unsqueeze(0).to(DEVICE)
    training = hasattr(self, "state")
    step = self.state.step if training else 0
    eps = (EPS_END + 0.5*(EPS_START - EPS_END)*(1 + math.cos(math.pi * min(1.0, step / EPS_DECAY_STEPS)))
           if training else EPS_EVAL)

    net = self.state.q if training else self.q_infer
    with torch.no_grad():
        q = net(feat).squeeze(0)  # (6,)

    mask = _mask_invalid_and_dangerous_actions(game_state)
    allowed = [i for i, m in enumerate(mask.tolist()) if m == 0.0]

    # emergency: if nothing allowed, try to evacuate instead of WAITing to die
    if not allowed:
        evac = _emergency_evac_action(game_state)
        return evac if evac is not None else "WAIT"

    x, y = game_state["self"][3]
    ttb  = time_to_blast_grid(game_state)
    if ttb[y, x] <= SAFE_TTB_LIMIT:
        evac = _emergency_evac_action(game_state)
        if evac is not None:
            if training and evac in DIR:
                self.state.last_move = evac
            return evac

    if training and self.state.my_bomb_pos is not None and _bomb_still_present(game_state, self.state.my_bomb_pos):
        bx, by = self.state.my_bomb_pos
        if _in_blast_line(game_state, (bx, by), (x, y)) and ttb[y, x] <= (SAFE_TTB_LIMIT + 1):
            evac = _emergency_evac_action(game_state) or _two_step_best_evac(game_state)
            if evac is not None:
                if evac in DIR:
                    self.state.last_move = evac
                return evac

    q_adj = q.clone()
    idx_wait = ACTIONS.index("WAIT")

    if training and self.state.my_bomb_pos is not None and _bomb_still_present(game_state, self.state.my_bomb_pos):
        q_adj[ACTIONS.index("BOMB")] = -1e9

    if idx_wait in allowed and len(allowed) > 1:
        q_adj[idx_wait] -= WAIT_DISCOURAGE

    last_move = getattr(self.state, "last_move", None) if training else None
    if training and last_move and last_move in REVERSE:
        idx_rev = ACTIONS.index(REVERSE[last_move])
        if idx_rev in allowed and len(allowed) > 1:
            q_adj[idx_rev] -= REVERSE_DISCOURAGE
            if _looks_like_two_cycle(self, game_state):
                q_adj[idx_rev] -= TWO_CYCLE_DISCOURAGE

    idx_bomb = ACTIONS.index("BOMB")
    if idx_bomb in allowed and _should_bomb_here(game_state) and _has_escape_route(game_state):
        if training:
            last_here = self.state.bomb_site_last_step.get((x, y), -10**9)
            since_here = step - last_here
            if since_here < REPEAT_BOMB_TILE_COOLDOWN:
                q_adj[idx_bomb] -= 0.5 

            since_global = step - self.state.last_bomb_step
            if since_global >= BOMB_COOLDOWN_STEPS:
                q_adj[idx_bomb] += BOMB_CRATE_BONUS

            if _is_strong_two_cycle(self):
                q_adj[idx_bomb] -= BOMB_LOOP_DISCOURAGE

            if self.state.last_bomb_pos is not None:
                bx, by = self.state.last_bomb_pos
                if abs(x - bx) + abs(y - by) <= BOMB_NEAR_PENALTY_R:
                    q_adj[idx_bomb] -= BOMB_NEAR_DISCOURAGE
        else:
            q_adj[idx_bomb] += BOMB_CRATE_BONUS

    coin_dir = _best_dir_toward_coin(game_state)
    if coin_dir is not None:
        idx_coin_dir = ACTIONS.index(coin_dir)
        if idx_coin_dir in allowed:
            q_adj[idx_coin_dir] += COIN_PULL

    if (training and random.random() < eps) or ((not training) and random.random() < eps):
        pool = [i for i in allowed if i != idx_wait] or allowed
        if training and _is_strong_two_cycle(self) and idx_bomb in pool:
            pool = [i for i in pool if i != idx_bomb] or pool
        a_idx = random.choice(pool)
    else:
        a_idx = allowed[int(torch.argmax(q_adj[allowed]).item())]

    if ACTIONS[a_idx] == "BOMB":
        if training:
            self.state.bomb_site_last_step[(x, y)] = step
            self.state.my_bomb_pos = (x, y)
            self.state.last_bomb_pos = (x, y)
            self.state.last_bomb_step = step
            self.state.prev_dist_to_safe = _timeaware_distance_to_safety(game_state, (x, y))
            self.state.prev_ttb_here = time_to_blast_grid(game_state)[y, x]
            self.state.took_shelter = False
        else:
            self.my_bomb_pos = (x, y)
            self.last_bomb_pos = (x, y)
            self.last_bomb_step = 0 

    if ACTIONS[a_idx] in DIR and training:
        self.state.last_move = ACTIONS[a_idx]

    if training:
        self.state.pos_hist.append(game_state["self"][3])
        self.state.last_s = state_to_features(self, game_state)
        self.state.last_a = a_idx

    # ======================================
    # METRICS TRACKING
    # ======================================
    if hasattr(self, 'metrics_tracker'):
        current_round = game_state.get('round', 0)
        current_step = game_state.get('step', 0)

        # Start new episode at step 1
        if current_step == 1:
            if self.episode_active and hasattr(self, 'last_game_state'):
                _end_episode_from_act(self, self.last_game_state)

            opponent_names = [o[0] for o in game_state.get('others', []) if o]
            scenario = "training" if self.train else "evaluation"
            self.metrics_tracker.start_episode(
                episode_id=current_round,
                opponent_types=opponent_names,
                map_name="default",
                scenario=scenario
            )
            self.episode_active = True
            self.current_step = 0

        if self.episode_active:
            self.current_step += 1

        self.last_game_state = game_state

    return ACTIONS[a_idx]


def _end_episode_from_act(self, game_state):
    """End episode and save metrics (called from act when new round detected)."""
    if not hasattr(self, 'metrics_tracker') or not self.episode_active:
        return

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

    self.metrics_tracker.end_episode(
        won=won,
        rank=rank,
        survival_steps=survival_steps,
        total_steps=total_steps,
        metadata={"mode": "evaluation"}
    )

    self.metrics_tracker.save()
    self.episode_active = False

# =========================
# Rewards
# =========================
def reward_from_events(self, events) -> float:
    """
    Use standardized evaluation rewards for fair comparison.
    Falls back to original DQN rewards if evaluation_rewards not available.
    """
    try:
        from evaluation_rewards import EVALUATION_REWARDS
        use_evaluation = True
    except ImportError:
        use_evaluation = False

    if use_evaluation:
        reward_sum = 0.0
        for ev in events:
            reward_sum += EVALUATION_REWARDS.get(ev, 0.0)
        return reward_sum
    else:
        # Original DQN rewards
        T = {
            "COIN_COLLECTED": 12.0, "CRATE_DESTROYED": 0.5, "KILLED_OPPONENT": 12.0,
            "SURVIVED_ROUND": 0.5, "INVALID_ACTION": -1.0, "KILLED_SELF": -15.0,
            "GOT_KILLED": -5.0, "WAITED": -0.4, "BOMB_DROPPED": 0.0, "BOMB_EXPLODED": 0.0,
        }
        r = 0.0
        for ev in events:
            r += T.get(ev, 0.0)
        return r

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    # Record events in metrics
    if hasattr(self, 'metrics_tracker') and self.episode_active:
        for event in events:
            self.metrics_tracker.record_event(event)
        self.metrics_tracker.record_action(self_action)

    if not hasattr(self, "state"): return

    s  = state_to_features(self, old_game_state)
    sp = state_to_features(self, new_game_state)
    if s is None or sp is None: return
    a = ACTIONS.index(self_action)

    r = reward_from_events(self, events)

    own_active = False
    if self.state.my_bomb_pos is not None:
        if _bomb_still_present(new_game_state, self.state.my_bomb_pos):
            own_active = True
        else:
            self.state.my_bomb_pos = None
            self.state.prev_dist_to_safe = None
            self.state.prev_ttb_here = None
            self.state.took_shelter = False

    if own_active:
        sx, sy = new_game_state["self"][3]
        ttb = time_to_blast_grid(new_game_state)
        bx, by = self.state.my_bomb_pos

        if _in_blast_line(new_game_state, (bx, by), (sx, sy)) and ttb[sy, sx] <= SAFE_TTB_LIMIT:
            r -= SHAPE_IMMINENT_PEN

        dist_safe = _timeaware_distance_to_safety(new_game_state, (bx, by))
        if self.state.prev_dist_to_safe is not None:
            if dist_safe < self.state.prev_dist_to_safe:
                r += SHAPE_TO_SAFETY_GAIN
            elif dist_safe > self.state.prev_dist_to_safe:
                r -= SHAPE_TO_SAFETY_COST
        self.state.prev_dist_to_safe = dist_safe

        if not self.state.took_shelter and _is_sheltered_by_wall(new_game_state, (bx, by)):
            r += SHAPE_SHELTER_BONUS
            self.state.took_shelter = True

        prev_ttb_here = self.state.prev_ttb_here
        cur_ttb_here  = ttb[sy, sx]
        if prev_ttb_here is not None and cur_ttb_here > prev_ttb_here:
            r += SHAPE_TTB_IMPROVE
        self.state.prev_ttb_here = cur_ttb_here

    self.state.rb.push(s, a, r, sp, False)
    self.state.step += 1

    if self.state.my_bomb_pos is not None and not _bomb_still_present(new_game_state, self.state.my_bomb_pos):
        self.state.my_bomb_pos = None
        self.state.prev_dist_to_safe = None
        self.state.prev_ttb_here = None
        self.state.took_shelter = False

    if len(self.state.rb) >= max(WARMUP_STEPS, BATCH_SIZE):
        _learn(self)

    if self.state.step % TARGET_SYNC == 0:
        self.state.tgt.load_state_dict(self.state.q.state_dict())

def end_of_round(self, last_game_state, last_action, events):
    # Record final events
    if hasattr(self, 'metrics_tracker') and self.episode_active:
        for event in events:
            self.metrics_tracker.record_event(event)
        if last_action:
            self.metrics_tracker.record_action(last_action)

        # Determine outcome
        import events as e
        won = e.SURVIVED_ROUND in events
        rank = 1 if won else 4
        survival_steps = self.current_step if hasattr(self, 'current_step') else 0
        total_steps = last_game_state.get('step', 400) if last_game_state else 400

        self.metrics_tracker.end_episode(
            won=won,
            rank=rank,
            survival_steps=survival_steps,
            total_steps=total_steps,
            metadata={"mode": "evaluation", "final_events": events}
        )
        self.metrics_tracker.save()
        self.episode_active = False

    if not hasattr(self, "state"): return
    s = state_to_features(self, last_game_state)
    if s is not None and last_action is not None:
        self.state.rb.push(s, ACTIONS.index(last_action), reward_from_events(self, events), s, True)

    self.state.episode = getattr(self.state, "episode", 0) + 1

    self.state.my_bomb_pos = None
    self.state.prev_dist_to_safe = None
    self.state.prev_ttb_here = None
    self.state.took_shelter = False
    self.state.last_move = None
    self.state.pos_hist.clear()

    print(f"[dqn_final] Round {self.state.episode} ended — steps={self.state.step}", flush=True)

    if self.state.episode % SAVE_EVERY_ROUNDS == 0:
        _save_round_ckpt(self)

# =========================
# Learn step
# =========================
def _learn(self):
    q, tgt, opt = self.state.q, self.state.tgt, self.state.opt
    s, a, r, sp, done = self.state.rb.sample(BATCH_SIZE)
    with torch.no_grad():
        q_next = tgt(sp).max(1).values
        target = r + (~done).float() * (GAMMA * q_next)
    q_sa = q(s).gather(1, a.view(-1,1)).squeeze(1)
    loss = F.smooth_l1_loss(q_sa, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), GRAD_CLIP)
    opt.step()

# =========================
# Saving
# =========================
def _save_round_ckpt(self):
    mdir = Path(__file__).parent / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    fname = f"dqn_final_ep{self.state.episode:06d}.pth"
    path = mdir / fname
    torch.save({
        "q": self.state.q.state_dict(),
        "tgt": self.state.tgt.state_dict(),
        "opt": self.state.opt.state_dict(),
        "step": self.state.step,
        "episode": self.state.episode,
        "timestamp": time.time(),
    }, path)


    print(f"[dqn_final] saved {path.name}", flush=True)

# =========================
# Features
# =========================
def state_to_features(self, game_state: dict) -> torch.Tensor | None:
    if game_state is None: return None

    field = np.asarray(game_state["field"])  # -1 wall, 0 free, 1 crate
    coins = game_state.get("coins", [])
    others = [ag[3] for ag in game_state.get("others", [])]
    (x, y) = game_state["self"][3]

    walls  = (field == -1).astype(np.float32)
    crates = (field ==  1).astype(np.float32)

    coins_map = np.zeros_like(field, dtype=np.float32)
    for (cx, cy) in coins: coins_map[cy, cx] = 1.0

    me = np.zeros_like(field, dtype=np.float32); me[y, x] = 1.0
    others_map = np.zeros_like(field, dtype=np.float32)
    for (ox, oy) in others: others_map[oy, ox] = 1.0

    danger_now, danger_next = compute_danger_maps(game_state)

    passable = (field == 0).astype(np.uint8)
    coin_dist = bfs_distance_map((x, y), coins, passable).astype(np.float32)

    my_bomb = np.zeros_like(field, dtype=np.float32)
    if hasattr(self, "state"):
        my_pos = getattr(self.state, "my_bomb_pos", None)
    else:
        my_pos = getattr(self, "my_bomb_pos", None)

    if my_pos is not None and _bomb_still_present(game_state, my_pos):
        bx, by = my_pos
        my_bomb[by, bx] = 1.0
    else:
        if hasattr(self, "state"): self.state.my_bomb_pos = None
        else: self.my_bomb_pos = None

    stacked = np.stack([
        walls, crates, coins_map, coin_dist, danger_now, danger_next, me, others_map, my_bomb
    ], axis=0)

    return torch.from_numpy(stacked)

# =========================
# Navigation helper
# =========================
def bfs_distance_map(origin, targets, passable):
    H, W = passable.shape
    INF = 10**9
    dist = np.full((H, W), INF, dtype=np.int32)
    from collections import deque
    q = deque()
    ox, oy = origin
    dist[oy, ox] = 0
    q.append((ox, oy))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and passable[ny, nx] and dist[ny, nx] == INF:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))

    if not targets:
        return np.ones_like(dist, dtype=np.float32)

    nearest = np.full((H, W), INF, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            if dist[y, x] == INF:
                nearest[y, x] = INF
            else:
                nearest[y, x] = min((abs(x-cx) + abs(y-cy)) for (cx, cy) in targets)
    nearest = nearest.astype(np.float32)
    maxd = np.max(nearest[nearest < INF]) if np.any(nearest < INF) else 1.0
    nearest[nearest >= INF] = maxd
    return nearest / maxd
