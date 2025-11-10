from __future__ import annotations
import os, random
from pathlib import Path
from collections import deque, namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ========= Config =========
torch.set_num_threads(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.95 
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAP = 80_000
WARMUP_STEPS = 400
TARGET_SYNC = 1_000
EPS_START = 1.0
EPS_END = 0.20 
EPS_DECAY_STEPS = 1_200_000
EPS_EVAL = 0.10
GRAD_CLIP = 10.0
SAVE_EVERY_ROUNDS = 100

POS_HISTORY = 6
STUCK_TICKS = 8

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
DIR = {"UP": (0,-1), "RIGHT": (1,0), "DOWN": (0,1), "LEFT": (-1,0)}
BOMB_TIMER = 4
BLAST_RADIUS = 3

# PHASE-AWARE REWARDS
COINS_REWARDS = {
    "COIN_COLLECTED": 50.0,
    "KILLED_SELF": -20.0,
    "INVALID_ACTION": -3.0,
    "WAITED": -1.5,
}

CRATES_REWARDS = {
    "COIN_COLLECTED": 30.0,
    "CRATE_DESTROYED": 8.0,
    "COIN_FOUND": 2.0,
    "BOMB_DROPPED": 0.3,
    "KILLED_SELF": -30.0,
    "INVALID_ACTION": -3.0,
    "WAITED": -1,
}

Transition = namedtuple("Transition", ("s","a","r","sp","done"))

# ========= Network =========
class SimpleQNet(nn.Module):
    def __init__(self, in_ch: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

# ========= Replay =========
class ReplayBuffer:
    def __init__(self, cap:int): self.buf=deque(maxlen=cap)
    def push(self,*args): self.buf.append(Transition(*args))
    def sample(self,n:int):
        b=random.sample(self.buf,n)
        return (torch.stack([x.s for x in b]),
                torch.tensor([x.a for x in b], dtype=torch.long, device=DEVICE),
                torch.tensor([x.r for x in b], dtype=torch.float32, device=DEVICE),
                torch.stack([x.sp for x in b]),
                torch.tensor([x.done for x in b], dtype=torch.bool, device=DEVICE))
    def __len__(self): return len(self.buf)

# ========= Helpers =========
def get_bomb_positions(gs)->List[Tuple[int,int,int]]:
    if not gs: return []
    out=[]
    for b in gs.get("bombs",[]):
        if isinstance(b,(list,tuple)):
            if len(b)==2 and isinstance(b[0],(list,tuple)):
                (x,y),t=b; out.append((int(x),int(y),int(t)))
            elif len(b)>=3: out.append((int(b[0]),int(b[1]),int(b[2])))
    return out

def compute_danger_map(field:np.ndarray,bombs,expl)->np.ndarray:
    H,W=field.shape; ttb=np.full((H,W),999,dtype=np.int32)
    if expl is not None: ttb[expl>0]=0
    for bx,by,t in bombs:
        if not (0<=bx<W and 0<=by<H): continue
        ttb[by,bx]=min(ttb[by,bx],t)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            for d in range(1,BLAST_RADIUS+1):
                nx,ny=bx+dx*d,by+dy*d
                if not (0<=nx<W and 0<=ny<H): break
                if field[ny,nx]==-1: break
                ttb[ny,nx]=min(ttb[ny,nx],t)
                if field[ny,nx]==1: break
    return ttb

def is_safe_tile(x,y,ttb,steps=1): return ttb[y,x] > steps

def find_escape_route(pos, field, ttb) -> bool:
    H,W=field.shape; x,y=pos
    q=deque([(x,y,0)]); seen={(x,y)}
    while q:
        cx,cy,st=q.popleft()
        if ttb[cy,cx] > st+1: return True
        if st>=6: continue
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=cx+dx,cy+dy
            if not (0<=nx<W and 0<=ny<H) or (nx,ny) in seen: continue
            if field[ny,nx]!=0 or ttb[ny,nx]<=st+1: continue
            seen.add((nx,ny)); q.append((nx,ny,st+1))
    return False

def count_adjacent_crates(pos, field):
    """Count crates in bomb radius."""
    x,y=pos; H,W=field.shape; count=0
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        for d in range(1,BLAST_RADIUS+1):
            nx,ny=x+dx*d,y+dy*d
            if not (0<=nx<W and 0<=ny<H): break
            if field[ny,nx]==-1: break
            if field[ny,nx]==1: count+=1; break
    return count

def should_bomb_here(pos, field, can_bomb, ttb):
    """True if bombing is useful AND safe."""
    if not can_bomb: return False
    x,y=pos; H,W=field.shape
    
    crates = count_adjacent_crates(pos, field)
    if crates < 1:
        return False
    
    fut=ttb.copy(); fut[y,x]=min(fut[y,x], BOMB_TIMER)
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        for d in range(1,BLAST_RADIUS+1):
            nx,ny=x+dx*d,y+dy*d
            if not (0<=nx<W and 0<=ny<H): break
            if field[ny,nx]==-1: break
            fut[ny,nx]=min(fut[ny,nx], BOMB_TIMER)
            if field[ny,nx]==1: break
    
    return find_escape_route(pos, field, fut)

def nearest_coin_distance(pos, coins):
    if not coins: return 999
    x,y=pos; return min(abs(x-cx)+abs(y-cy) for cx,cy in coins)

def state_to_features(gs:dict, ttb:np.ndarray) -> torch.Tensor|None:
    """5 channels: free, crates, coins, self, danger."""
    if gs is None: return None
    field=np.asarray(gs["field"])
    sx,sy=gs["self"][3]
    coins=gs.get("coins", [])

    ch_free =(field==0).astype(np.float32)
    ch_crate=(field==1).astype(np.float32)
    
    ch_coins=np.zeros_like(field,dtype=np.float32)
    for cx,cy in coins: ch_coins[cy,cx]=1.0

    ch_self=np.zeros_like(field,dtype=np.float32); ch_self[sy,sx]=1.0
    ch_danger=np.clip(1.0/(ttb+1),0,1).astype(np.float32)

    return torch.from_numpy(np.stack([
        ch_free, ch_crate, ch_coins, ch_self, ch_danger
    ])).float()

def detect_phase(gs)->str:
    if not gs: return "coins"
    field=np.asarray(gs["field"])
    if gs.get("others", []): return "adversary"
    if (field==1).any(): return "crates"
    return "coins"

def resolve_phase(gs)->str:
    env=os.getenv("PHASE","").strip().lower()
    return env if env in {"coins","crates","adversary"} else detect_phase(gs)

# ========= Agent State =========
class AgentState:
    def __init__(self,q,tgt,opt):
        self.q=q; self.tgt=tgt; self.opt=opt
        self.rb=ReplayBuffer(REPLAY_CAP)
        self.step=0; self.episode=0
        self.pos_hist=deque(maxlen=POS_HISTORY)
        self.last_pos=None
        self.stuck_ticks=0
        self.bomb_memory=deque(maxlen=3) 

# ========= I/O =========
def _models_dir(): return Path(__file__).parent/"models"
def _latest():
    mdir=_models_dir()
    if not mdir.exists(): return None
    m=list(mdir.glob("dqn_*.pth"))
    return max(m, key=lambda p:p.stat().st_mtime) if m else None

def setup(self):
    _models_dir().mkdir(parents=True, exist_ok=True)
    q=SimpleQNet(5,len(ACTIONS)).to(DEVICE); q.eval()
    mp=_latest()
    if mp:
        try:
            st=torch.load(mp,map_location=DEVICE)
            q.load_state_dict(st["q"]); print(f"[DQN] Loaded {mp.name}")
        except Exception as e: print(f"[DQN] Load: {e}")
    else: print("[DQN] Random init")
    self.q_eval=q; self._cache={}

def setup_training(self):
    _models_dir().mkdir(parents=True, exist_ok=True)
    q=SimpleQNet(5,len(ACTIONS)).to(DEVICE)
    tgt=SimpleQNet(5,len(ACTIONS)).to(DEVICE); tgt.load_state_dict(q.state_dict())
    opt=Adam(q.parameters(), lr=LR)
    self.state=AgentState(q,tgt,opt); self._cache={}
    mp=_latest()
    if mp:
        try:
            st=torch.load(mp,map_location=DEVICE)
            q.load_state_dict(st["q"]); tgt.load_state_dict(st["tgt"])
            opt.load_state_dict(st["opt"])
            self.state.step=st.get("step",0); self.state.episode=st.get("episode",0)
            print(f"[DQN] Resumed ep={self.state.episode}")
        except: print("[DQN] Resume failed, fresh start")
    else: print("[DQN] Fresh start")

# ========= Policy =========
def _eps(step):
    p=min(step/EPS_DECAY_STEPS,1.0)
    return EPS_START + (EPS_END-EPS_START)*p

def get_valid_actions(gs, field, ttb, phase:str):
    H,W=field.shape; x,y=gs["self"][3]; can_bomb=gs["self"][2]
    coins=gs.get("coins", [])
    valid=[True]*len(ACTIONS)

    for i,ac in enumerate(ACTIONS[:4]):
        dx,dy=DIR[ac]; nx,ny=x+dx,y+dy
        if not (0<=nx<W and 0<=ny<H): valid[i]=False; continue
        if field[ny,nx]!=0: valid[i]=False; continue
        if not is_safe_tile(nx,ny,ttb,1): valid[i]=False

    if phase == "coins":
        if coins and ttb[y,x] > 3:
            valid[4] = False
        else:
            valid[4] = is_safe_tile(x,y,ttb,1)
    else:
        valid[4] = is_safe_tile(x,y,ttb,1)

    # BOMB
    if phase == "coins":
        valid[5] = False
    else:
        valid[5] = should_bomb_here((x,y), field, can_bomb, ttb)

    return valid

def act(self, gs:dict) -> str:
    if gs is None: return "WAIT"
    field=np.asarray(gs["field"])
    bombs=get_bomb_positions(gs)
    expl=np.asarray(gs.get("explosion_map", np.zeros_like(field)))
    ttb=compute_danger_map(field,bombs,expl)
    feats=state_to_features(gs, ttb)
    if feats is None: return "WAIT"

    phase=resolve_phase(gs)
    vm=get_valid_actions(gs, field, ttb, phase)
    idxs=[i for i,v in enumerate(vm) if v]
    if not idxs: return "WAIT"

    train=hasattr(self,"state")
    net=self.state.q if train else self.q_eval
    eps=_eps(self.state.step) if train else EPS_EVAL

    x,y=gs["self"][3]
    coins=gs.get("coins", [])
    crate_count=count_adjacent_crates((x,y), field)

    if train:
        if self.state.last_pos == (x,y):
            self.state.stuck_ticks += 1
        else:
            self.state.stuck_ticks = 0
        self.state.last_pos = (x,y)

    self._cache["last"]={"s":feats.to(DEVICE),"pos":(x,y),"phase":phase}

    if train and self.state.stuck_ticks >= STUCK_TICKS and phase=="crates":
        if 5 in idxs and crate_count > 0:
            self._cache["last"]["a_idx"] = 5
            self.state.bomb_memory.append((x,y))
            return "BOMB"

    if random.random()<eps:
        if phase=="crates" and 5 in idxs and crate_count >= 2 and random.random() < 0.20:
            aidx = 5
        elif coins:
            closer=[i for i in idxs if i<4]
            if closer:
                best=[]
                cd=nearest_coin_distance((x,y),coins)
                for i in closer:
                    dx,dy=DIR[ACTIONS[i]]; nx,ny=x+dx,y+dy
                    if nearest_coin_distance((nx,ny),coins)<cd: best.append(i)
                aidx=random.choice(best) if best else random.choice(closer)
            else:
                aidx=random.choice(idxs)
        else:
            aidx=random.choice(idxs)
    else:
        with torch.no_grad():
            q=net(feats.unsqueeze(0).to(DEVICE)).squeeze(0).cpu().numpy()

        if phase == "coins":
            q[4] -= 5.0
            if coins:
                cd=nearest_coin_distance((x,y),coins)
                for i in range(4):
                    if vm[i]:
                        dx,dy=DIR[ACTIONS[i]]; nx,ny=x+dx,y+dy
                        if nearest_coin_distance((nx,ny),coins)<cd: q[i]+=1.5
        else: 
            q[4] -= 1.5  
            if 5 in idxs and crate_count >= 1:
                q[5] += 1.5 * crate_count
            # Avoid recently bombed locations
            if train:
                for bx,by in self.state.bomb_memory:
                    if abs(x-bx)+abs(y-by) <= 2:
                        if 5 in idxs: q[5] -= 2.0 

        q[~np.array(vm)]=-np.inf
        aidx=int(np.argmax(q))

    self._cache["last"]["a_idx"]=aidx
    if train:
        self.state.pos_hist.append((x,y))
        if aidx == 5:
            self.state.bomb_memory.append((x,y))
    return ACTIONS[aidx]

# ========= Rewards =========
def reward_from_events(events, phase:str)->float:
    table = CRATES_REWARDS if phase == "crates" else COINS_REWARDS
    return sum(table.get(e,0.0) for e in events)

def game_events_occurred(self, old_gs, self_action, new_gs, events):
    if not hasattr(self,"state") or new_gs is None: return
    c=self._cache.get("last")
    if c is None: return

    s=c["s"]; a=c.get("a_idx", ACTIONS.index(self_action))
    phase=c.get("phase","coins")

    field=np.asarray(new_gs["field"])
    bombs=get_bomb_positions(new_gs)
    expl=np.asarray(new_gs.get("explosion_map", np.zeros_like(field)))
    ttb=compute_danger_map(field,bombs,expl)
    sp=state_to_features(new_gs, ttb).to(DEVICE)

    r=reward_from_events(events, phase)
    self.state.rb.push(s,a,r,sp,False)
    self.state.step+=1

    if len(self.state.rb)>=max(WARMUP_STEPS,BATCH_SIZE):
        _train_step(self.state)

    if self.state.step % TARGET_SYNC == 0:
        self.state.tgt.load_state_dict(self.state.q.state_dict())

def end_of_round(self, last_gs, last_action, events):
    if not hasattr(self,"state"): return
    c=self._cache.pop("last", None)
    if c and last_action:
        s=c["s"]; a=c.get("a_idx", ACTIONS.index(last_action))
        phase=c.get("phase","coins")
        r=reward_from_events(events, phase)
        self.state.rb.push(s,a,r,s,True)

    self.state.episode+=1
    self.state.pos_hist.clear()
    self.state.last_pos=None
    self.state.stuck_ticks=0
    self.state.bomb_memory.clear()

    # Stats
    coins=events.count("COIN_COLLECTED")
    crates=events.count("CRATE_DESTROYED")
    if self.state.episode % 20 == 0:
        print(f"[DQN] Ep {self.state.episode:4d} | Coins:{coins} Crates:{crates} | "
              f"Eps:{_eps(self.state.step):.2f} | Buf:{len(self.state.rb)}")

    if self.state.episode % SAVE_EVERY_ROUNDS == 0: _save(self.state)

def _train_step(st):
    s,a,r,sp,d = st.rb.sample(BATCH_SIZE)
    with torch.no_grad():
        tgt=r + (~d).float()*GAMMA*st.tgt(sp).max(1).values
    q=st.q(s).gather(1,a.unsqueeze(1)).squeeze(1)
    loss=F.smooth_l1_loss(q,tgt)
    st.opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(st.q.parameters(), GRAD_CLIP)
    st.opt.step()

def _save(st):
    p=_models_dir()/f"dqn_ep{st.episode:06d}.pth"
    torch.save({"q":st.q.state_dict(),"tgt":st.tgt.state_dict(),
                "opt":st.opt.state_dict(),"step":st.step,"episode":st.episode}, p)
    print(f"[DQN] Saved {p.name}")