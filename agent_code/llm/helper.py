import os
import pickle
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

def state_to_features(game_state: dict):
    field = game_state.get('field', np.zeros((17, 17)))
    self_info = game_state.get('self')
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros((17, 17)))
    

def check_valid_movement(field, self_info) -> List[str]:
    """
    Return the list of valid directional moves ["UP","RIGHT","DOWN","LEFT"].
    A move is valid if the target cell is within bounds and field[ny, nx] == 0.
    field: np.ndarray of shape (17,17) with values like -1 (wall), 0 (free), 1 (crate)
    self_info: tuple (name, score, bomb_possible, (x, y))
    """
    # Extract current (x,y) from self_info
    x = y = None
    x, y = get_self_pos(self_info)
    if x is None or y is None:
        return []
    h, w = field.shape
    # Note: arrays are indexed as field[ny, nx] = field[row, col] = field[y, x]
    deltas = {
        "UP": (0, -1),
        "RIGHT": (1, 0),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
    }
    valid = []
    for act, (dx, dy) in deltas.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and field[ny, nx] == 0:
            valid.append(act)
    return valid

def check_bomb_radius(field: np.ndarray, self_info, bombs: List, explosions: np.ndarray) -> bool:
    """
    Returns True if the agent is currently in a bomb explosion radius (now or imminent).
    Heuristics:
      1) explosion_map[y, x] > 0 means the tile is scheduled to explode (or burning)
      2) Otherwise, check LOS from any bomb along +/-x and +/-y until blocked
    """
    x, y = get_self_pos(self_info)
    if x is None or y is None:
        return False
    # 1) Trust the explosion_map (BFS-based countdown of future/current blasts)
    if isinstance(explosions, np.ndarray):
        if 0 <= y < explosions.shape[0] and 0 <= x < explosions.shape[1]:
            if explosions[y, x] > 0:
                return True  # already in a current/imminent blast zone
    # [web:2]
    # 2) Line-of-sight check from bombs (plus-shaped rays stop at walls/crates)
    # bombs entries are commonly (bx, by, timer); treat any timer >= 0 as potential
    h, w = field.shape
    blockers = lambda cx, cy: not is_free(field, cx, cy)  # non-free blocks rays
    for b in bombs or []:
        bx = by = None
        if isinstance(b, (tuple, list)):
            # Support (bx, by, t) or ((bx,by), t)
            if len(b) >= 3 and isinstance(b[0], (int, np.integer)) and isinstance(b[1], (int, np.integer)):
                bx, by = int(b[0]), int(b[1])
            elif len(b) >= 2 and isinstance(b[0], (tuple, list)) and len(b[0]) >= 2:
                bx, by = int(b[0][0]), int(b[0][1])
        if bx is None or by is None:
            continue
        # Same row: scan horizontally from bomb to self, stopping at blockers
        if by == y:
            step = 1 if x > bx else -1
            cx = bx
            while 0 <= cx < w:
                if cx == x:
                    return True
                # move one step first; the bomb cell itself is affected as well
                cx += step
                if not (0 <= cx < w): break
                if blockers(cx, y):  # wall/crate blocks further propagation
                    if cx == x:
                        # If self is exactly on the blocker cell, it's blocked (not in blast)
                        pass
                    break
        # Same column: scan vertically from bomb to self, stopping at blockers
        if bx == x:
            step = 1 if y > by else -1
            cy = by
            while 0 <= cy < h:
                if cy == y:
                    return True
                cy += step
                if not (0 <= cy < h): break
                if blockers(x, cy):
                    if cy == y:
                        pass
                    break
    return False

def get_self_pos(self_info) -> Optional[Tuple[int, int]]:
    # self_info format: (name, score, bomb_possible, (x, y))
    if isinstance(self_info, (tuple, list)) and len(self_info) >= 4:
        pos = self_info[3]
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
    return None

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def in_bounds(field: np.ndarray, x: int, y: int) -> bool:
    h, w = field.shape
    return 0 <= x < w and 0 <= y < h

def is_free(field: np.ndarray, x: int, y: int) -> bool:
    # Free cell = 0; walls/crates are non-zero
    return in_bounds(field, x, y) and field[x, y] == 0

def neighbors4(field: np.ndarray, x: int, y: int) -> List[Tuple[int,int]]:
    out = []
    for dx, dy in [(0,-1),(1,0),(0,1),(-1,0)]:
        nx, ny = x+dx, y+dy
        if is_free(field, nx, ny):
            out.append((nx, ny))
    return out

def safe_mask(field: np.ndarray, explosions: np.ndarray) -> np.ndarray:
    # Safe = free cell and explosions countdown == 0
    h, w = field.shape
    mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            mask[y, x] = (field[y, x] == 0) and (0 <= y < explosions.shape[0]) and (0 <= x < explosions.shape[1]) and (explosions[y, x] == 0)
    return mask

def is_cell_safe(field: np.ndarray, explosions: np.ndarray, x: int, y: int) -> bool:
    return is_free(field, x, y) and (explosions[y, x] == 0)

def safe_neighbors4(field: np.ndarray, explosions: np.ndarray, x: int, y: int) -> List[Tuple[int,int]]:
    out = []
    for nx, ny in neighbors4(field, x, y):
        if explosions[ny, nx] == 0:
            out.append((nx, ny))
    return out

def bfs_shortest_path(field: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int],
                      explosions: Optional[np.ndarray] = None) -> Optional[List[Tuple[int,int]]]:
    """Return shortest path (list of coordinates including start and goal) avoiding blocked/unsafe cells."""
    h, w = field.shape
    if not in_bounds(field, *start) or not in_bounds(field, *goal):
        return None
    if not is_free(field, *start) or not is_free(field, *goal):
        return None
    # Validate start and goal
    sx, sy = start
    gx, gy = goal
    if not in_bounds(field, sx, sy) or not in_bounds(field, gx, gy):
        return None
    if not is_free(field, sx, sy) or not is_free(field, gx, gy):
        return None
    # If explosion map is provided, validate shape
    if explosions is not None:
        if explosions.shape != field.shape:
            # Could ignore or treat as no hazard; here we treat it as invalid
            return None
    # Optional: if start == goal
    if start == goal:
        return [start]
    def passable(x: int, y: int) -> bool:
        if not is_free(field, x, y):
            return False
        if explosions is not None and explosions[y, x] > 0:
            return False
        return True
    q = deque([start])
    came: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    # 4-direction moves
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    while q:
        cur = q.popleft()
        if cur == goal:
            # reconstruct path
            path: List[Tuple[int,int]] = []
            node = cur
            while node is not None:
                path.append(node)
                node = came[node]
            path.reverse()
            return path
        cx, cy = cur
        for dx, dy in deltas:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(field, nx, ny):
                continue
            if (nx, ny) in came:
                continue
            if not passable(nx, ny):
                continue
            came[(nx, ny)] = (cx, cy)
            q.append((nx, ny))
    return None

def nearest_safe_coin(field: np.ndarray, self_info, coins: List[Tuple[int,int]], explosions: np.ndarray
                      ) -> Optional[Tuple[Tuple[int,int], int, List[Tuple[int,int]]]]:
    """
    Find the coin with the shortest safe path; returns (coin_pos, path_len, path) or None if none is safely reachable.
    """
    me = get_self_pos(self_info)
    if me is None or not coins:
        return None
    best = None
    for c in coins:
        if not in_bounds(field, c[0], c[1]):
            continue
        path = bfs_shortest_path(field, me, (c[0], c[1]), explosions)
        if path is None:
            continue
        plen = len(path) - 1
        if best is None or plen < best[1]:
            best = (c, plen, path)
    return best

def next_action_toward(from_pos: Tuple[int,int], to_pos: Tuple[int,int]) -> str:
    """Map a single-step move to an ACTION label."""
    fx, fy = from_pos
    tx, ty = to_pos
    if (tx, ty) == (fx, fy-1): return "UP"
    if (tx, ty) == (fx+1, fy): return "RIGHT"
    if (tx, ty) == (fx, fy+1): return "DOWN"
    if (tx, ty) == (fx-1, fy): return "LEFT"
    return "WAIT"

def bfs_distance(field: np.ndarray,
                 start: Tuple[int, int],
                 hazard: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute shortest (unweighted) distances from start to all reachable cells,
    avoiding blocked cells (non-zero in field) and optionally cells in hazard > 0.
    Returns a 2D array `dist` with float distances or np.inf if unreachable.
    """
    h, w = field.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)

    def is_free_cell(x: int, y: int) -> bool:
        # free if inside bounds and field[y, x] == 0
        return in_bounds(field, x, y) and (field[y, x] == 0)

    # Validate start
    x0, y0 = start
    if not in_bounds(field, x0, y0):
        return dist
    if field[y0, x0] != 0:
        return dist

    if hazard is not None:
        if hazard.shape != field.shape:
            # you can choose: ignore hazard, or treat as unreachable
            # Here we choose to ignore hazard if mismatch
            hazard = None
        else:
            # if the start is already hazardous, we may still want to compute escapes
            # but if you prefer, you can return early. Here I let BFS run.
            pass

    # BFS queue
    dq = deque()
    dist[y0, x0] = 0.0
    dq.append((x0, y0))

    # 4-direction neighbors
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while dq:
        x, y = dq.popleft()
        d0 = dist[y, x]
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            # quick bounds / visited check
            if not in_bounds(field, nx, ny):
                continue
            # if we already visited
            if dist[ny, nx] != np.inf:
                continue
            # passable: free cell, not in hazard
            if field[ny, nx] != 0:
                continue
            if hazard is not None and hazard[ny, nx] > 0:
                continue
            # all good: assign
            dist[ny, nx] = d0 + 1.0
            dq.append((nx, ny))

    return dist

def get_others_positions(others: List) -> List[Tuple[int,int]]:
    out = []
    for o in others or []:
        if isinstance(o, (tuple, list)) and len(o) >= 4:
            pos = o[3]
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                out.append((int(pos[0]), int(pos[1])))
    return out

def choose_coin_opponent_aware(field: np.ndarray, self_info, coins: List[Tuple[int,int]],
                               explosions: np.ndarray, others: List,
                               lead_margin: int = 1) -> Optional[Tuple[Tuple[int,int], List[Tuple[int,int]]]]:
    """
    Pick a coin where self ETA < opponents' ETA - lead_margin (tie-breaking by shortest path),
    returning (coin_pos, path). If none satisfy, fall back to the nearest safe coin.
    """
    me = get_self_pos(self_info)
    if me is None or not coins:
        return None
    self_dist = bfs_distance(field, me, explosions)
    opps = get_others_positions(others)
    opp_dists = [bfs_distance(field, op, explosions) for op in opps]

    candidates = []
    for c in coins:
        if not in_bounds(field, c[0], c[1]):
            continue
        sd = self_dist[c[0], c[1]]
        if not np.isfinite(sd):
            continue
        if opp_dists:
            od_min = min((od[c[0], c[1]] for od in opp_dists), default=np.inf)
        else:
            od_min = 0
        lead = sd - od_min
        if lead > lead_margin:
            path = bfs_shortest_path(field, me, c, explosions)
            if path is not None:
                candidates.append((c, int(sd), int(od_min), path))
    if candidates:
        # Choose coin with minimal self ETA (then maximal lead)
        candidates.sort(key=lambda t: (t[1], -(t[2]-t[1])))
        best = candidates[0]
        return best[0], best[3]

    # Fallback to nearest safe coin
    nearest = nearest_safe_coin(field, self_info, coins, explosions)
    if nearest is None:
        return None
    return nearest[0], nearest[2]

def coins_within_k_steps(field: np.ndarray, self_info, coins: List[Tuple[int,int]],
                         explosions: np.ndarray, k: int) -> List[Tuple[int,int]]:
    me = get_self_pos(self_info)
    if me is None or not coins:
        return []
    dist = bfs_distance(field, me, explosions)
    return [c for c in coins if in_bounds(field, c[0], c[1]) and np.isfinite(dist[c[0], c[1]]) and dist[c[0], c[1]] <= k]

def next_action_for_coin(field: np.ndarray, self_info, coin: Tuple[int,int],
                         explosions: np.ndarray) -> str:
    me = get_self_pos(self_info)
    if me is None or coin is None:
        return "WAIT"
    path = bfs_shortest_path(field, me, coin, explosions)
    if not path or len(path) < 2:
        return "WAIT"
    return next_action_toward(path[0], path[1])

def coin_collection_policy(field: np.ndarray, self_info, coins: List[Tuple[int,int]],
                           explosions: np.ndarray, others: List, lead_margin: int = 1) -> str:
    """
    High-level choice: pick opponent-aware coin if possible; else nearest safe coin; else WAIT.
    Returns ACTION string.
    """
    choice = choose_coin_opponent_aware(field, self_info, coins, explosions, others, lead_margin)
    if choice is None:
        return "WAIT"
    coin, path = choice
    if not path or len(path) < 2:
        return "WAIT"
    return next_action_toward(path[0], path[1])

## Plant Bomb Functions
def blast_cells_from(pos: Tuple[int,int], field: np.ndarray, blast_strength: int) -> np.ndarray:
    """
    Returns a boolean mask (same shape as field) with True where a bomb at `pos` would hit.
    Stops at rigid wall (-1). Blast reaches crates (value==1) but won't go past them.
    """
    h, w = field.shape
    mask = np.zeros((h, w), dtype=bool)
    x0, y0 = pos
    if not in_bounds(field, x0, y0):
        return mask
    mask[y0, x0] = True
    for dx, dy in DELTAS:
        for step in range(1, blast_strength + 1):
            nx, ny = x0 + dx*step, y0 + dy*step
            if not in_bounds(field, nx, ny):
                break
            if field[ny, nx] == -1:
                # rigid wall: block and stop
                break
            mask[ny, nx] = True
            if field[ny, nx] == 1:
                # crate: blast reaches crate but stops beyond it
                break
    return mask

def bfs_distance_avoid(field: np.ndarray, start: Tuple[int,int], avoid_mask: Optional[np.ndarray]=None, max_depth: Optional[int]=None) -> np.ndarray:
    """
    BFS distances (float) from start avoiding cells where avoid_mask is True.
    If max_depth set, BFS stops expanding beyond that depth.
    Returns array of distances with np.inf where unreachable.
    """
    h, w = field.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    sx, sy = start
    if not in_bounds(field, sx, sy) or not is_free(field, sx, sy):
        return dist
    if avoid_mask is not None and avoid_mask[sy, sx]:
        # starting tile is avoided -> still allow exploring escapes from it (optional)
        # Here we allow start (so agent can move out) but mark it as visited at 0.
        pass

    dq = deque()
    dist[sy, sx] = 0.0
    dq.append((sx, sy))

    while dq:
        x, y = dq.popleft()
        d = dist[y, x]
        if max_depth is not None and d >= max_depth:
            continue
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if not in_bounds(field, nx, ny):
                continue
            if dist[ny, nx] != np.inf:
                continue
            # must be walkable
            if not is_free(field, nx, ny):
                continue
            # avoid hazardous cells (if mask provided)
            if avoid_mask is not None and avoid_mask[ny, nx]:
                continue
            dist[ny, nx] = d + 1.0
            dq.append((nx, ny))
    return dist

def should_plant_bomb(game_state: Dict,
                      field_key: str = "field",
                      self_key: str = "self",
                      others_key: str = "others",
                      blast_strength: int = 3,
                      explosion_timer: int = 3,
                      allow_suicide: bool = False) -> Dict:
    """
    Decide if planting a bomb now is sensible.

    game_state: dict with entries:
      - field: np.ndarray (h,w) with 0 free, -1 wall, 1 crate
      - self: (x,y) tuple (agent position)
      - others: list of (x,y) opponent positions (may be empty)

    Returns dict:
      { "plant": bool, "reason": str, "targets": {...}, "escape_distance": d or None }
    """
    field = game_state.get(field_key)
    me = game_state.get(self_key)
    others = game_state.get(others_key, [])
    if field is None or me is None:
        return {"plant": False, "reason": "missing field or self position", "targets": None, "escape_distance": None}

    sx, sy = me
    # 1) compute blast footprint if we plant now
    blast_mask = blast_cells_from((sx, sy), field, blast_strength)

    # 2) detect targets in blast lines
    opponents_hit = []
    crates_hit = []
    h, w = field.shape
    for ox, oy in others:
        if in_bounds(field, ox, oy) and blast_mask[oy, ox]:
            opponents_hit.append((ox, oy))

    # crates
    ys, xs = np.where(blast_mask)
    for y, x in zip(ys, xs):
        if field[y, x] == 1:
            crates_hit.append((x, y))

    # 3) compute escape distances avoiding blast mask cells (we need to reach a safe cell before explosion)
    # We allow starting on a blast cell but must reach a cell outside blast_mask within explosion_timer steps.
    # Build avoid_mask that includes blast cells (the tiles that will be unsafe at explosion time).
    avoid_mask = blast_mask.copy()
    # BFS distances avoiding those cells (so we find distances to safe cells)
    dist = bfs_distance_avoid(field, (sx, sy), avoid_mask=avoid_mask, max_depth=explosion_timer)
    # Find any cell with dist <= explosion_timer and not in blast_mask (safe landing cell)
    safe_cells = np.where((dist <= explosion_timer) & (~avoid_mask))
    safe_positions = list(zip(safe_cells[1].tolist(), safe_cells[0].tolist()))  # (x,y)
    safe_distance = None
    if safe_positions:
        # minimal distance to a safe cell
        safe_distance = float(np.min(dist[~avoid_mask]))
    else:
        safe_distance = None

    # 4) decision logic
    reasons = []
    if opponents_hit:
        reasons.append(f"opponent_in_blast: {len(opponents_hit)}")
    if crates_hit:
        reasons.append(f"crate_in_blast: {len(crates_hit)}")

    # if no target (no opponent & no crate) -> usually don't plant
    if not opponents_hit and not crates_hit:
        return {"plant": False, "reason": "no opponent or crate in blast footprint", "targets": {"opponents": opponents_hit, "crates": crates_hit}, "escape_distance": safe_distance}

    # if escape impossible within timer -> don't plant unless allow_suicide and you will at least kill opponent
    if safe_distance is None:
        if opponents_hit and allow_suicide:
            return {"plant": True, "reason": "no escape but opponents will be hit and suicide allowed", "targets": {"opponents": opponents_hit, "crates": crates_hit}, "escape_distance": None}
        return {"plant": False, "reason": "no safe escape within explosion timer", "targets": {"opponents": opponents_hit, "crates": crates_hit}, "escape_distance": None}

    # if safe and target exists -> plant
    return {"plant": True, "reason": "safe escape available and target in blast", "targets": {"opponents": opponents_hit, "crates": crates_hit}, "escape_distance": safe_distance}
