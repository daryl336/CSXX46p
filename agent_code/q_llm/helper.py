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
    

def check_valid_movement(field, self_info, bombs) -> List[str]:
    """
    Return the list of valid directional moves ["UP","RIGHT","DOWN","LEFT"].
    A move is valid if the target cell is within bounds and field[nx, ny] == 0.
    field: np.ndarray of shape (17,17) with values like -1 (wall), 0 (free), 1 (crate)
    self_info: tuple (name, score, bomb_possible, (x, y))
    """
    # Extract current (x,y) from self_info
    x = y = None
    x, y = get_self_pos(self_info)
    if x is None or y is None:
        return []
    w, h = field.shape
    def mark_bomb(bx, by):
        if not (0 <= bx < w and 0 <= by < h):
            return
        field[bx, by] = -1
    def mark_blast_zone(bx, by):
        if not (0 <= bx < w and 0 <= by < h):
            return
        field[bx, by] = -1
        for dx, dy in DELTAS:
            cx, cy = bx, by
            for _ in range(3):  # typical blast radius = 3 tiles
                cx += dx; cy += dy
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                if field[cx, cy] == -1:  # wall blocks blast
                    break
                field[cx, cy] = -1
                if field[cx, cy] == 1:  # crate stops blast too
                    break
    for b in bombs or []:
        if b[1] == 0:
            # Mark explosion since it is going to blow.
            if isinstance(b, (tuple, list)):
                if len(b) >= 3 and isinstance(b[0], (int, np.integer)) and isinstance(b[1], (int, np.integer)):
                    mark_blast_zone(int(b[0]), int(b[1]))
                elif len(b) >= 2 and isinstance(b[0], (tuple, list)):
                    bx, by = b[0]
                    mark_blast_zone(int(bx), int(by))
        else:
            # Mark bomb will do
            if isinstance(b, (tuple, list)):
                if len(b) >= 3 and isinstance(b[0], (int, np.integer)) and isinstance(b[1], (int, np.integer)):
                    mark_bomb(int(b[0]), int(b[1]))
                elif len(b) >= 2 and isinstance(b[0], (tuple, list)):
                    bx, by = b[0]
                    mark_bomb(int(bx), int(by))
    deltas = {
        "UP": (0, -1),
        "RIGHT": (1, 0),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
    }
    valid = []
    for act, (dx, dy) in deltas.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and field[nx, ny] == 0:
            valid.append(act)
    return valid

def check_bomb_radius_and_escape(field: np.ndarray, self_info, bombs: List, explosions: np.ndarray):
    """
    Returns (in_danger, escape_dir)
      - in_danger: True if agent is in a current or imminent bomb blast
      - escape_dir: suggested safe move ("UP", "DOWN", "LEFT", "RIGHT", or "WAIT")
    """
    x, y = get_self_pos(self_info)
    if x is None or y is None:
        return {"in_bomb_radius":"yes", "in_danger":"no", "escape_bomb_action":"WAIT"}
    w, h = field.shape
    # --- Step 1: Identify unsafe tiles ---
    unsafe = np.zeros_like(field, dtype=bool)
    # (a) Mark tiles that are currently exploding or scheduled
    if isinstance(explosions, np.ndarray):
        unsafe |= explosions > 0
    
    # (b) Mark bomb blast zones manually (cross-shaped)
    def mark_blast_zone(bx, by):
        if not (0 <= bx < w and 0 <= by < h):
            return
        unsafe[bx, by] = True
        for dx, dy in DELTAS:
            cx, cy = bx, by
            for _ in range(3):  # typical blast radius = 3 tiles
                cx += dx; cy += dy
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                if field[cx, cy] == -1:  # wall blocks blast
                    break
                unsafe[cx, cy] = True
                if field[cx, cy] == 1:  # crate stops blast too
                    break
    for b in bombs or []:
        if isinstance(b, (tuple, list)):
            if len(b) >= 3 and isinstance(b[0], (int, np.integer)) and isinstance(b[1], (int, np.integer)):
                mark_blast_zone(int(b[0]), int(b[1]))
            elif len(b) >= 2 and isinstance(b[0], (tuple, list)):
                bx, by = b[0]
                mark_blast_zone(int(bx), int(by))
    in_danger = unsafe[x, y]
    # --- Step 2: If not in danger, stay put ---
    if not in_danger:
        return {"in_bomb_radius":"no", "in_danger":"no", "escape_bomb_action":"WAIT"}
    # --- Step 3: BFS to find nearest safe tile ---
    # Compute distance grid from current location
    self_dist = bfs_distance(field, (x, y), explosions)
    # Get the first move toward nearest safe cell
    escape_dir = find_escape_direction(field, (x, y), self_dist, bombs, explosions)
    # print("Recommended escape move:", escape_dir)
    return {"in_bomb_radius":"yes", "in_danger":"yes", "escape_bomb_action":escape_dir}

def find_escape_direction(field: np.ndarray,
                          start: Tuple[int, int],
                          dist: np.ndarray,
                          bombs: List = [],
                          hazard: Optional[np.ndarray] = None,
                          tolerance: float = 1e-3,
                          debug: bool = False) -> Optional[str]:
    """
    Find immediate direction ("UP","DOWN","LEFT","RIGHT") from `start` (x, y)
    that leads along a shortest path to the nearest safe cell (hazard == 0).
    Returns None if no safe reachable cell or on failure.
    
    NOTES:
    - `field` and `dist` are indexed as [x, y] (numpy convention).
    - `start` must be (x, y) where 0 <= x < width and 0 <= y < height.
    """
    w, h = field.shape
    x0, y0 = int(start[0]), int(start[1])
    if hazard is None:
        hazard = np.zeros_like(field, dtype=np.float32)
    def mark_bomb(bx, by):
        if not (0 <= bx < w and 0 <= by < h):
            return
        field[bx, by] = -1
    for b in bombs or []:
        if isinstance(b, (tuple, list)):
            if len(b) >= 3 and isinstance(b[0], (int, np.integer)) and isinstance(b[1], (int, np.integer)):
                mark_bomb(int(b[0]), int(b[1]))
            elif len(b) >= 2 and isinstance(b[0], (tuple, list)):
                bx, by = b[0]
                mark_bomb(int(bx), int(by))
    # Basic bounds check
    if not (0 <= x0 < w and 0 <= y0 < h):
        if debug: print("[find_escape_direction] start out of bounds:", start)
        return None
    # Mask of candidate safe cells: reachable (finite dist), hazard <= 0, passable (field == 0)
    reachable = np.isfinite(dist) & (dist > 0)
    safe_mask = reachable & (hazard <= 0) & (field == 0)
    if not np.any(safe_mask):
        if debug: print("[find_escape_direction] no safe candidate cells")
        return None
    # Vectorized: pick the cell with minimal distance among safe_mask
    # Replace non-candidates with +inf and argmin
    candidate_dist = np.where(safe_mask, dist, np.inf)
    # flat_idx = int(np.argmax(candidate_dist))           # flattened index
    # tx, ty = np.unravel_index(flat_idx, dist.shape)     # row (y), col (x)
    finite_mask = np.isfinite(candidate_dist)
    if not np.any(finite_mask):
        flat_idx = None  # or handle safely
        if debug: print("[find_escape_direction] no safe candidate cells. flat_idx is none")
        return None
    else:
        flat_idx = np.argmax(np.where(finite_mask, candidate_dist, -np.inf))
        tx, ty = np.unravel_index(flat_idx, dist.shape)
    target = (int(tx), int(ty))
    if debug: print(f"[find_escape_direction] chosen target {target} with dist {dist[tx,ty]}")
    # If already at a safe spot, no movement required
    if target == (x0, y0):
        if debug: print("[find_escape_direction] already at target (safe)")
        return None
    # Backtrack from target to start following dist -> dist-1 -> ...
    directions = {
        "UP":    (0, -1),
        "DOWN":  (0,  1),
        "LEFT":  (-1, 0),
        "RIGHT": (1,  0),
    }
    neighbor_offsets = list(directions.values())
    # Determine whether exact integer check is appropriate
    dist_is_int = np.issubdtype(dist.dtype, np.integer)
    cx, cy = target
    path = [(cx, cy)]
    max_steps = h * w + 5
    steps = 0
    while (cx, cy) != (x0, y0):
        steps += 1
        if steps > max_steps:
            if debug: print("[find_escape_direction] exceeded max_steps during backtrack")
            return None
        current_dist = dist[cx, cy]
        if not np.isfinite(current_dist):
            if debug: print("[find_escape_direction] non-finite dist at current backtrack cell", (cx, cy))
            return None
        found_prev = False
        # Try predecessors (cells that are one step closer to the start)
        for ddx, ddy in neighbor_offsets:
            nx, ny = cx - ddx, cy - ddy   # predecessor coordinates
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            nd = dist[nx, ny]
            if not np.isfinite(nd):
                continue
            if dist_is_int:
                if int(nd) == int(current_dist) - 1:
                    cx, cy = int(nx), int(ny)
                    path.append((cx, cy))
                    found_prev = True
                    break
            else:
                if np.isclose(nd, current_dist - 1.0, atol=tolerance):
                    cx, cy = int(nx), int(ny)
                    path.append((cx, cy))
                    found_prev = True
                    break
        if not found_prev:
            if debug: print("[find_escape_direction] no predecessor found from", (cx, cy))
            return None
    # path is [target, ..., start]
    if len(path) < 2:
        if debug: print("[find_escape_direction] path too short")
        return None
    first_step = path[-2]  # the cell you must move to from start
    dx, dy = first_step[0] - x0, first_step[1] - y0
    for name, (ddx, ddy) in directions.items():
        if (dx, dy) == (ddx, ddy):
            if debug: print("[find_escape_direction] move:", name)
            return name
    if debug: print("[find_escape_direction] direction not found for delta:", (dx, dy))
    return None

### Coins Function
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
    w, h = field.shape
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
    w, h = field.shape
    mask = np.zeros((w, h), dtype=bool)
    for y in range(h):
        for x in range(w):
            mask[x, y] = (field[x, y] == 0) and (0 <= y < explosions.shape[0]) and (0 <= x < explosions.shape[1]) and (explosions[x, y] == 0)
    return mask

def is_cell_safe(field: np.ndarray, explosions: np.ndarray, x: int, y: int) -> bool:
    return is_free(field, x, y) and (explosions[x, y] == 0)

def safe_neighbors4(field: np.ndarray, explosions: np.ndarray, x: int, y: int) -> List[Tuple[int,int]]:
    out = []
    for nx, ny in neighbors4(field, x, y):
        if explosions[nx, ny] == 0:
            out.append((nx, ny))
    return out

def bfs_shortest_path(field: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                      explosions: Optional[np.ndarray] = None) -> Optional[List[Tuple[int, int]]]:
    """
    Return shortest path (list of (x,y) coordinates including start and goal)
    avoiding blocked or unsafe cells.
    Convention: field[x, y] indexing.
    """
    sx, sy = start
    gx, gy = goal
    # Validate start/goal
    if not (in_bounds(field, sx, sy) and in_bounds(field, gx, gy)):
        return None
    if not (is_free(field, sx, sy) and is_free(field, gx, gy)):
        return None
    if start == goal:
        return [start]
    if explosions is not None and explosions.shape != field.shape:
        return None
    def passable(x: int, y: int) -> bool:
        if not is_free(field, x, y):
            return False
        if explosions is not None and explosions[x, y] > 0:
            return False
        return True
    q = deque([start])
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    while q:
        cx, cy = q.popleft()
        if (cx, cy) == goal:
            # reconstruct path
            path = []
            node = (cx, cy)
            while node is not None:
                path.append(node)
                node = came[node]
            return path[::-1]

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

def bfs_shortest_path_crate(field: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                            explosions: Optional[np.ndarray] = None) -> Optional[List[Tuple[int, int]]]:
    """
    Shortest path from start to:
      - goal (if goal is passable), OR
      - one of the passable neighbors of the goal (if goal is a crate / not passable).
    Returns path (list of (x,y)) ending on the passable cell next to the crate (not on the crate).
    """
    sx, sy = start
    gx, gy = goal

    # Basic bounds check
    if not (in_bounds(field, sx, sy) and in_bounds(field, gx, gy)):
        return None
    if start == goal and is_free(field, sx, sy):
        return [start]

    # passable predicate (same semantics as your other BFS)
    def passable(x: int, y: int) -> bool:
        if not in_bounds(field, x, y):
            return False
        if not is_free(field, x, y):
            return False
        if explosions is not None and explosions[x, y] > 0:
            return False
        return True

    # If goal is a normal passable cell, just do usual BFS termination on goal
    goal_is_passable = is_free(field, gx, gy) and (explosions is None or explosions[gx, gy] == 0)

    # If goal is a crate (not passable), compute the set of passable neighbour targets.
    neighbour_targets = None
    if not goal_is_passable:
        neighbour_targets = []
        for dx, dy in DELTAS:
            nx, ny = gx + dx, gy + dy
            if passable(nx, ny):
                neighbour_targets.append((nx, ny))
        if not neighbour_targets:
            # no free neighbour to stand on -> crate unreachable
            return None
        neighbour_targets = set(neighbour_targets)  # for O(1) membership checks

    # BFS that stops when it reaches either the goal (if passable) or any neighbour target.
    q = deque([start])
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while q:
        cx, cy = q.popleft()

        # termination checks
        if goal_is_passable and (cx, cy) == (gx, gy):
            # reached goal itself
            node = (cx, cy)
            path = []
            while node is not None:
                path.append(node)
                node = came[node]
            return path[::-1]

        if not goal_is_passable and (cx, cy) in neighbour_targets:
            # reached a passable neighbour of crate -> return path to that neighbour
            node = (cx, cy)
            path = []
            while node is not None:
                path.append(node)
                node = came[node]
            return path[::-1]

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

    # nothing found
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
        if (best is None or
            plen < best[1] or
            (plen == best[1] and (c[0], c[1]) < (best[0][0], best[0][1]))):
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
                 hazard: Optional[np.ndarray] = None,
                 dtype: np.dtype = np.float32) -> np.ndarray:
    """
    BFS distances (unweighted) from start -> returns float array with np.inf for unreachable.
    Coordinate convention: start is (x, y) where x is column, y is row; arrays are indexed [x, y].
    """
    w, h = field.shape
    # allocate dist as float with np.inf sentinel
    dist = np.full((w, h), np.inf, dtype=dtype)
    x0, y0 = int(start[0]), int(start[1])
    # bounds check
    if not (0 <= x0 < w and 0 <= y0 < h):
        print("Bound check failed")
        return dist
    # # start must be passable (field == 0)
    # if field[x0, y0] != 0:
    #     return dist
    # handle hazard shape mismatch by ignoring hazard (same as your original choice)
    if hazard is not None and hazard.shape != field.shape:
        hazard = None
    # Precompute passable mask: passable == (field == 0) and not hazardous
    passable = (field == 0)
    if hazard is not None:
        # treat hazard > 0 as blocked
        passable = passable & (hazard <= 0)
    # if not passable[x0, y0]:
    #     print(f"Not passable {[x0, y0]}")
    #     return dist
    dq = deque()
    dist[x0, y0] = 0.0
    dq.append((x0, y0))
    # local references for speed
    dist_arr = dist
    passable_arr = passable
    inf = np.inf
    # neighbor offsets (dx, dy)
    deltas = ((0, -1), (1, 0), (0, 1), (-1, 0))
    # BFS loop
    while dq:
        x, y = dq.popleft()
        d0 = dist_arr[x, y]
        # iterate neighbors
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            # inline bounds check
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            # passable and not visited
            if not passable_arr[nx, ny]:
                continue
            if dist_arr[nx, ny] != inf:
                continue
            dist_arr[nx, ny] = d0 + 1.0
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
    Pick a coin where self ETA is at least `lead_margin` faster than opponents:
      od_min - sd > lead_margin
    Tie-breaking: smaller self ETA, then larger lead (od_min - sd), then smaller x then y.
    Returns (coin_pos, path) or None.
    """
    me = get_self_pos(self_info)
    if me is None or not coins:
        return None
    # compute distance maps (dist[x, y])
    self_dist = bfs_distance(field, me, explosions)
    opps = get_others_positions(others)
    opp_dists = [bfs_distance(field, op, explosions) for op in opps]  # list of dist arrays
    candidates = []
    for c in coins:
        cx, cy = int(c[0]), int(c[1])
        if not in_bounds(field, cx, cy):
            continue
        # --- CORRECT INDEXING: dist is [x, y] ---
        sd = self_dist[cx, cy]            # self ETA to coin (float or inf)
        if not np.isfinite(sd):
            continue
        # minimal opponent ETA (to same coin) — default INF if no opponents
        if opp_dists:
            od_min = min((od[cx, cy] for od in opp_dists))
        else:
            od_min = np.inf
        # lead = opponent ETA - self ETA  (positive -> we arrive earlier)
        lead = od_min - sd
        # require we have the lead_margin advantage
        if lead <= lead_margin:
            continue
        # Validate a concrete path exists (safe path considering explosions)
        path = bfs_shortest_path(field, me, (cx, cy), explosions)
        if path is None:
            continue
        # store candidate: coin, self ETA, opp ETA, lead, path
        candidates.append(( (cx, cy), float(sd), float(od_min), float(lead), path ))
    if candidates:
        # sort: prefer smallest sd, then largest lead, then smaller x, then y
        candidates.sort(key=lambda t: (t[1], -t[3], t[0][0], t[0][1]))
        chosen = candidates[0]
        return chosen[0], chosen[4]
    # fallback handled by caller
    return None

OPPOSITE = {"UP":"DOWN", "DOWN":"UP", "LEFT":"RIGHT", "RIGHT":"LEFT"}

def _next_action_avoid_backtrack(me, path, field, explosions, last_pos=None):
    """Choose next action from path but avoid stepping to last_pos if possible."""
    if not path or len(path) < 2:
        return "WAIT"
    next_cell = path[1]   # (x,y)
    # direct action
    act = next_action_toward(path[0], path[1])
    if last_pos is None or next_cell != last_pos:
        return act
    # next_cell is the cell we just came from -> try alternatives:
    # try all neighbors (prefer those that keep us closer to goal)
    goal = path[-1]
    best_alt = None
    for dx, dy in DELTAS:
        nx, ny = me[0] + dx, me[1] + dy
        if not in_bounds(field, nx, ny):
            continue
        if not is_free(field, nx, ny):
            continue
        if (nx, ny) == last_pos:
            continue
        # check reachable from that neighbor to goal
        p = bfs_shortest_path(field, (nx, ny), goal, explosions)
        if p is not None:
            # prefer neighbor that yields shortest remaining distance
            rem_len = len(p) - 1
            if best_alt is None or rem_len < best_alt[0]:
                best_alt = (rem_len, (nx, ny))
    if best_alt is not None:
        return next_action_toward(me, best_alt[1])
    # no safe alternative -> WAIT (better than oscillating)
    return "WAIT"

def coin_collection_policy(field: np.ndarray, self_info, coins: List[Tuple[int,int]],
                           explosions: np.ndarray, others: List, lead_margin: int = 1) -> Dict:
    """
    Returns dict with coin action. Expects optional last_pos in self_info for anti-oscillation.
    """
    last_pos = None
    # if self_info stores last pos (you can adapt the key name)
    if isinstance(self_info, dict):
        last_pos = self_info.get("last_pos", None)
    choice = choose_coin_opponent_aware(field, self_info, coins, explosions, others, lead_margin)
    if choice is None:
        # fallback nearest safe coin
        nearest = nearest_safe_coin(field, self_info, coins, explosions)
        if nearest is None:
            return {"coin_available":"no", "coin_action":"WAIT", "coin_reason":"No coins available to collect."}
        coin, _, path = nearest
    else:
        coin, path = choice
    if not path or len(path) < 2:
        return {"coin_available":"yes", "coin_action":"WAIT", "coin_reason":"Coins unreachable or not safe."}
    me = get_self_pos(self_info)
    action = _next_action_avoid_backtrack(me, path, field, explosions, last_pos=last_pos)
    return {"coin_available":"yes", "coin_action": action, "coin_reason": "Coins are reachable."}

## Plant Bomb Functions
def blast_cells_from(pos: Tuple[int,int], field: np.ndarray, blast_strength: int) -> np.ndarray:
    """
    Returns a boolean mask with True where a bomb at `pos` would hit.
    Stops at rigid wall (-1). Blast reaches crates (1) but won’t go past them.
    """
    w, h = field.shape
    mask = np.zeros((w, h), dtype=bool)
    x0, y0 = pos
    if not in_bounds(field, x0, y0):
        return mask
    mask[x0, y0] = True
    for dx, dy in DELTAS:
        for step in range(1, blast_strength + 1):
            nx, ny = x0 + dx * step, y0 + dy * step
            if not in_bounds(field, nx, ny):
                break
            if field[nx, ny] == -1:  # wall
                break
            mask[nx, ny] = True
            if field[nx, ny] == 1:  # crate stops blast
                break
    return mask

def bfs_distance_avoid(field: np.ndarray, start: Tuple[int,int], 
                       avoid_mask: Optional[np.ndarray]=None, 
                       max_depth: Optional[int]=None) -> np.ndarray:
    """
    BFS distances from start avoiding cells where avoid_mask is True.
    Returns float array (np.inf for unreachable).
    """
    w, h = field.shape
    dist = np.full((w, h), np.inf, dtype=np.float32)
    sx, sy = start
    if not in_bounds(field, sx, sy):
        return dist
    if not is_free(field, sx, sy):
        return dist
    # Ensure avoid_mask is boolean
    if avoid_mask is not None:
        avoid_mask = (avoid_mask.astype(bool))
    else:
        avoid_mask = np.zeros_like(field, dtype=bool)
    dq = deque()
    dist[sy, sx] = 0.0
    dq.append((sx, sy))
    while dq:
        x, y = dq.popleft()
        d = dist[x, y]
        # stop exploring *after* reaching max_depth
        if max_depth is not None and d > max_depth:
            continue
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if not in_bounds(field, nx, ny):
                continue
            if dist[nx, ny] != np.inf:
                continue
            if not is_free(field, nx, ny):
                continue
            if avoid_mask[nx, ny]:
                continue
            dist[nx, ny] = d + 1.0
            dq.append((nx, ny))
    return dist

def should_plant_bomb(game_state: Dict,
                      field: np.ndarray,
                      self_info: Tuple,
                      bombs: List,
                      others: List,
                      blast_strength: int = 3,
                      explosion_timer: int = 3,
                      allow_suicide: bool = False) -> Dict:
    """
    Decide if planting a bomb now is sensible.
    Adds condition: bomb must be adjacent to at least one crate to be worth planting.
    """
    if bombs:
        current_status = "Bomb is present, hide from bomb!"
    else:
        current_status = "No bomb detected."
    if field is None or self_info is None:
        return {"plant": "false", "reason": "missing field or self position", "targets": None, "escape_distance": None}
    sx, sy = get_self_pos(self_info)
    # 1) Compute blast footprint
    blast_mask = blast_cells_from((sx, sy), field, blast_strength)
    # 2) Detect targets (opponents / crates)
    opponents_hit = []
    crates_hit = []
    for opp in others:
        ox, oy = opp[-1]
        if in_bounds(field, ox, oy) and blast_mask[ox, oy]:
            opponents_hit.append((ox, oy))
    xs, ys = np.where(blast_mask)
    for x, y in zip(xs, ys):
        if field[x, y] == 1:
            crates_hit.append((x, y))
    # 3) Compute escape route avoiding blast
    avoid_mask = blast_mask.copy()
    dist = bfs_distance_avoid(field, (sx, sy), avoid_mask=None, max_depth=explosion_timer)
    # 4) Find any safe cell reachable within explosion_timer
    safe_cells = np.where((dist <= explosion_timer) & (~avoid_mask))
    safe_positions = list(zip(safe_cells[1], safe_cells[0]))  # (x, y)
    safe_distance = float(np.min(dist[~avoid_mask])) if len(safe_positions) > 0 else None
    # 5) Check adjacency to crate (new logic)
    adjacent_to_crate = False
    for dx, dy in DELTAS:
        nx, ny = sx + dx, sy + dy
        if in_bounds(field, nx, ny) and field[nx, ny] == 1:
            adjacent_to_crate = True
            break
    # 6) Decision logic
    reasons = []
    if opponents_hit:
        reasons.append(f"opponent_in_blast: {len(opponents_hit)}")
    if crates_hit:
        reasons.append(f"crate_in_blast: {len(crates_hit)}")
    if not opponents_hit and not crates_hit:
        return {"plant": "false", "reason": "no opponent or crate in blast footprint", 
                "current_status": current_status,
                "targets": {"opponents": opponents_hit, "crates": crates_hit}, 
                "escape_distance": safe_distance}
    if not adjacent_to_crate and not opponents_hit:
        return {"plant": "false", "reason": "not adjacent to crate", 
                "current_status": current_status,
                "targets": {"opponents": opponents_hit, "crates": crates_hit}, 
                "escape_distance": safe_distance}
    if safe_distance is None:
        if opponents_hit and allow_suicide:
            return {"plant": "true", "reason": "no escape but opponents will be hit (suicide allowed)",
                    "current_status": current_status,
                    "targets": {"opponents": opponents_hit, "crates": crates_hit},
                    "escape_distance": None}
        return {"plant": "false", "reason": "no safe escape within explosion timer",
                "current_status": current_status,
                "targets": {"opponents": opponents_hit, "crates": crates_hit},
                "escape_distance": None}
    # safe + valid target
    return {"plant": "true", "reason": "safe escape available and target in blast",
            "current_status": current_status,
            "targets": {"opponents": opponents_hit, "crates": crates_hit},
            "escape_distance": safe_distance}


## Nearest Crate
def nearest_crate_action(field: np.ndarray, self_info, explosions: np.ndarray) -> Dict:
    """
    Find the nearest crate (field == 1) that can be reached safely.
    Returns a dict:
    {
        "crate_available": "yes" | "no",
        "crate_action": str,          # e.g., "UP", "RIGHT", "DOWN", "LEFT", or "WAIT"
        "crate_pos": (x, y) | None,
        "crate_distance": float | None,
        "crate_reason": str
    }
    Tie-breaking: smaller x first, then smaller y.
    """
    me = get_self_pos(self_info)
    if me is None:
        return {"crate_available": "no", "crate_action": "WAIT",
                "crate_pos": None, "crate_distance": None,
                "crate_reason": "Self position unavailable."}
    # Find all crate coordinates
    crate_coords = list(zip(*np.where(field == 1)))
    if not crate_coords:
        return {"crate_available": "no", "crate_action": "WAIT",
                "crate_pos": None, "crate_distance": None,
                "crate_reason": "No crates available on the map."}
    # Compute BFS distances from agent position
    dist = bfs_distance(field, me, explosions)
    # Find reachable crates
    reachable_crates = []
    for cx, cy in crate_coords:
        path_to_adj = bfs_shortest_path_crate(field, me, (cx, cy), explosions)
        if path_to_adj is None:
            continue
        dist = len(path_to_adj) - 1
        reachable_crates.append((cx, cy, path_to_adj, dist))
    if not reachable_crates:
        return {"crate_available": "no", "crate_action": "WAIT",
                "crate_pos": None, "crate_distance": None,
                "crate_reason": "No reachable crates found."}
    # sort by distance then x then y
    reachable_crates.sort(key=lambda t: (t[3], t[0], t[1]))
    tx, ty, path, d = reachable_crates[0]

    if not path or len(path) < 2:
        # Either no path found or already adjacent to crate
        return {
            "crate_available": "yes",
            "crate_action": "WAIT",
            "crate_pos": (int(tx), int(ty)),
            "crate_distance": float(d),
            "crate_reason": (
                "Already adjacent to crate."
                if path and len(path) == 1
                else "Crate reachable but no path found."
            ),
        }
    action = next_action_toward(path[0], path[1])
    return {
        "crate_available": "yes",
        "crate_action": action,
        "crate_pos": (int(tx), int(ty)),
        "crate_distance": float(d),
        "crate_reason": "Nearest crate identified and reachable."
    }
    
## Extras
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

