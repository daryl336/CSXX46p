from __future__ import annotations
import os, json, logging, time
from collections import deque
from typing import List, Optional, Tuple, Deque, Set, Dict, Any
import numpy as np

ACTIONS: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
LLM_ACTIONS: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BUBBLE"]
ALIAS_TO_ENGINE = {"BUBBLE": "BOMB", "STAY": "WAIT"}

BUBBLE_LIFETIME = 4
BURST_RADIUS = 3
LLM_MODEL = "gemini-2.5-flash-lite"
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "{add_your_api_key_here}")

SNAPSHOT_SAVE_EVERY = 1 
LLM_CALL_INTERVAL = 5.0
HUD_ENABLED = False

# Oscillation tracking
OSC_WINDOW = 6
OSC_UNIQUE_THRESHOLD = 2

# History sent to LLM
HISTORY_LEN = 6

# Anti-oscillation / coin pull
REVISIT_WINDOW = 8
AVOID_REVERSE = True
DEPRIORITIZE_WAIT = True
COIN_PULL = 1

# Helpers
def get_self_pos(self_info):
    if isinstance(self_info, (tuple, list)) and len(self_info) >= 4:
        pos = self_info[3]
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            return int(pos[0]), int(pos[1])
    return None

def parse_bombs_left(self_info) -> int:
    try:
        v = self_info[2]
        if isinstance(v, dict):
            return int(v.get("bombs_left", 0))
        return int(bool(v)) if isinstance(v, bool) else int(v)
    except Exception:
        return 0

def in_bounds(field, x, y):
    w, h = field.shape
    return 0 <= x < w and 0 <= y < h

def is_free(field, x, y):
    return in_bounds(field, x, y) and field[x, y] == 0

def neighbors4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

def los_blocked(field, ax, ay, bx, by):
    if ax == bx:
        step = 1 if by > ay else -1
        for yy in range(ay + step, by, step):
            if field[ax, yy] == -1:
                return True
    elif ay == by:
        step = 1 if bx > ax else -1
        for xx in range(ax + step, bx, step):
            if field[xx, ay] == -1:
                return True
    return False

def is_in_blast_zone(field, pos, bomb_pos, radius=BURST_RADIUS):
    if not pos or not bomb_pos:
        return False
    px, py = pos
    bx, by = bomb_pos
    if px == bx and abs(py - by) <= radius:
        return not los_blocked(field, px, py, bx, by)
    if py == by and abs(px - bx) <= radius:
        return not los_blocked(field, px, py, bx, by)
    return False

def is_corner(field, xy):
    w, h = field.shape
    return xy in [(1,1), (w-2,1), (1,h-2), (w-2,h-2)]

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def degree_free(field, xy: Tuple[int,int]) -> int:
    x, y = xy
    d = 0
    for nx, ny in neighbors4(x, y):
        if in_bounds(field, nx, ny) and field[nx, ny] == 0:
            d += 1
    return d

def predicted_blast_tiles_for_bomb(field, bx, by, radius=BURST_RADIUS) -> Set[Tuple[int,int]]:
    tiles = {(bx, by)}
    # up
    for dy in range(1, radius+1):
        yy = by - dy
        if not in_bounds(field, bx, yy) or field[bx, yy] == -1: break
        tiles.add((bx, yy))
    # down
    for dy in range(1, radius+1):
        yy = by + dy
        if not in_bounds(field, bx, yy) or field[bx, yy] == -1: break
        tiles.add((bx, yy))
    # left
    for dx in range(1, radius+1):
        xx = bx - dx
        if not in_bounds(field, xx, by) or field[xx, by] == -1: break
        tiles.add((xx, by))
    # right
    for dx in range(1, radius+1):
        xx = bx + dx
        if not in_bounds(field, xx, by) or field[xx, by] == -1: break
        tiles.add((xx, by))
    return tiles

def predicted_blast_tiles_for_our_bomb(field, bx, by, radius=BURST_RADIUS) -> Set[Tuple[int,int]]:
    return predicted_blast_tiles_for_bomb(field, bx, by, radius)

def in_future_cross_from(center_xy: Optional[Tuple[int,int]], pxy: Tuple[int,int], radius=BURST_RADIUS) -> bool:
    if not center_xy: return False
    bx, by = center_xy; px, py = pxy
    return (px == bx and abs(py - by) <= radius) or (py == by and abs(px - bx) <= radius)

# Bomb danger map (ALL bombs)
def compute_bomb_danger_map(field, bombs, radius=BURST_RADIUS) -> Dict[Tuple[int,int], int]:
    """
    Returns dict: tile -> min remaining ticks among bombs whose future cross hits that tile.
    """
    danger: Dict[Tuple[int,int], int] = {}
    for bpos, t in (bombs or []):
        if not isinstance(bpos, (tuple, list)) or t is None:
            continue
        bx, by = map(int, bpos)
        tiles = predicted_blast_tiles_for_bomb(field, bx, by, radius)
        tt = int(t)
        for tile in tiles:
            if tile not in danger or tt < danger[tile]:
                danger[tile] = tt
    return danger

# Opponents (utility)
def list_opponents(others) -> List[Tuple[int,int]]:
    out = []
    for o in (others or []):
        if len(o) >= 4 and isinstance(o[3], (tuple, list)):
            ox, oy = map(int, o[3])
            out.append((ox, oy))
    return out

def nearest_opponent_info(self_xy: Tuple[int,int], opps: List[Tuple[int,int]]) -> Optional[Tuple[Tuple[int,int], int]]:
    if not opps: return None
    px, py = self_xy
    best = min(opps, key=lambda q: abs(q[0]-px)+abs(q[1]-py))
    return best, abs(best[0]-px)+abs(best[1]-py)

def can_hit_target_from_here(field, src_xy: Tuple[int,int], dst_xy: Tuple[int,int], radius=BURST_RADIUS) -> bool:
    sx, sy = src_xy; dx, dy = dst_xy
    if sx == dx and abs(sy - dy) <= radius and not los_blocked(field, sx, sy, dx, dy):
        return True
    if sy == dy and abs(sx - dx) <= radius and not los_blocked(field, sx, sy, dx, dy):
        return True
    return False

def opp_line_of_sight_danger(field, dst_xy: Tuple[int,int], opps: List[Tuple[int,int]], radius=BURST_RADIUS) -> bool:
    """If we move to dst_xy, can ANY opponent place a bomb now to hit it (LOS within radius)?"""
    if not opps: return False
    x, y = dst_xy
    for ox, oy in opps:
        if (ox == x and abs(oy - y) <= radius and not los_blocked(field, ox, oy, x, y)) or \
           (oy == y and abs(ox - x) <= radius and not los_blocked(field, ox, oy, x, y)):
            return True
    return False

# Risk modeling
def time_to_blast_at(field, bombs, xy, radius=BURST_RADIUS) -> Optional[int]:
    if not bombs:
        return None
    px, py = xy
    best = None
    for bpos, t in bombs:
        if not isinstance(bpos, (tuple, list)) or t is None:
            continue
        bx, by = map(int, bpos)
        if (px == bx and abs(py - by) <= radius) or (py == by and abs(px - bx) <= radius):
            if not los_blocked(field, px, py, bx, by):
                best = int(t) if best is None else min(best, int(t))
    return best

def nearest_bomb_info(field, bombs, xy) -> Optional[Tuple[Tuple[int,int], int]]:
    if not bombs:
        return None
    px, py = xy
    cand = []
    for bpos, _t in bombs:
        if not isinstance(bpos, (tuple, list)):
            continue
        bx, by = map(int, bpos)
        cand.append(((bx, by), abs(px - bx) + abs(py - by)))
    if not cand:
        return None
    cand.sort(key=lambda z: z[1])
    return cand[0]

def per_action_risk_profile(field, self_xy, bombs, explosion_map):
    x, y = self_xy
    dirs = {"LEFT": (x - 1, y), "RIGHT": (x + 1, y), "UP": (x, y - 1), "DOWN": (x, y + 1), "WAIT": (x, y)}
    prof: Dict[str, Dict[str, Any]] = {}
    for act, (nx, ny) in dirs.items():
        ok = in_bounds(field, nx, ny) and field[nx, ny] == 0
        if isinstance(explosion_map, np.ndarray) and ok and explosion_map[nx, ny] > 0:
            ttb = 0
            blast = True
        else:
            ttb = time_to_blast_at(field, bombs, (nx, ny))
            blast = ttb is not None and ttb <= BUBBLE_LIFETIME
        prof[act] = {
            "dst": (nx, ny),
            "passable": bool(ok),
            "time_to_blast": None if not blast else int(ttb),
            "danger": bool(blast)
        }
    return prof

def choose_last_resort_move(risk_prof, prefer_not="WAIT"):
    order = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]
    def score(act):
        r = risk_prof[act]
        t = r["time_to_blast"]
        tscore = -1_000 if t is None else int(t)
        passable = 1 if r["passable"] else 0
        non_wait = 1 if act != prefer_not else 0
        return (tscore, passable, non_wait, -order.index(act))
    return max(order, key=score)

def escape_exists_after_plant(field, start_xy, explosion_map, lifetime=BUBBLE_LIFETIME, radius=BURST_RADIUS) -> Tuple[bool, Optional[Tuple[int,int]]]:
    bx, by = start_xy
    unsafe = predicted_blast_tiles_for_bomb(field, bx, by, radius)
    from collections import deque as _dq
    q = _dq()
    q.append((start_xy, 0, None))
    seen = {start_xy}
    while q:
        (x, y), t, first = q.popleft()
        if (x, y) not in unsafe and (not isinstance(explosion_map, np.ndarray) or explosion_map[x, y] == 0):
            if first is not None:
                return True, first
        if t >= lifetime:
            continue
        for nx, ny in neighbors4(x, y):
            if not in_bounds(field, nx, ny): continue
            if field[nx, ny] != 0: continue
            if isinstance(explosion_map, np.ndarray) and explosion_map[nx, ny] > 0: continue
            if (nx, ny) in seen: continue
            seen.add((nx, ny))
            q.append(((nx, ny), t+1, (nx, ny) if first is None else first))
    return False, None

def compute_context_metrics(field, self_xy, bombs, explosion_map):
    x, y = self_xy
    free_neighbors = sum(1 for nx, ny in neighbors4(x, y) if is_free(field, nx, ny))
    adjacent_crates = sum(1 for nx, ny in neighbors4(x, y) if in_bounds(field, nx, ny) and field[nx, ny] == 1)
    in_danger = any(
        isinstance(pos, (tuple, list)) and t <= 2 and is_in_blast_zone(field, (x, y), tuple(map(int, pos)))
        for pos, t in (bombs or [])
    )
    return {"free_neighbors": free_neighbors, "adjacent_crates": adjacent_crates, "in_danger": in_danger}

# Valid actions
def compute_valid_actions(field, self_info, bombs, explosion_map, others, logger=None):
    x, y = get_self_pos(self_info); bombs_left = parse_bombs_left(self_info)
    bomb_xys = [tuple(map(int, xy)) for (xy, _t) in (bombs or []) if isinstance(xy, (tuple, list))]
    others_xy = [tuple(map(int, o[3])) for o in (others or []) if len(o) >= 4]

    in_danger = any(
        isinstance(pos, (tuple, list)) and t <= 2 and is_in_blast_zone(field, (x, y), tuple(map(int, pos)))
        for pos, t in (bombs or [])
    )

    valid = []
    dirs = {"LEFT": (x - 1, y), "RIGHT": (x + 1, y), "UP": (x, y - 1), "DOWN": (x, y + 1), "WAIT": (x, y)}
    for act, (nx, ny) in dirs.items():
        if not in_bounds(field, nx, ny): continue
        if field[nx, ny] != 0: continue
        if isinstance(explosion_map, np.ndarray) and explosion_map.shape == field.shape and explosion_map[nx, ny] > 0:
            continue
        if (nx, ny) in bomb_xys or (nx, ny) in others_xy: continue
        imminent = any(t <= 1 and is_in_blast_zone(field, (nx, ny), tuple(map(int, bpos))) for bpos, t in (bombs or []))
        if imminent: continue
        valid.append(act)

    ctx = compute_context_metrics(field, (x, y), bombs, explosion_map)

    escape_ok, first_step = False, None
    if bombs_left > 0 and not ctx["in_danger"]:
        if ctx["adjacent_crates"] > 0:
            escape_ok, first_step = escape_exists_after_plant(field, (x, y), explosion_map, BUBBLE_LIFETIME, BURST_RADIUS)
    should_allow_bomb = (ctx["free_neighbors"] >= 2) or (ctx["adjacent_crates"] > 0 and escape_ok)
    if bombs_left > 0 and is_free(field, x, y) and not ctx["in_danger"] and should_allow_bomb:
        valid = ["BOMB"] + [a for a in valid if a != "BOMB"]

    if "WAIT" not in valid and is_free(field, x, y) and not ctx["in_danger"]:
        valid.append("WAIT")
    if not valid:
        valid.append("WAIT")

    if logger:
        logger.debug(
            f"[VALID] at={(x, y)} bombs_left={bombs_left} in_danger={ctx['in_danger']} "
            f"free_nbrs={ctx['free_neighbors']} adj_crates={ctx['adjacent_crates']} "
            f"escape_ok={escape_ok} first_escape_step={first_step} "
            f"others={others_xy} bombs={bomb_xys} -> valid={valid}"
        )
    return valid, ctx, escape_ok, first_step

# Anti-oscillation ordering
MOVE_TO_DELTA = {"UP": (0,-1), "RIGHT": (1,0), "DOWN": (0,1), "LEFT": (-1,0)}
REVERSE_OF = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}

def _next_xy(xy: Tuple[int,int], act: str) -> Tuple[int,int]:
    if act not in MOVE_TO_DELTA: return xy
    dx, dy = MOVE_TO_DELTA[act]
    return (xy[0]+dx, xy[1]+dy)

def _nearest_coin_dist(xy: Tuple[int,int], coins: List[Tuple[int,int]]) -> int:
    if not coins: return 1_000_000
    return min(abs(xy[0]-cx) + abs(xy[1]-cy) for (cx,cy) in coins)

def reorder_valid_actions_for_anti_osc_and_coins(valid: List[str],
                                                 me: Tuple[int,int],
                                                 recent_coords: List[Tuple[int,int]],
                                                 last_action: Optional[str],
                                                 oscillating: bool,
                                                 coins: List[Tuple[int,int]]) -> List[str]:
    if not valid or not me: return valid[:]
    recent_set = set(recent_coords[-REVISIT_WINDOW:]) if REVISIT_WINDOW > 0 else set()
    prev_xy = recent_coords[-2] if len(recent_coords) >= 2 else None
    base_coin_dist = _nearest_coin_dist(me, coins)

    indexed = list(enumerate(valid))  # preserve original order

    def score(item) -> Tuple[int,int,int,int,int,int]:
        idx, act = item
        s_rev = 1 if (AVOID_REVERSE and last_action and REVERSE_OF.get(last_action) == act) else 0
        nxt = _next_xy(me, act)
        s_revisit = 1 if (nxt in recent_set) else 0
        s_wait = 1 if (DEPRIORITIZE_WAIT and oscillating and act == "WAIT" and any(a != "WAIT" for _,a in indexed)) else 0
        s_pingpong = 1 if (prev_xy is not None and nxt == prev_xy) else 0
        coin_term = 0
        if COIN_PULL and coins and act != "BOMB":
            next_coin_dist = _nearest_coin_dist(nxt, coins)
            coin_term = max(0, next_coin_dist - base_coin_dist)
        return (s_rev, s_revisit, s_wait, s_pingpong, coin_term, idx)

    return [a for _, a in sorted(indexed, key=score)]

# TEXT snapshot (ASCII)
def _tick_glyph(t: int) -> str:
    # Visual urgency for danger tiles (ANY bomb): 0/1 most urgent
    if t <= 0: return "!"
    if t == 1: return "!"
    if t == 2: return ":"
    if t == 3: return "-"
    return "~"  # 4+

def render_text_snapshot_from_state(game_state,
                                    danger_map: Optional[Dict[Tuple[int,int], int]] = None) -> str:
    """
    Produce an ASCII grid representing the board, with overlays for:
      - walls, crates, floor
      - coins, bombs (digits = ticks)
      - self (A), opponents (E)
      - danger tiles from ANY bomb using urgency glyphs: ! (0-1), : (2), - (3), ~ (4+)
    """
    field = game_state["field"]
    coins = set(tuple(map(int, c)) for c in (game_state.get("coins") or []))
    bombs = game_state.get("bombs", [])
    self_info = game_state["self"]
    others = game_state.get("others", []) or []

    W, H = int(field.shape[0]), int(field.shape[1])
    me = get_self_pos(self_info)
    opps = list_opponents(others)
    bomb_dict: Dict[Tuple[int,int], int] = {}
    for bpos, t in bombs:
        if isinstance(bpos, (tuple, list)) and t is not None:
            bomb_dict[tuple(map(int, bpos))] = int(t)

    rows: List[str] = []

    for y in range(H):
        line = []
        for x in range(W):
            xy = (x, y)
            ch = "."
            if field[x, y] == -1:
                ch = "█"              # solid wall
            elif field[x, y] == 1:
                ch = "▓"              # crate
            else:
                ch = "."              # free

            if xy in coins:
                ch = "$"
            if xy in bomb_dict:
                # Show bomb timer digit (0..9 capped)
                bt = max(0, min(9, bomb_dict[xy]))
                ch = str(bt)

            if me and xy == me:
                if danger_map and xy in danger_map:
                    ch = "X"
                else:
                    ch = "A"

            elif xy in opps:
                ch = "E" if (not danger_map or xy not in danger_map) else "e"  # lowercase if endangered

            if danger_map and xy in danger_map and ch in {".", "$", "▓"}:
                ch = _tick_glyph(danger_map[xy])

            line.append(ch)
        rows.append("".join(line))

    legend = [
        "LEGEND: █ wall  ▓ crate  . free  $ coin  A you  E enemy  digits=bomb timer",
        "        Danger from ANY bomb: ! (0-1 ticks), : (2), - (3), ~ (4+), X = YOU in danger, e = enemy in danger"
    ]
    return "\n".join(legend + rows)

class GeminiClient:
    def __init__(self, logger):
        self.logger = logger; self.enabled = False
        if not LLM_API_KEY:
            logger.warning("[LLM] No API key, LLM disabled.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=LLM_API_KEY)
            sys_inst = (
                "You control a Bomberman agent.\n"
                'Return STRICT JSON on one line: {"move":"UP|RIGHT|DOWN|LEFT|WAIT|BUBBLE"}.\n'
                "Explosion geometry: BUBBLE explodes in a CROSS (±3 tiles) along UP/DOWN/LEFT/RIGHT after 4 ticks; DIAGONALS ARE SAFE.\n"
                "Goals (priority): (1) survive / stay safe; (2) destroy crates to open paths; (3) collect coins fast; (4) opportunistically eliminate opponents.\n"
                "Corner rule: if in a corner and adjacent to a crate, prefer BUBBLE **only if** an escape route exists.\n"
                "Policy:\n"
                "A) If ADJACENT TO CRATES and ESCAPE ROUTE exists (reach any tile outside the future CROSS within 4 ticks), PICK BUBBLE NOW.\n"
                "B) AFTER PLANTING a BUBBLE: while inside the future CROSS of your bubble and before it explodes, DO NOT WAIT. Move perpendicular off the cross, remain at least (radius+1) away until detonation.\n"
                "C) Otherwise, progress to new tiles, avoid immediate reversals, avoid WAIT unless no safe progress exists.\n"
            )
            self.model = genai.GenerativeModel(LLM_MODEL, system_instruction=sys_inst)
            self.enabled = True
            logger.info(f"[LLM] Gemini loaded: {LLM_MODEL}")
        except Exception as e:
            logger.error(f"[LLM] init failed: {e}")

    def _read_text(self, response):
        try:
            cands = getattr(response,"candidates",None) or []
            if not cands: return ""
            parts = getattr(cands[0],"content",None).parts if getattr(cands[0],"content",None) else []
            return "".join([getattr(p,"text","") for p in parts if getattr(p,"text","")])
        except Exception:
            return ""

    def ask_move(self, prompt, logger):
        if not self.enabled: return None
        try:
            import google.generativeai as genai
            resp = self.model.generate_content(
                [prompt],
                generation_config={"temperature": 0.0, "candidate_count": 1, "response_mime_type": "text/plain"},
                safety_settings=[]
            )
            text = (self._read_text(resp) or "").strip()
            if text.startswith("{") and text.endswith("}"):
                obj = json.loads(text); return str(obj.get("move","")).upper()
            t = text.upper()
            for tok in t.replace(",", " ").split():
                if tok in LLM_ACTIONS or tok in ACTIONS or tok == "STAY":
                    return tok
            return None
        except Exception as e:
            logger.error(f"[LLM] ask_move failed: {e}", exc_info=True)
            return None

def setup(self):
    self.logger.info("LLM agent setup.")
    self.llm = GeminiClient(self.logger)
    self.last_pos=None; self.last_action=None; self.turns_since_bubble=999
    self.last_bomb_xy=None
    self.coordinate_history=deque([], 16)
    self.action_history=deque([], 8)
    self.dialog_history=deque([], HISTORY_LEN)
    self.snapshot_counter=0
    self.last_llm_call_time=0.0
    self.min_llm_interval=LLM_CALL_INTERVAL

def reset_self(self):
    self.logger.info("LLM agent reset.")
    self.last_pos=None; self.last_action=None
    self.turns_since_bubble=999
    self.last_bomb_xy=None
    self.coordinate_history.clear()
    self.action_history.clear()
    self.dialog_history.clear()

# Main decision
def act(self, game_state):
    start=time.time()
    field=game_state["field"]; self_info=game_state["self"]
    others=game_state.get("others",[]); bombs=game_state.get("bombs",[])
    explosions=game_state.get("explosion_map",np.zeros_like(field))
    coins=game_state.get("coins",[]) or []
    me=get_self_pos(self_info)
    if me is None:
        self.logger.warning("[ACT] No self position; WAIT.")
        return "WAIT"

    self.turns_since_bubble+=1

    valid_actions, ctx, escape_ok, first_escape_step = compute_valid_actions(
        field,self_info,bombs,explosions,others,logger=self.logger
    )

    risk_prof = per_action_risk_profile(field, me, bombs, explosions)
    all_danger = all(risk_prof[a]["danger"] for a in ["UP","RIGHT","DOWN","LEFT","WAIT"])
    nb_info = nearest_bomb_info(field, bombs, me) or ((None, None), 999)
    nearest_bomb_xy, nearest_bomb_md = nb_info

    self.coordinate_history.append(me)
    recent=list(self.coordinate_history)[-OSC_WINDOW:]
    unique_recent = len(set(recent))
    oscillating = unique_recent <= OSC_UNIQUE_THRESHOLD and len(recent) == OSC_WINDOW
    corner = is_corner(field, me)

    in_my_future_cross = in_future_cross_from(self.last_bomb_xy, me, BURST_RADIUS) and self.turns_since_bubble <= BUBBLE_LIFETIME
    remaining_ticks = max(0, BUBBLE_LIFETIME - min(self.turns_since_bubble, BUBBLE_LIFETIME))
    dist_from_bomb = (manhattan(me, self.last_bomb_xy) if self.last_bomb_xy else None)

    # Global danger from ALL bombs
    danger_map = compute_bomb_danger_map(field, bombs, BURST_RADIUS)

    # Build text snapshot
    text_snapshot = render_text_snapshot_from_state(game_state, danger_map)
    self.snapshot_counter += 1
    if SNAPSHOT_SAVE_EVERY > 0 and (self.snapshot_counter % SNAPSHOT_SAVE_EVERY) == 0:
        self.logger.info("\n[SNAPSHOT]\n" + text_snapshot)

    # Ordered valid actions (soft)
    ordered_valid = reorder_valid_actions_for_anti_osc_and_coins(
        valid_actions, me, list(self.coordinate_history), self.last_action, oscillating, coins
    )

    # Rolling history
    hist_line = (
        f"Pos={me} Last={self.last_action} Valid={ordered_valid} "
        f"Ctx(adj={ctx['adjacent_crates']},free={ctx['free_neighbors']},danger={ctx['in_danger']},escape_ok={escape_ok}) "
        f"Corner={corner} Osc={oscillating} Coins={coins} "
        f"PostPlant(last_bomb={self.last_bomb_xy}, in_cross={in_my_future_cross}, remaining={remaining_ticks}, dist={dist_from_bomb})"
    )
    self.dialog_history.append(hist_line)
    history_text = "\n".join(self.dialog_history)

    # Per-action risk summary (compact JSON)
    risk_summary = {a: {
        "dst": list(map(int, risk_prof[a]["dst"])),
        "time_to_blast": (None if risk_prof[a]["time_to_blast"] is None else int(risk_prof[a]["time_to_blast"])),
        "danger": bool(risk_prof[a]["danger"])
    } for a in ["UP","RIGHT","DOWN","LEFT","WAIT"]}
    risk_summary_text = json.dumps(risk_summary, separators=(",", ":"))

    prompt_lines = [
        'Return JSON on one line: {"move":"UP|RIGHT|DOWN|LEFT|WAIT|BUBBLE"}',
        f"Tick={self.snapshot_counter}  LastAction={self.last_action}",
        f"ValidMoves(ordered)={ordered_valid}",
        f"Context: free_neighbors={ctx['free_neighbors']}, adjacent_crates={ctx['adjacent_crates']}, "
        f"in_danger={ctx['in_danger']}, escape_ok={escape_ok}, corner={corner}, oscillating={oscillating}",
        f"Explosion model: CROSS ±{BURST_RADIUS} (cardinals), diagonals SAFE; bubble explodes after {BUBBLE_LIFETIME} ticks.",
        f"Coins={coins}",
        f"PostPlant: last_bomb_xy={self.last_bomb_xy}, in_future_cross={in_my_future_cross}, "
        f"remaining_ticks={remaining_ticks}, dist_from_bomb={dist_from_bomb}, required_safe_dist={BURST_RADIUS+1}",
        "PER-ACTION RISK (JSON):",
        risk_summary_text,
        f"ALL_DESTINATIONS_DANGEROUS={all_danger}  NEAREST_BOMB={nearest_bomb_xy}  NEAREST_BOMB_MD={nearest_bomb_md}",
        "",
        "TEXT SNAPSHOT (ANY bomb danger shown inline):",
        text_snapshot,
        "",
        "DECISION RULES (important):",
        "1) If adjacent_crates>0 AND escape_ok==True -> choose BUBBLE now.",
        "2) If you are in any future CROSS and remaining_ticks>0 -> move OFF the centerline (perpendicular), "
        f"and keep moving until your distance from the bomb >= {BURST_RADIUS+1}. Do NOT WAIT on the cross.",
        "3) If ALL_DESTINATIONS_DANGEROUS==True:",
        "   3a) Choose the action whose destination has the LARGEST time_to_blast.",
        "   3b) If tied, prefer a move PERPENDICULAR to the line to the NEAREST_BOMB.",
        "   3c) If still tied, prefer the move that INCREASES Manhattan distance from NEAREST_BOMB.",
        "   3d) Avoid WAIT unless WAIT is strictly the safest.",
        "4) Otherwise, avoid immediate reversals and make progress to new tiles; prefer coin-reducing moves.",
        'Return only {"move":"..."} with no commentary.',
        "",
        "History (most recent last):",
        history_text
    ]
    prompt = "\n".join(prompt_lines)

    # Ask LLM
    raw_move = None
    if self.llm.enabled:
        self.logger.info("[LLM] Calling")
        raw_move = self.llm.ask_move(prompt, self.logger)
        self.logger.info(f"[LLM] Response raw='{raw_move}'")
    else:
        self.logger.warning("[LLM] Disabled or no API key; minimal safe fallback.")

    # Normalize
    move=(raw_move or "WAIT").strip().upper()
    if move in ALIAS_TO_ENGINE: move=ALIAS_TO_ENGINE[move]

    if all_danger:
        chosen_ok = (move in ACTIONS) and (move in valid_actions or move == "WAIT")
        if (move == "WAIT") or (not chosen_ok):
            alt = choose_last_resort_move(risk_prof, prefer_not="WAIT")
            self.logger.debug(f"[ESCAPE] All red; overriding '{move}' -> '{alt}' (max time_to_blast)")
            move = alt

    if move not in valid_actions:
        alt = next((a for a in ordered_valid if a in valid_actions and a!="WAIT"), valid_actions[0])
        self.logger.debug(f"[GUARD] Invalid/empty LLM move -> picked '{alt}'")
        move = alt

    final_move=move
    self.last_pos=me; self.last_action=final_move
    if final_move=="BOMB":
        self.turns_since_bubble=0
        self.last_bomb_xy = me
        self.logger.info("💣 BUBBLE PLACED (LLM decision)")

    self.action_history.append((final_move, int((time.time()-start)*1000)))

    try:
        elapsed=time.time()-start
        if elapsed<LLM_CALL_INTERVAL:
            sleep=LLM_CALL_INTERVAL-elapsed
            self.logger.info(f"⏱ Sleeping {sleep:.2f}s (pad interval)")
            time.sleep(sleep)
    except Exception:
        pass

    self.logger.info(f"✅ Turn complete: {final_move}")
    return final_move
