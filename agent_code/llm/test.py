import os
import pickle
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

ACTIONS: List[str] = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

with open("agent_code/rule_based_agent/game_state.pkl", 'rb') as f:
    game_state = pickle.load(f)
game_state.keys()
field = game_state.get('field', np.zeros((17, 17)))
self_info = game_state.get('self')
others = game_state.get('others', [])
coins = game_state.get('coins', [])
bombs = game_state.get('bombs', [])
explosions = game_state.get('explosion_map', np.zeros((17, 17)))
lead_margin=1

from agent_code.llm.helper import \
    coin_collection_policy, \
    choose_coin_opponent_aware, \
    nearest_safe_coin, \
    get_self_pos, \
    bfs_distance, \
    get_others_positions, \
    in_bounds, \
    bfs_shortest_path, \
    should_plant_bomb, \
    check_bomb_radius, \
    check_bomb_radius_and_escape, \
    check_valid_movement

coin_collection_policy(field, self_info, coins, explosions, others, lead_margin) 
choice = choose_coin_opponent_aware(field, self_info, coins, explosions, others, lead_margin)
nearest = nearest_safe_coin(field, self_info, coins, explosions)

me = get_self_pos(self_info)
self_dist = bfs_distance(field, me, explosions)
opps = get_others_positions(others)
opp_dists = [bfs_distance(field, op, explosions) for op in opps]
c = coins[0]
c = coins[1]
if not in_bounds(field, c[0], c[1]):
    print("not in bound.")

sd = self_dist[c[0], c[1]]
od_min = min((od[c[0], c[1]] for od in opp_dists), default=np.inf)
od_min = 0
lead = sd - od_min
candidates = []
if lead > lead_margin:
    path = bfs_shortest_path(field, me, c, explosions)
    if path is not None:
        candidates.append((c, int(sd), int(od_min), path))
if candidates:
    # Choose coin with minimal self ETA (then maximal lead)
    candidates.sort(key=lambda t: (t[1], -(t[2]-t[1])))
    best = candidates[0]
    print(best[0], best[3])

should_plant_bomb(game_state,field,self_info,others)

check_bomb_radius(field, self_info, bombs, explosions)
check_bomb_radius_and_escape(field, self_info, bombs, explosions)
valid_movements = check_valid_movement(field, self_info)