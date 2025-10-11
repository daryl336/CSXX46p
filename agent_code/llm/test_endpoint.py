import os
import json
import pickle
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import requests

def load_data(round, step):
    with open(f"agent_code/llm/data_{round}_{step}.pkl", 'rb') as f:
        data = pickle.load(f)
    game_state = data.get("game_state")
    payload = data.get("payload")
    results = data.get("results")
    field = game_state.get('field', np.zeros((17, 17)))
    self_info = game_state.get('self')
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros((17, 17)))
    lead_margin=1
    return game_state, payload, results, field, self_info, others, coins, bombs, explosions, lead_margin

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
    check_bomb_radius_and_escape, \
    check_valid_movement, \
    find_escape_direction, \
    blast_cells_from, \
    bfs_distance_avoid, \
    nearest_crate_action, \
    bfs_shortest_path_crate


def get_llm_action(round, step):
    game_state, payload, results, field, self_info, others, coins, bombs, explosions, lead_margin = load_data(round, step)
    valid_movement = check_valid_movement(field, self_info)
    nearest_crate = nearest_crate_action(field, self_info, explosions)
    bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
    plant_bomb_full_data = should_plant_bomb(game_state,field,self_info,others)
    coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin) 
    plant_bomb_data = {
        "plant":plant_bomb_full_data.get("plant"), 
        "reason":plant_bomb_full_data.get("reason"),
    }
    BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000"
    payload = {
        "valid_movement": json.dumps(valid_movement),
        "nearest_crate": json.dumps(nearest_crate),
        "check_bomb_radius": json.dumps(bomb_radius_data),
        "plant_bomb_available": json.dumps(plant_bomb_data),
        "coins_collection_policy": json.dumps(coins_collection_data),
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", BOMBERMAN_AGENT_ENDPOINT, headers=headers, json=payload)
    results = response.json()
    reasoning = results.get("reasoning")
    action = results.get("action")
    # print("====================================")
    # print(results)
    # print("====================================")
    # print(f"Action Taken: {action}")
    return action

round=1
for step in range(1,14):
    action = get_llm_action(round, step)
    print(f"Step : {step}, Action Taken: {action}")

step=1
game_state, payload, results, field, self_info, others, coins, bombs, explosions, lead_margin = load_data(round, step)
valid_movement = check_valid_movement(field, self_info, bombs)
bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
plant_bomb_full_data = should_plant_bomb(game_state, field, self_info, bombs, others)
coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin) 
results

step=5
game_state, payload, results, field, self_info, others, coins, bombs, explosions, lead_margin = load_data(round, step)
valid_movement = check_valid_movement(field, self_info, bombs)
bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
plant_bomb_full_data = should_plant_bomb(game_state, field, self_info, bombs, others)
coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin) 
results
