from collections import deque
import numpy as np
import pickle
import requests
import json
from .helper import check_valid_movement, check_bomb_radius_and_escape, should_plant_bomb, coin_collection_policy, get_self_pos

BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000"

def setup(self):
    self.movement_history = []
    self.bomb = []
    self.bomb_history = deque([], 5)

def act(self, game_state):
    round = game_state.get('round')
    step = game_state.get('step')
    field = game_state.get('field', np.zeros((17, 17)))
    self_info = game_state.get('self')
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros((17, 17)))
    lead_margin=1
    print(f"Round : {round}, Step : {step}")
    valid_movement = check_valid_movement(field, self_info)
    bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
    plant_bomb_full_data = should_plant_bomb(game_state,field,self_info,others)
    coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin) 
    plant_bomb_data = {"plant":plant_bomb_full_data.get("plant"), "reason":plant_bomb_full_data.get("reason")}
    payload = {
        "valid_movement": json.dumps(valid_movement),
        "check_bomb_radius": json.dumps(bomb_radius_data),
        "plant_bomb_available": json.dumps(plant_bomb_data),
        "coins_collection_policy": json.dumps(coins_collection_data),
        "movement_history": json.dumps(self.movement_history[-5:]),
    }
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", BOMBERMAN_AGENT_ENDPOINT, headers=headers, json=payload)
    results = response.json()
    print("====================================")
    print(results)
    print("====================================")
    reasoning = results.get("reasoning")
    action = results.get("action")
    print(f"Action Taken: {action}")
    data = {
        "game_state":game_state,
        "payload":payload,
        "results":results,
    }
    with open(f"data_{round}_{step}.pkl", 'wb') as f:
        pickle.dump(data, f)
    self.movement_history.append(results)
    if action == "BOMB":
        current_position = get_self_pos(self_info)
        self.bomb = [current_position, 3]
    else:
        if self.bomb:
            countdown = self.bomb[1]
            if countdown == 0:
                print("BOMB exploded")
                self.bomb = []
            else:
                self.bomb[1] = countdown - 1
                print(self.bomb)
    return action
    