import os
import pickle
import random
import requests
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from random import shuffle

from .Model import Maverick
from .ManagerFeatures import state_to_features
from .ManagerRuleBased import act_rulebased, initialize_rule_based

# Import LLM helper functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'llm_predict'))
from agent_code.llm_predict.helper import (
    check_valid_movement,
    check_bomb_radius_and_escape,
    should_plant_bomb,
    coin_collection_policy,
    get_self_pos,
    nearest_crate_action,
    analyze_all_opponents
)

import events as e

# PARAMETERS = 'last_save' #select parameter_set stored in network_parameters/
PARAMETERS = 'final_parameters' #select parameter_set stored in network_parameters/

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']

# LLM endpoint for final decision validation
BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000/agentic-predict"
LLM_TIMEOUT = 60  # 400ms timeout for LLM server


def setup(self):
    """
    This is called once when loading each agent.
    Preperation such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = Maverick()

    if self.train:
        self.logger.info("Trainiere ein neues Model.")

    else:
        # Try to load existing weights, but handle case where they don't exist or are incompatible
        filename = os.path.join("network_parameters", f'{PARAMETERS}.pt')
        if os.path.exists(filename):
            try:
                self.logger.info(f"Lade Model '{PARAMETERS}'.")
                self.network.load_state_dict(torch.load(filename))
                self.network.eval()
            except RuntimeError as e:
                self.logger.warning(f"Could not load weights (incompatible architecture): {e}")
                self.logger.info("Starting with fresh network - you need to train from scratch")
        else:
            self.logger.info(f"No weights found at {filename}. Starting with fresh network.")

    initialize_rule_based(self)

    self.bomb_buffer = 0

    # LLM-related tracking
    self.movement_history = []
    self.bomb_history = deque([], 5)
    self.opponent_histories = {}  # Track opponent positions/actions
    self.llm_available = True  # Track if LLM server is responding
    

def act(self, game_state: dict) -> str:
    """
    OPTION 2: LLM action is included as a feature in the neural network

    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # STEP 1: Get LLM recommendation FIRST (to use as feature)
    llm_action = None

    if self.llm_available:  # Use LLM during both training and inference
        field = game_state.get('field', np.zeros((17, 17)))
        self_info = game_state.get('self')
        others = game_state.get('others', [])
        coins = game_state.get('coins', [])
        bombs = game_state.get('bombs', [])
        explosions = game_state.get('explosion_map', np.zeros((17, 17)))
        lead_margin = 1

        try:
            # Extract LLM helper features
            valid_movement = check_valid_movement(field, self_info, bombs)
            nearest_crate = nearest_crate_action(field, self_info, explosions)
            bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
            plant_bomb_full_data = should_plant_bomb(game_state, field, self_info, bombs, others)
            coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin)
            plant_bomb_data = {
                "plant": plant_bomb_full_data.get("plant"),
                "reason": plant_bomb_full_data.get("reason"),
                "current_status": plant_bomb_full_data.get("current_status"),
            }

            # Analyze opponents
            opponents_analysis = analyze_all_opponents(game_state, self.opponent_histories)

            payload = {
                "valid_movement": json.dumps(valid_movement),
                "nearest_crate": json.dumps(nearest_crate),
                "check_bomb_radius": json.dumps(bomb_radius_data),
                "plant_bomb_available": json.dumps(plant_bomb_data),
                "coins_collection_policy": json.dumps(coins_collection_data),
                "movement_history": json.dumps(self.movement_history[-5:]),
                "opponents": json.dumps(opponents_analysis),
            }

            headers = {'Content-Type': 'application/json'}
            response = requests.post(BOMBERMAN_AGENT_ENDPOINT,
                                    headers=headers,
                                    json=payload,
                                    timeout=LLM_TIMEOUT)
            results = response.json()

            llm_action = results.get("action", None)
            reasoning = results.get("reasoning", "No reasoning provided")

            self.logger.info(f"LLM Recommendation: {llm_action}")
            self.logger.info(f"LLM Reasoning: {reasoning}")

            # Cache LLM action for this step
            self.llm_cache = {
                'step': game_state.get('step', -1),
                'action': llm_action,
                'reasoning': reasoning
            }

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            self.logger.warning(f"LLM server unavailable: {e}. Using Maverick without LLM feature.")
            self.llm_available = False
            llm_action = None
        except Exception as e:
            self.logger.error(f"LLM error: {e}. Using Maverick without LLM feature.")
            llm_action = None

    # STEP 2: Extract features with LLM action included
    features = state_to_features(self, game_state, llm_action=llm_action)
    Q = self.network(features)

    # STEP 3: Training exploration
    if self.train:
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps:
            if eps > 0.1:
                if np.random.randint(10) == 0:
                    action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
                    self.logger.info(f"Waehle Aktion {action} komplett zufaellig")
                    return action
                else:
                    action = act_rulebased(self)
                    self.logger.info(f"Waehle Aktion {action} nach dem rule based agent.")
                    return action
            else:
                action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                self.logger.info(f"Waehle Aktion {action} komplett zufaellig")
                return action

    # STEP 4: Inference - use network output (which already considered LLM action as feature)
    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]

    if llm_action:
        self.logger.info(f"Waehle Aktion {best_action} (LLM suggested: {llm_action})")
    else:
        self.logger.info(f"Waehle Aktion {best_action} nach dem Hardmax der Q-Funktion")

    # STEP 5: Update tracking (for both training and inference)
    for opponent_info in game_state.get('others', []):
        opp_name = opponent_info[0] if isinstance(opponent_info, (tuple, list)) else "Unknown"
        opp_pos = opponent_info[3] if isinstance(opponent_info, (tuple, list)) and len(opponent_info) >= 4 else None

        if opp_name not in self.opponent_histories:
            self.opponent_histories[opp_name] = deque(maxlen=5)

        if opp_pos:
            self.opponent_histories[opp_name].append(opp_pos)

    self.movement_history.append({
        "action": best_action,
        "llm_suggestion": llm_action,
        "reasoning": self.llm_cache.get('reasoning', '') if hasattr(self, 'llm_cache') else ''
    })

    return best_action