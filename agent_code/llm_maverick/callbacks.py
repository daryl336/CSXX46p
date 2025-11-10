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
# BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000/agentic-predict"
BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000"
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
        self.logger.info(f"Lade Model '{PARAMETERS}'.")
        filename = os.path.join("network_parameters", f'{PARAMETERS}.pt')
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()

    initialize_rule_based(self)

    self.bomb_buffer = 0

    # LLM-related tracking
    self.movement_history = []
    self.bomb_history = deque([], 5)
    self.opponent_histories = {}  # Track opponent positions/actions
    self.llm_available = True  # Track if LLM server is responding

    # Metrics tracking for evaluation
    from metrics.metrics_tracker import MetricsTracker
    self.name = "LLM Maverick"
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.episode_active = False
    self.current_step = 0


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    # METRICS TRACKING
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

    # STEP 1: Get Maverick's Q-values and recommendations
    features = state_to_features(self, game_state)
    Q = self.network(features)

    if self.train: # Exploration vs exploitation during training
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps: # choose random action
            if eps > 0.1:
                if np.random.randint(10) == 0:    # old: 10 / 100 now: 3/4
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

    # STEP 2: Compute Maverick's top 3 recommendations
    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    Q_values = Q.detach().squeeze().numpy()

    # Get top 3 actions by Q-value
    top_3_indices = np.argsort(Q_values)[-3:][::-1]  # Descending order
    top_3_actions = [
        {"action": ACTIONS[i], "q_value": float(Q_values[i]), "probability": float(action_prob[i])}
        for i in top_3_indices
    ]

    maverick_best_action = ACTIONS[top_3_indices[0]]

    # STEP 3: Extract LLM helper features from game state
    field = game_state.get('field', np.zeros((17, 17)))
    self_info = game_state.get('self')
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros((17, 17)))
    lead_margin = 1

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

    # STEP 4: If LLM is available, send combined payload for final decision
    final_action = maverick_best_action  # Fallback to Maverick

    if self.llm_available and not self.train:  # Only use LLM during inference
        try:
            payload = {
                # NEW: Maverick's recommendations
                "maverick_top_actions": json.dumps(top_3_actions),
                "maverick_features": json.dumps(features.squeeze().tolist()),
                "maverick_best_action": maverick_best_action,

                # LLM helper features
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

            final_action = results.get("action", maverick_best_action)
            reasoning = results.get("reasoning", "No reasoning provided")

            self.logger.info(f"LLM Decision: {final_action} (Maverick suggested: {maverick_best_action})")
            self.logger.info(f"LLM Reasoning: {reasoning}")

            # Track movement history
            self.movement_history.append({
                "action": final_action,
                "reasoning": reasoning,
                "maverick_suggestion": maverick_best_action
            })

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            self.logger.warning(f"LLM server unavailable: {e}. Using Maverick fallback.")
            self.llm_available = False
            final_action = maverick_best_action
        except Exception as e:
            self.logger.error(f"LLM error: {e}. Using Maverick fallback.")
            final_action = maverick_best_action
    else:
        self.logger.info(f"Waehle Aktion {maverick_best_action} nach dem Hardmax der Q-Funktion")
        final_action = maverick_best_action

    # STEP 5: Update opponent tracking
    for opponent_info in others:
        opp_name = opponent_info[0] if isinstance(opponent_info, (tuple, list)) else "Unknown"
        opp_pos = opponent_info[3] if isinstance(opponent_info, (tuple, list)) and len(opponent_info) >= 4 else None

        if opp_name not in self.opponent_histories:
            self.opponent_histories[opp_name] = deque(maxlen=5)

        if opp_pos:
            self.opponent_histories[opp_name].append(opp_pos)

    return final_action


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
        metadata={"mode": "llm_maverick"}
    )

    self.metrics_tracker.save()
    self.episode_active = False


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Called at the end of each round during play mode."""
    if hasattr(self, 'episode_active') and self.episode_active:
        _end_episode_from_act(self, last_game_state)