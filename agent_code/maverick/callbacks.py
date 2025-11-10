import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from random import shuffle

from .Model import Maverick
from .ManagerFeatures import state_to_features
from .ManagerRuleBased import act_rulebased, initialize_rule_based

import events as e

# PARAMETERS = 'last_save' #select parameter_set stored in network_parameters/
PARAMETERS = 'final_parameters' #select parameter_set stored in network_parameters/

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']


def setup(self):
    """
    This is called once when loading each agent.
    Preperation such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = Maverick()

    # Always load the trained model for evaluation
    # (even if --train flag is used for metrics tracking)
    agent_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(agent_dir, "network_parameters", f'{PARAMETERS}.pt')

    if os.path.exists(model_path):
        self.logger.info(f"Loading trained model '{PARAMETERS}' from {model_path}")
        self.network.load_state_dict(torch.load(model_path))
        self.network.eval()
    else:
        if self.train:
            self.logger.info("Training a new model from scratch (no saved model found).")
        else:
            self.logger.warning(f"Model file not found: {model_path}. Using untrained network!")

    initialize_rule_based(self)

    self.bomb_buffer = 0

    # Metrics tracking for evaluation
    from metrics.metrics_tracker import MetricsTracker
    self.name = "Maverick"
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

    features = state_to_features(self, game_state)
    Q = self.network(features)

    # Only use epsilon-greedy exploration if actively training (not just evaluating with metrics)
    # Check if EVALUATE_WITH_METRICS flag is set
    from .train import EVALUATE_WITH_METRICS

    if self.train and hasattr(self, 'epsilon_arr') and hasattr(self, 'episode_counter') and not EVALUATE_WITH_METRICS:
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

    action_prob	= np.array(torch.softmax(Q,dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    # best_action = act_rulebased(self)
    self.logger.info(f"Waehle Aktion {best_action} nach dem Hardmax der Q-Funktion")

    return best_action


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