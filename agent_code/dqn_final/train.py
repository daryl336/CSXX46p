import torch
from pathlib import Path
from .callbacks import _latest_model_path

from .callbacks import (
    setup as _setup,
    setup_training as _setup_training,
    act as _act,
    game_events_occurred as _game_events_occurred,
    end_of_round as _end_of_round,
    reward_from_events as _reward_from_events,
)

def setup(self):
    return _setup(self)

def setup_training(self):
    return _setup_training(self)

def act(self, game_state):
    return _act(self, game_state)

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    return _game_events_occurred(self, old_game_state, self_action, new_game_state, events)

def end_of_round(self, last_game_state, last_action, events):
    return _end_of_round(self, last_game_state, last_action, events)

def reward_from_events(self, events):
    return _reward_from_events(self, events)


def load_for_eval():
    from .callbacks import ConvQNet, DEVICE
    in_ch, n_actions = 7, 6
    q = ConvQNet(in_ch, n_actions).to(DEVICE)
    path = _latest_model_path()
    if path is None:
        raise FileNotFoundError("No saved dqn_final model under models/")
    state = torch.load(path, map_location=DEVICE)
    q.load_state_dict(state["q"])
    q.eval()
    return q
