"""
Training interface for Bomberman DQN.
Forwards to callbacks.py implementations.
"""

from .callbacks import (
    setup,
    setup_training,
    act,
    game_events_occurred,
    end_of_round,
    reward_from_events,
)

__all__ = [
    'setup',
    'setup_training', 
    'act',
    'game_events_occurred',
    'end_of_round',
    'reward_from_events',
]