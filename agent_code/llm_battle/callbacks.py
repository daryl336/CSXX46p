from collections import deque
import numpy as np
import pickle
import requests
import json
from .helper import (check_valid_movement, check_bomb_radius_and_escape, should_plant_bomb,
                     coin_collection_policy, get_self_pos, nearest_crate_action, analyze_all_opponents)

BOMBERMAN_AGENT_ENDPOINT = "http://0.0.0.0:6000/battle-agent"

def setup(self):
    self.movement_history = []
    self.bomb = []
    self.bomb_history = deque([], 5)
    self.opponent_histories = {}  # Track opponent positions/actions

    # Metrics tracking for evaluation
    from metrics.metrics_tracker import MetricsTracker
    self.name = "LLM Battle"
    self.metrics_tracker = MetricsTracker(agent_name=self.name, save_dir="evaluation_metrics")
    self.episode_active = False
    self.current_step = 0

def act(self, game_state):
    with open("game_state.pkl", 'wb') as f:
        pickle.dump(game_state, f)
    round = game_state.get('round')
    step = game_state.get('step')

    # METRICS TRACKING
    if hasattr(self, 'metrics_tracker'):
        # Start new episode at step 1
        if step == 1:
            if self.episode_active and hasattr(self, 'last_game_state'):
                _end_episode_from_act(self, self.last_game_state)

            opponent_names = [o[0] for o in game_state.get('others', []) if o]
            self.metrics_tracker.start_episode(
                episode_id=round,
                opponent_types=opponent_names,
                map_name="default",
                scenario="evaluation"
            )
            self.episode_active = True
            self.current_step = 0

        if self.episode_active:
            self.current_step += 1

        self.last_game_state = game_state

    field = game_state.get('field', np.zeros((17, 17)))
    self_info = game_state.get('self')
    others = game_state.get('others', [])
    coins = game_state.get('coins', [])
    bombs = game_state.get('bombs', [])
    explosions = game_state.get('explosion_map', np.zeros((17, 17)))
    lead_margin=1
    print(f"Round : {round}, Step : {step}")
    valid_movement = check_valid_movement(field, self_info, bombs)
    nearest_crate = nearest_crate_action(field, self_info, explosions)
    bomb_radius_data = check_bomb_radius_and_escape(field, self_info, bombs, explosions)
    plant_bomb_full_data = should_plant_bomb(game_state, field, self_info, bombs, others)
    coins_collection_data = coin_collection_policy(field, self_info, coins, explosions, others, lead_margin)
    plant_bomb_data = {
        "plant":plant_bomb_full_data.get("plant"),
        "reason":plant_bomb_full_data.get("reason"),
        "current_status":plant_bomb_full_data.get("current_status"),
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

    # Update opponent histories (track last 5 positions)
    for opponent_info in others:
        opp_name = opponent_info[0] if isinstance(opponent_info, (tuple, list)) else "Unknown"
        opp_pos = opponent_info[3] if isinstance(opponent_info, (tuple, list)) and len(opponent_info) >= 4 else None

        if opp_name not in self.opponent_histories:
            self.opponent_histories[opp_name] = deque(maxlen=5)

        if opp_pos:
            self.opponent_histories[opp_name].append(opp_pos)

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


def _end_episode_from_act(self, game_state, events=None):
    """End episode and save metrics (called from act when new round detected or end_of_round)."""
    if not hasattr(self, 'metrics_tracker') or not self.episode_active:
        return

    # Record final events if provided with evaluation rewards
    if events and hasattr(self, 'metrics_tracker'):
        from evaluation_rewards import EVALUATION_REWARDS
        for event in events:
            event_name = event if isinstance(event, str) else str(event)
            reward = EVALUATION_REWARDS.get(event_name, 0.0)
            self.metrics_tracker.record_event(event_name, reward=reward)

    import events as e
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
        metadata={"mode": "llm_battle"}
    )

    self.metrics_tracker.save()
    self.episode_active = False


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """Called at the end of each round during play mode."""
    if hasattr(self, 'episode_active') and self.episode_active:
        # Record final action if provided
        if last_action and hasattr(self, 'metrics_tracker'):
            self.metrics_tracker.record_action(last_action)
        _end_episode_from_act(self, last_game_state, events)
