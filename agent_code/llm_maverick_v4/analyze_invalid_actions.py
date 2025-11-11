#!/usr/bin/env python3
"""
Analyze invalid actions from llm_maverick_v3 metrics.
"""

import pickle
import json
from pathlib import Path
from collections import Counter

metrics_dir = Path("individual_metrics")
pkl_files = list(metrics_dir.glob("*.pkl"))

print("=" * 80)
print("INVALID ACTIONS ANALYSIS")
print("=" * 80)
print()

all_invalid_actions = []
total_actions = 0
total_invalid = 0

for pkl_file in pkl_files[:5]:  # Analyze first 5 episodes
    print(f"Analyzing: {pkl_file.name}")
    print("-" * 80)

    with open(pkl_file, 'rb') as f:
        episode_data = pickle.load(f)

    # Extract transitions
    transitions = episode_data.get('transitions', [])

    episode_invalid = 0
    episode_actions = []

    for transition in transitions:
        action = transition.get('action')
        events = transition.get('events', [])

        total_actions += 1
        episode_actions.append(action)

        # Check if INVALID_ACTION event occurred
        if 'INVALID_ACTION' in events:
            total_invalid += 1
            episode_invalid += 1
            all_invalid_actions.append(action)

            # Print context
            print(f"  Step {len(episode_actions)}: Invalid action '{action}'")

            # Try to get game state context
            if 'old_game_state' in transition:
                game_state = transition['old_game_state']
                self_pos = game_state.get('self', [None, None, None, None])[3]
                field = game_state.get('field')
                print(f"    Position: {self_pos}")

                # Check what would happen if agent tried this action
                if self_pos and field is not None:
                    x, y = self_pos
                    deltas = {
                        'UP': (0, -1),
                        'DOWN': (0, 1),
                        'LEFT': (-1, 0),
                        'RIGHT': (1, 0)
                    }
                    if action in deltas:
                        dx, dy = deltas[action]
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                            tile_value = field[nx, ny]
                            tile_type = {-1: "WALL", 0: "FREE", 1: "CRATE"}.get(tile_value, f"UNKNOWN({tile_value})")
                            print(f"    Target tile ({nx},{ny}): {tile_type}")
                        else:
                            print(f"    Target tile ({nx},{ny}): OUT OF BOUNDS")

    invalid_rate = (episode_invalid / len(episode_actions) * 100) if episode_actions else 0
    print(f"  Episode invalid rate: {episode_invalid}/{len(episode_actions)} ({invalid_rate:.1f}%)")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total actions: {total_actions}")
print(f"Total invalid: {total_invalid}")
print(f"Invalid rate: {total_invalid/total_actions*100:.2f}%" if total_actions > 0 else "N/A")
print()

if all_invalid_actions:
    print("Invalid action breakdown:")
    action_counts = Counter(all_invalid_actions)
    for action, count in action_counts.most_common():
        print(f"  {action}: {count} times ({count/total_invalid*100:.1f}%)")
else:
    print("No invalid actions found!")

print()
print("=" * 80)
