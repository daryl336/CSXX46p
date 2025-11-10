#!/usr/bin/env python3
"""
Check actual self-kill data from the most recent training run.
"""

import os
import pickle
import glob

metrics_dir = "agent_code/ppo/metrics"

# Find the most recent metrics file
metrics_files = glob.glob(os.path.join(metrics_dir, "PPO_*.pkl"))
if not metrics_files:
    print("No metrics files found!")
    exit(1)

# Get the most recent file
latest_file = max(metrics_files, key=os.path.getmtime)
print(f"Loading metrics from: {latest_file}")

# Load the data
with open(latest_file, 'rb') as f:
    data = pickle.load(f)

episodes = data['episodes']
total_episodes = len(episodes)

# Analyze self-kills
self_kill_count = 0
episodes_with_self_kills = 0
survival_times = []
ranks = []

for ep in episodes:
    if ep['self_kills'] > 0:
        self_kill_count += ep['self_kills']
        episodes_with_self_kills += 1
    survival_times.append(ep['survival_time'])
    ranks.append(ep['rank'])

print(f"\nTotal episodes analyzed: {total_episodes}")
print(f"Episodes with self-kills: {episodes_with_self_kills}")
print(f"Total self-kills: {self_kill_count}")
print(f"Self-kill rate (per episode): {self_kill_count / total_episodes:.4f}")
print(f"Episode with self-kill %: {episodes_with_self_kills / total_episodes * 100:.1f}%")

import numpy as np
print(f"\nAvg survival time: {np.mean(survival_times):.2f}")
print(f"Avg rank: {np.mean(ranks):.2f}")

# Check last 100 episodes
print("\n--- LAST 100 EPISODES ---")
recent_episodes = episodes[-100:]
recent_self_kills = sum(1 for ep in recent_episodes if ep['self_kills'] > 0)
recent_survival = np.mean([ep['survival_time'] for ep in recent_episodes])
recent_ranks = np.mean([ep['rank'] for ep in recent_episodes])

print(f"Episodes with self-kills: {recent_self_kills}/100 ({recent_self_kills}%)")
print(f"Avg survival time: {recent_survival:.2f}")
print(f"Avg rank: {recent_ranks:.2f}")

# Show a few recent episodes with details
print("\n--- LAST 10 EPISODES DETAILS ---")
for i, ep in enumerate(episodes[-10:], 1):
    print(f"Episode {ep['episode_id']}: survival={ep['survival_time']}, rank={ep['rank']}, "
          f"self_kills={ep['self_kills']}, crates={ep['crates_destroyed']}, "
          f"coins={ep['coins_collected']}, bombs={ep['bombs_placed']}")
