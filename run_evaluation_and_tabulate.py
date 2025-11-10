#!/usr/bin/env python3
"""
Run Evaluation and Generate Tabulation

This script:
1. Runs agents with --train flag to enable metrics tracking
2. Uses evaluation_rewards.py for standardized rewards
3. Generates comparison tables from metrics files

Usage:
    python run_evaluation_and_tabulate.py --agents q_learning ppo dqn_final --n-rounds 100
"""

import sys
import subprocess
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
import json

def run_evaluation(agents, n_rounds=100, train_mode=True):
    """
    Run agents with metrics tracking enabled.

    Args:
        agents: List of agent names
        n_rounds: Number of rounds to run
        train_mode: Enable training callbacks for metrics (but agents won't update if evaluation_mode flag set)
    """
    print("=" * 80)
    print("RUNNING EVALUATION WITH STANDARDIZED REWARDS")
    print("=" * 80)
    print(f"\nAgents: {', '.join(agents)}")
    print(f"Rounds: {n_rounds}")
    print(f"Metrics Tracking: {'ENABLED' if train_mode else 'DISABLED'}")
    print()

    # Build command
    cmd = [
        sys.executable, "main.py", "play",
        "--agents", *agents,
        "--n-rounds", str(n_rounds),
        "--no-gui"
    ]

    # Enable training mode for metrics tracking (agents should skip learning if evaluation_mode=True)
    if train_mode:
        cmd.extend(["--train", str(len(agents))])

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run the game
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running evaluation: {e}")
        return False


def load_metrics(agent_name, metrics_dir="evaluation_metrics"):
    """
    Load metrics from the metrics tracker pickle file.

    Args:
        agent_name: Name of the agent
        metrics_dir: Directory where metrics are saved

    Returns:
        MetricsTracker object or None
    """
    from glob import glob

    # Try to find the most recent metrics file
    # MetricsTracker saves as "{agent_name}_{episode_num}.pkl"

    # Try multiple possible locations
    possible_locations = [
        metrics_dir,  # Root evaluation_metrics/
        f"agent_code/{agent_name}/{metrics_dir}",  # agent_code/q_learning/evaluation_metrics/
        f"agent_code/{agent_name}/metrics",  # Legacy location
    ]

    all_matching_files = []

    for location in possible_locations:
        if not Path(location).exists():
            continue

        # Look for files matching the agent name pattern
        # Try different name variations (case-insensitive)
        name_patterns = [
            f"{location}/{agent_name}*.pkl",                                    # lowercase: ppo
            f"{location}/{agent_name.upper()}*.pkl",                           # UPPERCASE: PPO
            f"{location}/{agent_name.replace('_', ' ').title()}*.pkl",        # Title Case: Q Learning, Dqn Final
            f"{location}/{agent_name.capitalize()}*.pkl",                      # Capitalized: Ppo
            f"{location}/*{agent_name}*.pkl",                                   # Wildcard lowercase
            f"{location}/*{agent_name.upper()}*.pkl",                          # Wildcard UPPERCASE
            f"{location}/*{agent_name.replace('_', ' ').title()}*.pkl",       # Wildcard Title Case
            f"{location}/{agent_name.replace('_', ' ').upper()}*.pkl",        # Upper with space: DQN FINAL
            f"{location}/*{agent_name.replace('_', ' ').upper()}*.pkl",       # Wildcard Upper with space
        ]

        # Special case: DQN Final (all caps first word)
        if '_' in agent_name:
            parts = agent_name.split('_')
            special_case = ' '.join([parts[0].upper()] + [p.title() for p in parts[1:]])
            name_patterns.append(f"{location}/{special_case}*.pkl")
            name_patterns.append(f"{location}/*{special_case}*.pkl")

        # Special case: q_llm saves as "Hybrid Q-LLM Training"
        if agent_name == 'q_llm':
            name_patterns.append(f"{location}/Hybrid Q-LLM Training*.pkl")
            name_patterns.append(f"{location}/*Hybrid Q-LLM Training*.pkl")

        # Special case: ppo_llm saves as "PPO→LLM Sequential Agent" (with arrow character)
        if agent_name == 'ppo_llm':
            name_patterns.append(f"{location}/PPO LLM*.pkl")
            name_patterns.append(f"{location}/*PPO LLM*.pkl")
            name_patterns.append(f"{location}/PPO→LLM*.pkl")  # Arrow character
            name_patterns.append(f"{location}/*PPO→LLM*.pkl")  # Arrow character wildcard

        # Special case: llm saves as "LLM Agent"
        if agent_name == 'llm':
            name_patterns.append(f"{location}/LLM Agent*.pkl")
            name_patterns.append(f"{location}/*LLM Agent*.pkl")

        # Special case: llm_battle saves as "LLM Battle"
        if agent_name == 'llm_battle':
            name_patterns.append(f"{location}/LLM Battle*.pkl")
            name_patterns.append(f"{location}/*LLM Battle*.pkl")

        # Special case: llm_predict saves as "LLM Predict"
        if agent_name == 'llm_predict':
            name_patterns.append(f"{location}/LLM Predict*.pkl")
            name_patterns.append(f"{location}/*LLM Predict*.pkl")

        # Special case: llm_maverick saves as "LLM Maverick"
        if agent_name == 'llm_maverick':
            name_patterns.append(f"{location}/LLM Maverick*.pkl")
            name_patterns.append(f"{location}/*LLM Maverick*.pkl")

        # Special case: llm_maverick_v3 saves as "LLM Maverick" (same as llm_maverick)
        if agent_name == 'llm_maverick_v3':
            name_patterns.append(f"{location}/LLM Maverick*.pkl")
            name_patterns.append(f"{location}/*LLM Maverick*.pkl")

        # Special case: maverick_enhanced saves as "Maverick Enhanced"
        if agent_name == 'maverick_enhanced':
            name_patterns.append(f"{location}/Maverick Enhanced*.pkl")
            name_patterns.append(f"{location}/*Maverick Enhanced*.pkl")

        for pattern in name_patterns:
            all_matching_files.extend(glob(pattern))

    if not all_matching_files:
        print(f"Warning: No metrics files found for {agent_name}")
        print(f"Searched in: {metrics_dir}/")
        print(f"Patterns: {name_patterns}")
        return None

    # Get the most recent file
    latest_file = max(all_matching_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading metrics from: {latest_file}")

    try:
        with open(latest_file, 'rb') as f:
            data = pickle.load(f)

        # Check if it's a MetricsTracker object or serialized data
        if hasattr(data, 'get_summary_stats'):
            # It's already a MetricsTracker object
            return data
        elif isinstance(data, dict) and 'agent_name' in data:
            # It's serialized data, recreate the MetricsTracker and EpisodeMetrics
            from metrics.metrics_tracker import MetricsTracker, EpisodeMetrics

            tracker = MetricsTracker.__new__(MetricsTracker)

            # Reconstruct episodes as EpisodeMetrics objects
            if 'episodes' in data and isinstance(data['episodes'], list):
                reconstructed_episodes = []
                for ep_dict in data['episodes']:
                    if isinstance(ep_dict, dict):
                        ep = EpisodeMetrics(**ep_dict)
                        reconstructed_episodes.append(ep)
                    else:
                        reconstructed_episodes.append(ep_dict)  # Already an object
                data['episodes'] = reconstructed_episodes

            tracker.__dict__.update(data)
            return tracker
        else:
            print(f"Unknown data format in {latest_file}")
            return None

    except Exception as e:
        print(f"Error loading {latest_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_comparison_table(agents, metrics_dir="metrics"):
    """
    Generate comparison table from metrics files.

    Args:
        agents: List of agent names
        metrics_dir: Directory where metrics are saved
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - STANDARDIZED REWARDS")
    print("=" * 80)
    print()

    # Collect stats for each agent
    agent_stats = {}

    for agent_name in agents:
        tracker = load_metrics(agent_name, metrics_dir)

        if tracker is None:
            print(f"⚠️  No metrics found for {agent_name}")
            continue

        # Get summary statistics
        try:
            summary = tracker.get_summary_stats()

            agent_stats[agent_name] = {
                'episodes': len(tracker.episodes),
                'avg_reward': summary.get('avg_reward', summary.get('avg_total_reward', 0)),
                'total_reward': summary.get('total_reward', summary.get('sum_total_reward', 0)),
                'win_rate': summary.get('win_rate', 0) * 100,
                'coins': summary.get('total_coins', summary.get('sum_coins_collected', 0)),
                'avg_coins': summary.get('avg_coins_per_episode', summary.get('avg_coins_collected', 0)),
                'kills': summary.get('total_kills', summary.get('sum_opponents_killed', 0)),
                'deaths': summary.get('total_deaths', summary.get('sum_deaths', 0)),
                'deaths_by_opponent': summary.get('total_deaths_by_opponent', 0),
                'deaths_by_bomb': summary.get('total_deaths_by_bomb', 0),
                'avg_deaths': summary.get('avg_deaths', 0),
                'survival_rate': summary.get('survival_rate', 0) * 100,  # Already 0-1, multiply by 100 for percentage
                'avg_survival_time': summary.get('avg_survival_time', 0),
                'total_bombs': summary.get('total_bombs_placed', 0),
                'avg_bombs': summary.get('avg_bombs_per_episode', 0),
                'bomb_effectiveness': summary.get('bomb_effectiveness', 0) * 100,
                'total_crates': summary.get('total_crates', 0),
                'avg_crates': summary.get('avg_crates_per_episode', 0),
                'invalid_action_rate': summary.get('invalid_action_rate', 0) * 100,
            }
        except Exception as e:
            print(f"Error processing metrics for {agent_name}: {e}")
            continue

    if not agent_stats:
        print("❌ No metrics data available. Make sure agents ran with --train flag.")
        return

    # Sort by average reward (descending)
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['avg_reward'], reverse=True)

    # Print table header
    print(f"{'Rank':<6} {'Agent':<20} {'Avg Reward':>12} {'Win Rate':>10} {'Coins':>8} {'Kills':>8} {'Deaths':>8}")
    print("-" * 80)

    # Print each agent's stats
    for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "

        print(f"{medal} #{rank:<3} {agent_name:<20} "
              f"{stats['avg_reward']:>12.2f} "
              f"{stats['win_rate']:>9.1f}% "
              f"{stats['coins']:>8} "
              f"{stats['kills']:>8} "
              f"{stats['deaths']:>8}")

    print("-" * 80)
    print()

    # Detailed breakdown
    print("DETAILED METRICS:")
    print("=" * 80)

    for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
        print(f"\n#{rank} {agent_name}:")
        print(f"  Episodes:           {stats['episodes']}")
        print(f"  Total Reward:       {stats['total_reward']:.2f}")
        print(f"  Avg Reward:         {stats['avg_reward']:.2f}")
        print(f"  Win Rate:           {stats['win_rate']:.1f}%")
        print(f"  Avg Survival Time:  {stats['avg_survival_time']:.1f} steps")
        print()
        print(f"  Coins Collected:    {stats['coins']} (avg: {stats['avg_coins']:.2f}/episode)")
        print(f"  Crates Destroyed:   {stats['total_crates']} (avg: {stats['avg_crates']:.2f}/episode)")
        print()
        print(f"  Opponents Killed:   {stats['kills']}")
        print(f"  Deaths (Total):     {stats['deaths']} (avg: {stats['avg_deaths']:.2f}/episode)")
        print(f"    - By Opponent:    {stats['deaths_by_opponent']}")
        print(f"    - Self Kills:     {stats['deaths_by_bomb']}")
        print(f"  Survival Rate:      {stats['survival_rate']:.1f}%")
        print()
        print(f"  Bombs Placed:       {stats['total_bombs']} (avg: {stats['avg_bombs']:.2f}/episode)")
        print(f"  Bomb Effectiveness: {stats['bomb_effectiveness']:.1f}%")
        print(f"  Invalid Actions:    {stats['invalid_action_rate']:.1f}%")

    print("\n" + "=" * 80)

    # Save results to JSON
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'rankings': [
                {'rank': i+1, 'agent': name, **stats}
                for i, (name, stats) in enumerate(sorted_agents)
            ]
        }, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with standardized rewards")

    parser.add_argument('--agents', type=str, nargs='+', required=True,
                       help='Agent names to evaluate')

    parser.add_argument('--n-rounds', type=int, default=100,
                       help='Number of rounds (default: 100)')

    parser.add_argument('--skip-run', action='store_true',
                       help='Skip running evaluation, just generate table from existing metrics')

    parser.add_argument('--metrics-dir', type=str, default='evaluation_metrics',
                       help='Directory where evaluation metrics are saved')

    args = parser.parse_args()

    # Run evaluation
    if not args.skip_run:
        success = run_evaluation(args.agents, args.n_rounds, train_mode=True)
        if not success:
            print("\n❌ Evaluation failed!")
            return 1

    # Generate comparison table
    generate_comparison_table(args.agents, args.metrics_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
