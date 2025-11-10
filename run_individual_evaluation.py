#!/usr/bin/env python3
"""
Individual Agent Evaluation Script
===================================
Runs each agent separately against rule-based opponents for fair comparison.

Each agent plays solo with 3 rule_based opponents, collecting their own metrics.
This gives a better measure of individual agent capability without competition.

IMPORTANT: Saves results to separate directory to avoid overwriting multi-agent results!
- Multi-agent results: agent_code/{agent}/evaluation_metrics/
- Individual results: agent_code/{agent}/individual_metrics/

Usage:
    python run_individual_evaluation.py --agents q_learning dqn_final maverick --n-rounds 100
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json

def setup_individual_metrics_dir(agent_name):
    """Create individual_metrics directory for the agent."""
    metrics_dir = Path(f"agent_code/{agent_name}/individual_metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir


def run_individual_evaluation(agent_name, n_rounds=100):
    """
    Run a single agent with rule-based opponents.

    Temporarily changes the agent's save_dir to 'individual_metrics' to avoid
    overwriting multi-agent evaluation results.

    Args:
        agent_name: Name of the agent to evaluate
        n_rounds: Number of rounds to run
    """
    print("=" * 80)
    print(f"EVALUATING: {agent_name} (Individual Mode)")
    print("=" * 80)
    print(f"Rounds: {n_rounds}")
    print(f"Opponents: 3x rule_based bots")

    # Setup individual metrics directory
    metrics_dir = setup_individual_metrics_dir(agent_name)
    print(f"Metrics will be saved to: {metrics_dir}")
    print()

    # Build command - single agent in training mode (for metrics tracking)
    # cmd = [
    #     sys.executable, "main.py", "play",
    #     "--agents", agent_name, "rule_based_agent", "rule_based_agent", "rule_based_agent",
    #     "--n-rounds", str(n_rounds),
    #     "--no-gui",
    #     "--train", "1"  # Just this one agent
    # ]
    cmd = [
        sys.executable, "main.py", "play",
        "--agents", agent_name,
        "--n-rounds", str(n_rounds),
        "--no-gui",
        "--train", "1"  # Just this one agent
    ]
    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {agent_name} evaluation complete!")

        # Move metrics from evaluation_metrics to individual_metrics
        move_metrics_to_individual(agent_name, n_rounds)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {agent_name} evaluation failed: {e}")
        return False


def move_metrics_to_individual(agent_name, n_rounds):
    """Move the latest metrics from evaluation_metrics to individual_metrics."""
    import shutil
    from glob import glob

    eval_dir = Path(f"agent_code/{agent_name}/evaluation_metrics")
    individual_dir = Path(f"agent_code/{agent_name}/individual_metrics")

    if not eval_dir.exists():
        return

    # Find the metrics files that were just created (highest episode numbers)
    pkl_files = sorted(eval_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    json_files = sorted(eval_dir.glob("*summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Move the most recent n_rounds files
    files_to_move = pkl_files[:n_rounds] + json_files[:min(n_rounds, len(json_files))]

    for file in files_to_move:
        dest = individual_dir / file.name
        shutil.move(str(file), str(dest))
        print(f"  Moved: {file.name} → individual_metrics/")


def main():
    parser = argparse.ArgumentParser(
        description="Run individual agent evaluations (each vs rule_based bots)"
    )

    parser.add_argument(
        '--agents',
        type=str,
        nargs='+',
        required=True,
        help='Agent names to evaluate individually'
    )

    parser.add_argument(
        '--n-rounds',
        type=int,
        default=100,
        help='Number of rounds per agent (default: 100)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("INDIVIDUAL AGENT EVALUATION")
    print("=" * 80)
    print(f"\nAgents to evaluate: {', '.join(args.agents)}")
    print(f"Rounds per agent: {args.n_rounds}")
    print(f"Mode: Individual (each agent vs 3 rule_based bots)")
    print("\n" + "=" * 80)
    print()

    # Run each agent individually
    results = {}
    for agent_name in args.agents:
        success = run_individual_evaluation(agent_name, args.n_rounds)
        results[agent_name] = success
        print()

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    for agent_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {agent_name:<20} {status}")
    print("=" * 80)

    # Now load and compare metrics from individual_metrics directories
    print("\nLoading metrics and generating comparison table...")
    print()

    # Import the comparison function from the other script
    try:
        from run_evaluation_and_tabulate import load_metrics
        import json

        print("=" * 80)
        print("INDIVIDUAL EVALUATION RESULTS - STANDARDIZED REWARDS")
        print("=" * 80)
        print()

        # Collect stats for each agent from individual_metrics
        agent_stats = {}
        for agent_name in args.agents:
            # Try to load from individual_metrics directory
            tracker = load_metrics(agent_name, "individual_metrics")

            if tracker is None:
                print(f"⚠️  No individual metrics found for {agent_name}")
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
            print("❌ No metrics data available.")
            return 1

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
        results_file = "llm_maverick_v3_coins.json"
        with open(results_file, 'w') as f:
            json.dump({
                'mode': 'individual',
                'description': 'Coin collections',
                'rounds_per_agent': args.n_rounds,
                'rankings': [
                    {'rank': i+1, 'agent': name, **stats}
                    for i, (name, stats) in enumerate(sorted_agents)
                ]
            }, f, indent=2)

        print(f"\n✅ Individual evaluation results saved to: {results_file}")

    except Exception as e:
        print(f"Could not generate comparison table: {e}")
        import traceback
        traceback.print_exc()

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
