#!/usr/bin/env python3
"""
Post-Game Evaluation Script

Runs agents in play mode and calculates standardized evaluation metrics
by tracking game events and computing rewards retrospectively.

Usage:
    python evaluate_agents.py --agents q_learning ppo dqn_final --n-rounds 100
"""

import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import re

from evaluation_rewards import EVALUATION_REWARDS, get_reward_from_events
import events as e


class AgentEvaluator:
    """Evaluates agents by parsing game logs and computing standardized metrics."""

    def __init__(self):
        self.agent_stats = defaultdict(lambda: {
            'total_reward': 0.0,
            'episodes': 0,
            'coins_collected': 0,
            'kills': 0,
            'deaths': 0,
            'suicides': 0,
            'survived_rounds': 0,
            'crates_destroyed': 0,
            'invalid_actions': 0,
            'rewards_per_episode': [],
        })

    def parse_log_file(self, log_file):
        """
        Parse game log file to extract events and calculate rewards.

        This is a template - you'll need to adapt it to your actual log format.
        """
        print(f"Parsing log file: {log_file}")

        if not Path(log_file).exists():
            print(f"Warning: Log file {log_file} not found")
            return

        with open(log_file, 'r') as f:
            content = f.read()

        # Parse events from logs (this is game-dependent - adjust as needed)
        # Look for patterns like:
        # - "Agent X collected coin"
        # - "Agent X killed Agent Y"
        # - "Agent X died"
        # etc.

        # Example parsing (ADJUST TO YOUR LOG FORMAT):
        current_agent = None
        current_events = []

        for line in content.split('\n'):
            # Example: Extract agent actions and events from logs
            # You'll need to customize this based on your actual log format

            if 'COIN_COLLECTED' in line or 'collected coin' in line.lower():
                if current_agent:
                    self.agent_stats[current_agent]['coins_collected'] += 1
                    current_events.append(e.COIN_COLLECTED)

            if 'KILLED_OPPONENT' in line or 'killed' in line.lower():
                if current_agent:
                    self.agent_stats[current_agent]['kills'] += 1
                    current_events.append(e.KILLED_OPPONENT)

            if 'KILLED_SELF' in line or 'suicide' in line.lower():
                if current_agent:
                    self.agent_stats[current_agent]['suicides'] += 1
                    self.agent_stats[current_agent]['deaths'] += 1
                    current_events.append(e.KILLED_SELF)

            if 'GOT_KILLED' in line:
                if current_agent:
                    self.agent_stats[current_agent]['deaths'] += 1
                    current_events.append(e.GOT_KILLED)

            if 'SURVIVED_ROUND' in line or 'survived' in line.lower():
                if current_agent:
                    self.agent_stats[current_agent]['survived_rounds'] += 1
                    current_events.append(e.SURVIVED_ROUND)

    def calculate_episode_reward(self, events):
        """Calculate total reward for an episode from events."""
        return get_reward_from_events(events)

    def print_results(self):
        """Print evaluation results in a nice format."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS - Standardized Rewards")
        print("=" * 80)
        print()

        # Sort agents by total reward
        sorted_agents = sorted(
            self.agent_stats.items(),
            key=lambda x: x[1]['total_reward'],
            reverse=True
        )

        # Print header
        print(f"{'Agent':<20} {'Avg Reward':>12} {'Coins':>8} {'Survival':>10} {'Kills':>8} {'Deaths':>8}")
        print("-" * 80)

        for agent_name, stats in sorted_agents:
            episodes = stats['episodes'] if stats['episodes'] > 0 else 1
            avg_reward = stats['total_reward'] / episodes
            survival_rate = (stats['survived_rounds'] / episodes) * 100

            print(f"{agent_name:<20} {avg_reward:>12.2f} {stats['coins_collected']:>8} "
                  f"{survival_rate:>9.1f}% {stats['kills']:>8} {stats['deaths']:>8}")

        print("-" * 80)
        print()

        # Detailed breakdown
        print("DETAILED METRICS:")
        print("=" * 80)
        for agent_name, stats in sorted_agents:
            print(f"\n{agent_name}:")
            print(f"  Total Episodes: {stats['episodes']}")
            print(f"  Total Reward: {stats['total_reward']:.2f}")
            print(f"  Coins Collected: {stats['coins_collected']}")
            print(f"  Opponents Killed: {stats['kills']}")
            print(f"  Deaths: {stats['deaths']} (Suicides: {stats['suicides']})")
            print(f"  Rounds Survived: {stats['survived_rounds']}")
            print(f"  Crates Destroyed: {stats['crates_destroyed']}")
            print(f"  Invalid Actions: {stats['invalid_actions']}")

        print("\n" + "=" * 80)

        # Save results to JSON
        results_file = "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(dict(self.agent_stats), f, indent=2)
        print(f"\nResults saved to: {results_file}")


def run_evaluation(agents, n_rounds=100, no_gui=True):
    """
    Run agents in play mode and collect statistics.

    Args:
        agents: List of agent names
        n_rounds: Number of rounds to run
        no_gui: Run without GUI for faster evaluation
    """
    print(f"Running evaluation for {len(agents)} agents over {n_rounds} rounds...")
    print(f"Agents: {', '.join(agents)}")
    print()

    # Build command
    cmd = [
        sys.executable, "main.py", "play",
        "--agents", *agents,
        "--n-rounds", str(n_rounds),
    ]

    if no_gui:
        cmd.append("--no-gui")

    print(f"Command: {' '.join(cmd)}")
    print()

    # Run the game
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Game completed successfully!")
        print()

        # Parse output for basic stats
        output = result.stdout
        print(output)

        return output

    except subprocess.CalledProcessError as e:
        print(f"Error running game: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return None


def simple_evaluation(agents, n_rounds=100):
    """
    Simple evaluation that just tracks game scores (coins + kills).
    This works without parsing logs by using the game's built-in scoring.
    """
    print("=" * 80)
    print("SIMPLE EVALUATION - Using Game Scores")
    print("=" * 80)
    print()
    print(f"Running {n_rounds} rounds for agents: {', '.join(agents)}")
    print()
    print("Scoring:")
    print("  - Coins: 1 point each")
    print("  - Kills: 5 points each")
    print()

    output = run_evaluation(agents, n_rounds, no_gui=True)

    if output:
        print("\n" + "=" * 80)
        print("Evaluation complete! Check game logs for detailed results.")
        print("=" * 80)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Bomberman agents with standardized metrics"
    )

    parser.add_argument(
        '--agents',
        type=str,
        nargs='+',
        required=True,
        help='Agent names to evaluate (e.g., q_learning ppo dqn_final)'
    )

    parser.add_argument(
        '--n-rounds',
        type=int,
        default=100,
        help='Number of rounds to run (default: 100)'
    )

    parser.add_argument(
        '--mode',
        choices=['simple', 'detailed'],
        default='simple',
        help='Evaluation mode: simple (game scores only) or detailed (parse logs)'
    )

    args = parser.parse_args()

    if args.mode == 'simple':
        simple_evaluation(args.agents, args.n_rounds)
    else:
        # Detailed evaluation requires log parsing
        print("Detailed evaluation mode:")
        print("  1. Running games...")
        output = run_evaluation(args.agents, args.n_rounds, no_gui=True)

        print("  2. Parsing logs...")
        evaluator = AgentEvaluator()
        # You'll need to specify your log file location
        log_file = "logs/game.log"  # Adjust path as needed
        evaluator.parse_log_file(log_file)

        print("  3. Calculating standardized rewards...")
        evaluator.print_results()


if __name__ == "__main__":
    main()
