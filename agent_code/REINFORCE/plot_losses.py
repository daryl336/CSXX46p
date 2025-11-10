#!/usr/bin/env python3
"""
Plot training losses from REINFORCE training logs.

Usage:
    python plot_losses.py                          # Plot latest log file
    python plot_losses.py --file <filename>        # Plot specific file
    python plot_losses.py --all                    # Plot all log files
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_data(filepath):
    """Load training data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def smooth_curve(data, window=50):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2)
        smoothed.append(np.mean(data[start:end]))

    return smoothed


def plot_training_metrics(data, title_suffix="", save_path=None):
    """
    Plot training metrics: loss and reward curves.

    Args:
        data: List of dicts with 'round', 'loss', 'reward', 'episode_length'
        title_suffix: Additional text for plot title
        save_path: If provided, save figure to this path
    """
    rounds = [d['round'] for d in data]
    losses = [d['loss'] for d in data]
    rewards = [d['reward'] for d in data]
    lengths = [d['episode_length'] for d in data]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'REINFORCE Training Metrics{title_suffix}', fontsize=16, fontweight='bold')

    # 1. Loss curve
    ax = axes[0, 0]
    ax.plot(rounds, losses, alpha=0.3, label='Raw Loss', color='blue')
    smoothed_loss = smooth_curve(losses, window=50)
    ax.plot(rounds, smoothed_loss, label='Smoothed Loss (window=50)',
            color='darkblue', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Policy Loss')
    ax.set_title('Policy Loss Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Reward curve
    ax = axes[0, 1]
    ax.plot(rounds, rewards, alpha=0.3, label='Raw Reward', color='green')
    smoothed_reward = smooth_curve(rewards, window=50)
    ax.plot(rounds, smoothed_reward, label='Smoothed Reward (window=50)',
            color='darkgreen', linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero Reward')
    ax.set_xlabel('Round')
    ax.set_ylabel('Total Episode Reward')
    ax.set_title('Episode Reward Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Episode length
    ax = axes[1, 0]
    ax.plot(rounds, lengths, alpha=0.3, label='Raw Length', color='orange')
    smoothed_length = smooth_curve(lengths, window=50)
    ax.plot(rounds, smoothed_length, label='Smoothed Length (window=50)',
            color='darkorange', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Length Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Loss vs Reward scatter (to see correlation)
    ax = axes[1, 1]
    scatter = ax.scatter(losses, rewards, alpha=0.5, c=rounds,
                         cmap='viridis', s=20)
    ax.set_xlabel('Policy Loss')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Loss vs Reward (colored by round)')
    plt.colorbar(scatter, ax=ax, label='Round')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_multiple_runs(log_dir="training_logs"):
    """Plot all training runs in the log directory."""
    log_files = sorted(Path(log_dir).glob("training_losses_*.json"))

    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} training runs")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Comparison of Multiple Training Runs', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))

    for i, log_file in enumerate(log_files):
        data = load_training_data(log_file)
        rounds = [d['round'] for d in data]
        losses = [d['loss'] for d in data]
        rewards = [d['reward'] for d in data]

        # Smooth the curves
        smoothed_loss = smooth_curve(losses, window=50)
        smoothed_reward = smooth_curve(rewards, window=50)

        # Extract timestamp from filename
        timestamp = log_file.stem.replace("training_losses_", "")
        label = f"Run {timestamp}"

        # Plot loss
        axes[0].plot(rounds, smoothed_loss, label=label,
                     color=colors[i], linewidth=2, alpha=0.7)

        # Plot reward
        axes[1].plot(rounds, smoothed_reward, label=label,
                     color=colors[i], linewidth=2, alpha=0.7)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Policy Loss (smoothed)')
    axes[0].set_title('Policy Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Episode Reward (smoothed)')
    axes[1].set_title('Episode Reward Comparison')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_statistics(data):
    """Print summary statistics of training run."""
    rounds = [d['round'] for d in data]
    losses = [d['loss'] for d in data]
    rewards = [d['reward'] for d in data]
    lengths = [d['episode_length'] for d in data]

    print("\n" + "=" * 60)
    print("TRAINING STATISTICS")
    print("=" * 60)
    print(f"Total rounds: {len(data)}")
    print(f"Round range: {min(rounds)} - {max(rounds)}")
    print(f"\nPolicy Loss:")
    print(f"  Mean: {np.mean(losses):.4f}")
    print(f"  Std:  {np.std(losses):.4f}")
    print(f"  Min:  {np.min(losses):.4f}")
    print(f"  Max:  {np.max(losses):.4f}")
    print(f"\nEpisode Reward:")
    print(f"  Mean: {np.mean(rewards):.2f}")
    print(f"  Std:  {np.std(rewards):.2f}")
    print(f"  Min:  {np.min(rewards):.2f}")
    print(f"  Max:  {np.max(rewards):.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {np.mean(lengths):.1f} steps")
    print(f"  Std:  {np.std(lengths):.1f} steps")

    # Check for improvement (compare first 100 vs last 100)
    if len(data) >= 200:
        early_reward = np.mean([d['reward'] for d in data[:100]])
        late_reward = np.mean([d['reward'] for d in data[-100:]])
        improvement = late_reward - early_reward
        print(f"\nImprovement (last 100 vs first 100 rounds):")
        print(f"  Early avg reward: {early_reward:.2f}")
        print(f"  Late avg reward:  {late_reward:.2f}")
        print(f"  Change: {improvement:+.2f} ({improvement / abs(early_reward) * 100:+.1f}%)")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot REINFORCE training losses')
    parser.add_argument('--file', type=str, help='Specific log file to plot')
    parser.add_argument('--all', action='store_true', help='Plot all training runs')
    parser.add_argument('--log-dir', type=str, default='training_logs',
                        help='Directory containing log files')
    parser.add_argument('--save', type=str, help='Save plot to this file')
    parser.add_argument('--no-show', action='store_true',
                        help='Don\'t display plot (only save)')

    args = parser.parse_args()

    if args.all:
        # Plot multiple runs
        plot_multiple_runs(args.log_dir)
    elif args.file:
        # Plot specific file
        data = load_training_data(args.file)
        print_statistics(data)
        plot_training_metrics(data, save_path=args.save)
    else:
        # Plot latest file
        log_files = sorted(Path(args.log_dir).glob("training_losses_*.json"))
        if not log_files:
            print(f"No log files found in {args.log_dir}")
            return

        latest_file = log_files[-1]
        print(f"Plotting latest training run: {latest_file}")
        data = load_training_data(latest_file)
        print_statistics(data)
        plot_training_metrics(data, save_path=args.save)


if __name__ == "__main__":
    main()