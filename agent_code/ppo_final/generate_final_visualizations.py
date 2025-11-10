#!/usr/bin/env python3
"""
Generate final visualizations from training metrics CSV.

This script can be run after training completes to generate
comprehensive visualizations from the accumulated metrics.

Usage:
    python generate_final_visualizations.py [log_dir]

Arguments:
    log_dir: Optional path to log directory (default: ./logs)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ppo.visualization import TrainingVisualizer
except ImportError:
    from visualization import TrainingVisualizer


def generate_visualizations(log_dir="logs"):
    """
    Generate all visualizations from existing CSV file.

    Args:
        log_dir: Directory containing training_metrics.csv
    """
    print(f"Loading training metrics from {log_dir}...")

    # Initialize visualizer and load CSV
    visualizer = TrainingVisualizer(log_dir=log_dir, load_existing=True)

    if len(visualizer.rounds) == 0:
        print(f"ERROR: No training data found in {log_dir}/training_metrics.csv")
        return False

    print(f"Loaded {len(visualizer.rounds)} rounds of training data")
    print(f"Training range: Round {visualizer.rounds[0]} to {visualizer.rounds[-1]}")

    # Generate comprehensive training progress plots
    print("\nGenerating comprehensive training progress plot...")
    try:
        plot_path = visualizer.plot_training_progress(smooth=True)
        print(f"✓ Saved: {plot_path}")
    except Exception as e:
        print(f"✗ Failed to generate training progress plot: {e}")

    # Generate summary report
    print("\nGenerating summary report...")
    try:
        summary = visualizer.generate_summary_report()
        print(f"✓ Summary saved to {os.path.join(log_dir, 'training_summary.txt')}")
        print("\n" + summary)
    except Exception as e:
        print(f"✗ Failed to generate summary: {e}")

    return True


def main():
    """Main entry point."""
    # Get log directory from command line or use default
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        # Default to logs directory in same folder as this script
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

    print("=" * 70)
    print("PPO Training Visualization Generator")
    print("=" * 70)

    success = generate_visualizations(log_dir)

    if success:
        print("\n" + "=" * 70)
        print("✓ Visualization generation complete!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ Visualization generation failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
