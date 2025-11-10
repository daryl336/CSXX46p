"""
Training visualization module for PPO agent.
Inspired by Maverick's training progress visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os


class TrainingVisualizer:
    """Tracks and visualizes training metrics."""

    def __init__(self, log_dir="logs", load_existing=True):
        """
        Initialize training visualizer.

        Args:
            log_dir: Directory to save visualizations and CSV files
            load_existing: If True, load existing metrics from CSV file
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Metrics tracking
        self.rounds = []
        self.scores = []
        self.returns = []
        self.bomb_usage_rates = []
        self.survival_rates = []
        self.death_rates = []
        self.entropy_values = []
        self.learning_rates = []
        self.actor_losses = []
        self.critic_losses = []

        # Per-agent metrics tracking (for self-play)
        self.per_agent_scores = {}  # {agent_id: [scores]}
        self.per_agent_returns = {}  # {agent_id: [returns]}

        # Load existing data if available
        if load_existing:
            self.load_csv()

    def record_round(self, round_num, score=0, episode_return=0, bomb_usage=0.0,
                    survival_rate=0.0, death_rate=0.0, entropy=0.0, lr=0.0,
                    actor_loss=0.0, critic_loss=0.0, per_agent_metrics=None):
        """
        Record metrics for a single round.

        Args:
            round_num: Round number
            score: True game score (coins + kills) - collective average
            episode_return: Total return from RL agent - collective average
            bomb_usage: Bomb usage rate (0-1)
            survival_rate: Survival rate after bombing (0-1)
            death_rate: Self-destruction rate (0-1)
            entropy: Current entropy coefficient
            lr: Current learning rate
            actor_loss: Actor loss value
            critic_loss: Critic loss value
            per_agent_metrics: Dict with per-agent metrics {agent_id: {'score': x, 'return': y}}
        """
        self.rounds.append(round_num)
        self.scores.append(score)
        self.returns.append(episode_return)
        self.bomb_usage_rates.append(bomb_usage)
        self.survival_rates.append(survival_rate)
        self.death_rates.append(death_rate)
        self.entropy_values.append(entropy)
        self.learning_rates.append(lr)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)

        # Record per-agent metrics if provided
        if per_agent_metrics:
            for agent_id, metrics in per_agent_metrics.items():
                if agent_id not in self.per_agent_scores:
                    self.per_agent_scores[agent_id] = []
                    self.per_agent_returns[agent_id] = []

                self.per_agent_scores[agent_id].append(metrics.get('score', 0.0))
                self.per_agent_returns[agent_id].append(metrics.get('return', 0.0))

    def save_csv(self, append_mode=False):
        """
        Save all metrics to CSV file, including per-agent metrics if available.

        Args:
            append_mode: If True, append only the last recorded round to existing CSV.
                        If False, save all metrics (overwriting existing file).
        """
        csv_path = os.path.join(self.log_dir, "training_metrics.csv")

        # Get sorted list of agent IDs for consistent column ordering
        agent_ids = sorted(self.per_agent_scores.keys())

        if append_mode and len(self.rounds) > 0:
            # Append only the last round to existing CSV
            file_exists = os.path.exists(csv_path)

            # Get last row of data
            last_row = [
                self.rounds[-1],
                self.scores[-1],
                self.returns[-1],
                self.bomb_usage_rates[-1],
                self.survival_rates[-1],
                self.death_rates[-1],
                self.entropy_values[-1],
                self.learning_rates[-1],
                self.actor_losses[-1],
                self.critic_losses[-1]
            ]

            # Add per-agent metrics
            for agent_id in agent_ids:
                last_row.append(self.per_agent_scores[agent_id][-1])
                last_row.append(self.per_agent_returns[agent_id][-1])

            with open(csv_path, 'a') as f:
                if not file_exists:
                    # Write header if file doesn't exist
                    header = "round,score,return,bomb_usage,survival_rate,death_rate,entropy,learning_rate,actor_loss,critic_loss"
                    for agent_id in agent_ids:
                        # Shorten agent IDs for readability
                        short_id = f"agent{agent_ids.index(agent_id)}"
                        header += f",{short_id}_score,{short_id}_return"
                    f.write(header + "\n")

                # Write data row
                format_str = "%d,%.2f,%.2f,%.4f,%.4f,%.4f,%.6f,%.6f,%.4f,%.4f"
                format_str += ",%.2f,%.2f" * len(agent_ids)
                f.write(format_str % tuple(last_row) + "\n")
        else:
            # Save all metrics (overwrite mode)
            header = "round,score,return,bomb_usage,survival_rate,death_rate,entropy,learning_rate,actor_loss,critic_loss"

            # Create data array with base metrics
            data_columns = [
                self.rounds,
                self.scores,
                self.returns,
                self.bomb_usage_rates,
                self.survival_rates,
                self.death_rates,
                self.entropy_values,
                self.learning_rates,
                self.actor_losses,
                self.critic_losses
            ]

            # Add per-agent columns
            format_parts = ['%d', '%.2f', '%.2f', '%.4f', '%.4f', '%.4f', '%.6f', '%.6f', '%.4f', '%.4f']
            for agent_id in agent_ids:
                short_id = f"agent{agent_ids.index(agent_id)}"
                header += f",{short_id}_score,{short_id}_return"
                data_columns.append(self.per_agent_scores[agent_id])
                data_columns.append(self.per_agent_returns[agent_id])
                format_parts.extend(['%.2f', '%.2f'])

            data = np.column_stack(data_columns)

            # Save to CSV
            np.savetxt(csv_path, data, delimiter=',', header=header, comments='',
                       fmt=','.join(format_parts))

        return csv_path

    def load_csv(self):
        """
        Load existing metrics from CSV file.

        Returns:
            bool: True if CSV was loaded successfully, False otherwise
        """
        csv_path = os.path.join(self.log_dir, "training_metrics.csv")

        if not os.path.exists(csv_path):
            return False

        try:
            # Load CSV data (skip header)
            data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)

            # Handle empty or single-row CSV
            if data.ndim == 1 and len(data) > 0:
                data = data.reshape(1, -1)

            if len(data) == 0:
                return False

            # Extract columns
            self.rounds = data[:, 0].astype(int).tolist()
            self.scores = data[:, 1].tolist()
            self.returns = data[:, 2].tolist()
            self.bomb_usage_rates = data[:, 3].tolist()
            self.survival_rates = data[:, 4].tolist()
            self.death_rates = data[:, 5].tolist()
            self.entropy_values = data[:, 6].tolist()
            self.learning_rates = data[:, 7].tolist()
            self.actor_losses = data[:, 8].tolist()
            self.critic_losses = data[:, 9].tolist()

            return True

        except Exception as e:
            print(f"Failed to load CSV: {e}")
            return False

    def plot_training_progress(self, smooth=True, window_size=None):
        """
        Generate comprehensive training progress plots.

        Args:
            smooth: Whether to apply smoothing to plots
            window_size: Size of smoothing window (auto-calculated if None)
        """
        if len(self.rounds) == 0:
            return

        # Auto-calculate window size
        if window_size is None:
            window_size = max(1, len(self.rounds) // 25)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('PPO Training Progress', fontsize=24, fontweight='bold')

        # Plot 1: Score over time (main metric - like Maverick)
        ax = axes[0, 0]
        self._plot_metric(ax, self.rounds, self.scores, 'Score (Coins + Kills)',
                         'Score', smooth, window_size, use_colormap=True)

        # Plot 2: Returns over time
        ax = axes[0, 1]
        self._plot_metric(ax, self.rounds, self.returns, 'Episode Returns',
                         'Return', smooth, window_size, use_colormap=False)

        # Plot 3: Bomb usage rate
        ax = axes[1, 0]
        self._plot_metric(ax, self.rounds, [r * 100 for r in self.bomb_usage_rates],
                         'Bomb Usage Rate', 'Usage %', smooth, window_size,
                         use_colormap=False, color='orange')

        # Plot 4: Survival vs Death rates
        ax = axes[1, 1]
        if smooth:
            survival_smooth = uniform_filter1d(self.survival_rates, window_size, mode="nearest")
            death_smooth = uniform_filter1d(self.death_rates, window_size, mode="nearest")
        else:
            survival_smooth = self.survival_rates
            death_smooth = self.death_rates

        ax.plot(self.rounds, [r * 100 for r in survival_smooth],
               label='Survival Rate', color='green', linewidth=2, alpha=0.8)
        ax.plot(self.rounds, [r * 100 for r in death_smooth],
               label='Death Rate', color='red', linewidth=2, alpha=0.8)
        ax.set_title('Bomb Outcomes', fontsize=16, fontweight='bold')
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Rate %', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot 5: Learning parameters (entropy + LR)
        ax = axes[2, 0]
        ax2 = ax.twinx()

        ax.plot(self.rounds, self.entropy_values, label='Entropy', color='blue', linewidth=2)
        ax2.plot(self.rounds, self.learning_rates, label='Learning Rate', color='purple', linewidth=2)

        ax.set_title('Learning Parameters', fontsize=16, fontweight='bold')
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Entropy', fontsize=14, color='blue')
        ax2.set_ylabel('Learning Rate', fontsize=14, color='purple')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax.grid(True, alpha=0.3)

        # Plot 6: Losses (dual y-axis for separate scales)
        ax = axes[2, 1]
        if len(self.actor_losses) > 0:
            # Create second y-axis for critic loss
            ax2 = ax.twinx()

            if smooth and len(self.actor_losses) > window_size:
                actor_smooth = uniform_filter1d(self.actor_losses, window_size, mode="nearest")
                critic_smooth = uniform_filter1d(self.critic_losses, window_size, mode="nearest")
            else:
                actor_smooth = self.actor_losses
                critic_smooth = self.critic_losses

            # Plot actor loss on left axis (red)
            ax.plot(self.rounds, actor_smooth, label='Actor Loss', color='red', linewidth=2, alpha=0.8)
            # Plot critic loss on right axis (blue)
            ax2.plot(self.rounds, critic_smooth, label='Critic Loss', color='blue', linewidth=2, alpha=0.8)

        ax.set_title('Training Losses (Separate Scales)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel('Actor Loss', fontsize=14, color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)

        if len(self.actor_losses) > 0:
            ax2.set_ylabel('Critic Loss', fontsize=14, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            # Add combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper right')

        # Save figure
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _plot_metric(self, ax, x, y, title, ylabel, smooth, window_size,
                    use_colormap=False, color=None):
        """
        Plot a single metric with optional smoothing and color mapping.

        Args:
            ax: Matplotlib axis
            x: X values (rounds)
            y: Y values (metric)
            title: Plot title
            ylabel: Y-axis label
            smooth: Whether to smooth
            window_size: Smoothing window size
            use_colormap: Whether to use colormap (Maverick style)
            color: Fixed color (if not using colormap)
        """
        if len(x) == 0 or len(y) == 0:
            return

        # Apply smoothing
        if smooth and len(y) > window_size:
            y_smooth = uniform_filter1d(y, window_size, mode="nearest")
        else:
            y_smooth = y

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Round', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.tick_params(labelsize=12)

        # Plot line
        if color:
            ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
            ax.plot(x, y_smooth, color=color, linewidth=2, alpha=0.9, zorder=2)
        else:
            ax.plot(x, y, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
            ax.plot(x, y_smooth, color='steelblue', linewidth=2, alpha=0.9, zorder=2)

        # Add scatter with colormap (Maverick style)
        if use_colormap:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "", ["red", "darkorange", "green"]
            )
            scatter = ax.scatter(x, y, c=y, cmap=cmap, s=30, alpha=0.6, zorder=3)

    def generate_summary_report(self):
        """
        Generate text summary of training progress.

        Returns:
            str: Summary text
        """
        if len(self.rounds) == 0:
            return "No training data available."

        # Calculate statistics
        recent_window = min(100, len(self.scores))

        summary = []
        summary.append("=" * 60)
        summary.append("TRAINING SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total rounds: {self.rounds[-1]}")
        summary.append("")

        summary.append("RECENT PERFORMANCE (last {} rounds):".format(recent_window))
        summary.append(f"  Avg Score: {np.mean(self.scores[-recent_window:]):.2f}")
        summary.append(f"  Avg Return: {np.mean(self.returns[-recent_window:]):.2f}")
        summary.append(f"  Bomb Usage: {np.mean(self.bomb_usage_rates[-recent_window:]) * 100:.1f}%")
        summary.append(f"  Survival Rate: {np.mean(self.survival_rates[-recent_window:]) * 100:.1f}%")
        summary.append(f"  Death Rate: {np.mean(self.death_rates[-recent_window:]) * 100:.1f}%")
        summary.append("")

        summary.append("OVERALL STATISTICS:")
        summary.append(f"  Best Score: {max(self.scores):.2f} (round {self.rounds[np.argmax(self.scores)]})")
        summary.append(f"  Best Return: {max(self.returns):.2f} (round {self.rounds[np.argmax(self.returns)]})")
        summary.append(f"  Peak Bomb Usage: {max(self.bomb_usage_rates) * 100:.1f}%")
        summary.append("")

        summary.append("CURRENT PARAMETERS:")
        if len(self.entropy_values) > 0:
            summary.append(f"  Entropy: {self.entropy_values[-1]:.6f}")
        if len(self.learning_rates) > 0:
            summary.append(f"  Learning Rate: {self.learning_rates[-1]:.6f}")
        summary.append("=" * 60)

        summary_text = "\n".join(summary)

        # Save to file
        summary_path = os.path.join(self.log_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)

        return summary_text
