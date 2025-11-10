"""
Comprehensive Metrics Tracking System for Bomberman RL/LLM Agents

This module provides agent-agnostic metrics collection, analysis, and comparison
capabilities for evaluating different agent architectures.
"""

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import pickle

@dataclass
class EpisodeMetrics:
    """Metrics collected for a single episode/round."""
    
    # Episode identification
    episode_id: int
    timestamp: str
    agent_name: str
    
    # Outcome metrics
    won: bool = False
    rank: int = 0  # 1 = won, 2 = second place, etc.
    survival_time: int = 0  # steps survived
    total_steps: int = 0  # total episode length
    
    # Combat metrics
    opponents_killed: int = 0
    self_kills: int = 0
    deaths_by_opponent: int = 0
    deaths_by_bomb: int = 0
    
    # Bomb metrics
    bombs_placed: int = 0
    bombs_hit_opponent: int = 0  # bombs that damaged/killed opponent
    bombs_destroyed_crate: int = 0
    bombs_hit_self: int = 0
    
    # Resource collection
    coins_collected: int = 0
    crates_destroyed: int = 0
    
    # Action metrics
    total_actions: int = 0
    invalid_actions: int = 0
    action_distribution: Dict[str, int] = field(default_factory=dict)

    # Event tracking
    events: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    # Reward metrics
    total_reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Opponent info
    opponent_types: List[str] = field(default_factory=list)
    
    # Environment info
    map_name: str = "default"
    scenario: str = "default"
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EpisodeMetrics':
        """Create from dictionary."""
        return cls(**data)


class MetricsTracker:
    """
    Tracks metrics during agent execution and provides analysis capabilities.
    
    This class is agent-agnostic and can be used with any RL or LLM agent.
    It accumulates metrics over episodes and provides statistical analysis.
    """
    
    def __init__(self, agent_name: str, save_dir: str = "metrics"):
        """
        Initialize metrics tracker.
        
        Args:
            agent_name: Name/identifier for the agent being tracked
            save_dir: Directory to save metrics
        """
        self.agent_name = agent_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Current episode tracking
        self.current_episode: Optional[EpisodeMetrics] = None
        self.episode_events: List[str] = []
        self.episode_actions: List[str] = []
        self.episode_start_step: int = 0
        
        # Historical data
        self.episodes: List[EpisodeMetrics] = []
        self.episode_count: int = 0
        
        # Running statistics (for efficiency)
        self.running_stats = defaultdict(lambda: deque(maxlen=100))
        self.game_rewards = {
            "COIN_COLLECTED": 10.0,
            "KILLED_OPPONENT": 50.0,
            "CRATE_DESTROYED": 5.0,
            "BOMB_DROPPED": 1.0,
            "KILLED_SELF": -100.0,
            "GOT_KILLED": -50.0,
            "INVALID_ACTION": -5.0,
            "WAITED": -1.0,
            "SURVIVED_ROUND": 30.0,
        }
        
    def start_episode(self, episode_id: int, opponent_types: List[str] = None,
                     map_name: str = "default", scenario: str = "default"):
        """
        Start tracking a new episode/round.
        
        Args:
            episode_id: Unique episode identifier
            opponent_types: List of opponent agent types
            map_name: Name of the map
            scenario: Scenario/game mode identifier
        """
        self.current_episode = EpisodeMetrics(
            episode_id=episode_id,
            timestamp=datetime.now().isoformat(),
            agent_name=self.agent_name,
            opponent_types=opponent_types or [],
            map_name=map_name,
            scenario=scenario
        )
        self.episode_events = []
        self.episode_actions = []
        self.episode_start_step = 0
        
    def record_action(self, action: str, is_valid: bool = True):
        """
        Record an action taken by the agent.
        
        Args:
            action: Action name (e.g., 'UP', 'BOMB')
            is_valid: Whether the action was valid
        """
        if self.current_episode is None:
            return

        self.episode_actions.append(action)

        self.current_episode.total_actions += 1
        if not is_valid:
            self.current_episode.invalid_actions += 1
        
        # Track action distribution
        if action not in self.current_episode.action_distribution:
            self.current_episode.action_distribution[action] = 0
        self.current_episode.action_distribution[action] += 1
    
    def record_event(self, event: str, reward: float = 0.0, game_rewards: dict = None):
        """
        Record a game event that occurred.
        
        Args:
            event: Event name (e.g., 'COIN_COLLECTED', 'KILLED_OPPONENT')
            reward: Reward associated with this event
        """
        if self.current_episode is None:
            return
        if not game_rewards:
            reward = self.game_rewards.get(event, 0)
            
        # print(f"Recording event: {event} with reward {reward}")
        self.episode_events.append(event)
        
        # Update reward breakdown
        if event not in self.current_episode.reward_breakdown:
            self.current_episode.reward_breakdown[event] = 0.0
        self.current_episode.reward_breakdown[event] += reward
        self.current_episode.total_reward += reward
        
        # Update specific metrics based on event
        if event == "COIN_COLLECTED":
            self.current_episode.coins_collected += 1
        elif event == "KILLED_OPPONENT":
            self.current_episode.opponents_killed += 1
        elif event == "KILLED_SELF":
            self.current_episode.self_kills += 1
            self.current_episode.deaths_by_bomb += 1
            # Mark that this is a self-kill so GOT_KILLED won't double-count
            self.current_episode.metadata['is_self_kill'] = True
        elif event == "GOT_KILLED":
            # Only count as death by opponent if NOT a self-kill
            if not self.current_episode.metadata.get('is_self_kill', False):
                self.current_episode.deaths_by_opponent += 1
        elif event == "BOMB_DROPPED":
            self.current_episode.bombs_placed += 1
        elif event == "CRATE_DESTROYED":
            self.current_episode.crates_destroyed += 1
        elif event == "BOMB_HIT_OPPONENT":
            self.current_episode.bombs_hit_opponent += 1
        elif event == "BOMB_HIT_SELF":
            self.current_episode.bombs_hit_self += 1
        elif event == "INVALID_ACTION":
            self.current_episode.invalid_actions += 1
    
    def end_episode(self, won: bool, rank: int, survival_steps: int, 
                   total_steps: int, metadata: Dict[str, Any] = None):
        """
        Finalize the current episode and store metrics.
        
        Args:
            won: Whether the agent won this episode
            rank: Final ranking (1=first, 2=second, etc.)
            survival_steps: Number of steps the agent survived
            total_steps: Total episode length
            metadata: Additional metadata to store
        """
        # print("Ending episode and finalizing metrics.")
        if self.current_episode is None:
            return
        
        # Update outcome metrics
        self.current_episode.won = won
        self.current_episode.rank = rank
        self.current_episode.survival_time = survival_steps
        self.current_episode.total_steps = total_steps
        
        if metadata:
            self.current_episode.metadata.update(metadata)
        
        # Calculate derived metrics
        self._compute_derived_metrics()

        # Copy events and actions to episode
        self.current_episode.events = self.episode_events.copy()
        self.current_episode.actions = self.episode_actions.copy()

        # Store episode
        self.episodes.append(self.current_episode)
        self.episode_count += 1

        # Update running statistics
        self._update_running_stats()

        # Clear current episode
        self.current_episode = None
        self.episode_events = []
        self.episode_actions = []
    
    def _compute_derived_metrics(self):
        """Compute derived metrics from raw data."""
        if self.current_episode is None:
            return
        
        ep = self.current_episode
        
        # Bomb effectiveness
        if ep.bombs_placed > 0:
            ep.metadata['bomb_kill_rate'] = ep.bombs_hit_opponent / ep.bombs_placed
            ep.metadata['bomb_crate_rate'] = ep.bombs_destroyed_crate / ep.bombs_placed
            ep.metadata['bomb_self_hit_rate'] = ep.bombs_hit_self / ep.bombs_placed
        
        # Survival rate
        if ep.total_steps > 0:
            ep.metadata['survival_rate'] = ep.survival_time / ep.total_steps
        
        # Action efficiency
        if ep.total_actions > 0:
            ep.metadata['invalid_action_rate'] = ep.invalid_actions / ep.total_actions
        
        # Resource collection rate
        if ep.survival_time > 0:
            ep.metadata['coins_per_step'] = ep.coins_collected / ep.survival_time
            ep.metadata['crates_per_step'] = ep.crates_destroyed / ep.survival_time
    
    def _update_running_stats(self):
        """Update running statistics for quick access."""
        if self.current_episode is None:
            return
        
        ep = self.current_episode
        self.running_stats['win_rate'].append(1.0 if ep.won else 0.0)
        self.running_stats['survival_time'].append(ep.survival_time)
        self.running_stats['total_reward'].append(ep.total_reward)
        self.running_stats['coins_collected'].append(ep.coins_collected)
        self.running_stats['opponents_killed'].append(ep.opponents_killed)
        self.running_stats['self_kills'].append(ep.self_kills)
    
    def get_summary_stats(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary statistics over all or recent episodes.
        
        Args:
            last_n: If provided, only use last N episodes
            
        Returns:
            Dictionary of summary statistics
        """
        episodes = self.episodes[-last_n:] if last_n else self.episodes
        
        if not episodes:
            return {}
        
        # Success metrics
        wins = sum(1 for ep in episodes if ep.won)
        win_rate = wins / len(episodes)
        
        avg_rank = np.mean([ep.rank for ep in episodes])
        avg_survival = np.mean([ep.survival_time for ep in episodes])
        avg_reward = np.mean([ep.total_reward for ep in episodes])
        
        # Combat metrics
        total_kills = sum(ep.opponents_killed for ep in episodes)
        total_self_kills = sum(ep.self_kills for ep in episodes)
        total_deaths_by_opponent = sum(ep.deaths_by_opponent for ep in episodes)
        total_deaths_by_bomb = sum(ep.deaths_by_bomb for ep in episodes)

        # Fix double-counting: agents who die by own bomb get BOTH KILLED_SELF and GOT_KILLED
        # Total deaths = self_kills + deaths_by_opponent (not both, they overlap)
        # deaths_by_bomb is same as self_kills, deaths_by_opponent is from GOT_KILLED
        total_deaths = total_self_kills + total_deaths_by_opponent

        avg_kills = total_kills / len(episodes)
        avg_deaths = total_deaths / len(episodes)
        self_kill_rate = total_self_kills / len(episodes)

        # Survival rate: percentage of episodes where agent did NOT die
        # Agent survived if they didn't self-kill AND didn't get killed by opponent
        survived_episodes = sum(1 for ep in episodes
                               if ep.self_kills == 0 and ep.deaths_by_opponent == 0)
        survival_rate = survived_episodes / len(episodes)

        # Bomb metrics
        total_bombs = sum(ep.bombs_placed for ep in episodes)
        effective_bombs = sum(ep.bombs_hit_opponent for ep in episodes)
        bomb_effectiveness = effective_bombs / total_bombs if total_bombs > 0 else 0.0
        
        # Resource metrics
        avg_coins = np.mean([ep.coins_collected for ep in episodes])
        avg_crates = np.mean([ep.crates_destroyed for ep in episodes])
        
        # Action metrics
        total_actions = sum(ep.total_actions for ep in episodes)
        total_invalid = sum(ep.invalid_actions for ep in episodes)
        invalid_rate = total_invalid / total_actions if total_actions > 0 else 0.0
        
        return {
            'agent_name': self.agent_name,
            'num_episodes': len(episodes),
            
            # Success metrics
            'win_rate': win_rate,
            'win_rate_std': np.std([1.0 if ep.won else 0.0 for ep in episodes]),
            'avg_rank': avg_rank,
            'avg_survival_time': avg_survival,
            'survival_time_std': np.std([ep.survival_time for ep in episodes]),
            
            # Reward metrics
            'avg_reward': avg_reward,
            'reward_std': np.std([ep.total_reward for ep in episodes]),
            'total_reward': sum(ep.total_reward for ep in episodes),
            
            # Combat metrics
            'avg_kills_per_episode': avg_kills,
            'total_kills': total_kills,
            'self_kill_rate': self_kill_rate,
            'total_self_kills': total_self_kills,
            'total_deaths': total_deaths,
            'total_deaths_by_opponent': total_deaths_by_opponent,
            'total_deaths_by_bomb': total_deaths_by_bomb,
            'avg_deaths': avg_deaths,
            'survival_rate': survival_rate,  # Percentage of episodes survived (0.0 to 1.0)
            
            # Bomb metrics
            'bomb_effectiveness': bomb_effectiveness,
            'avg_bombs_per_episode': total_bombs / len(episodes),
            'total_bombs_placed': total_bombs,
            
            # Resource metrics
            'avg_coins_per_episode': avg_coins,
            'avg_crates_per_episode': avg_crates,
            'total_coins': sum(ep.coins_collected for ep in episodes),
            'total_crates': sum(ep.crates_destroyed for ep in episodes),
            
            # Efficiency metrics
            'invalid_action_rate': invalid_rate,
            'avg_actions_per_episode': total_actions / len(episodes),
        }
    
    def get_learning_curve(self, metric: str = 'total_reward', 
                          window_size: int = 10) -> Tuple[List[int], List[float], List[float]]:
        """
        Get learning curve data for a specific metric.
        
        Args:
            metric: Metric to plot ('total_reward', 'survival_time', etc.)
            window_size: Moving average window size
            
        Returns:
            Tuple of (episode_ids, values, moving_average)
        """
        if not self.episodes:
            return [], [], []
        
        episode_ids = [ep.episode_id for ep in self.episodes]
        values = []
        
        for ep in self.episodes:
            if metric == 'total_reward':
                values.append(ep.total_reward)
            elif metric == 'survival_time':
                values.append(ep.survival_time)
            elif metric == 'won':
                values.append(1.0 if ep.won else 0.0)
            elif metric == 'coins_collected':
                values.append(ep.coins_collected)
            elif metric == 'opponents_killed':
                values.append(ep.opponents_killed)
            else:
                values.append(ep.metadata.get(metric, 0.0))
        
        # Compute moving average
        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size + 1)
            window = values[start_idx:i+1]
            moving_avg.append(np.mean(window))
        
        return episode_ids, values, moving_avg
    
    def compare_by_opponent(self) -> Dict[str, Dict[str, float]]:
        """
        Compare performance against different opponent types.
        
        Returns:
            Dictionary mapping opponent type to performance metrics
        """
        opponent_stats = defaultdict(lambda: {'episodes': [], 'wins': 0})
        
        for ep in self.episodes:
            # Use first opponent type as key (or 'mixed' if multiple)
            if not ep.opponent_types:
                opp_key = 'unknown'
            elif len(ep.opponent_types) == 1:
                opp_key = ep.opponent_types[0]
            else:
                opp_key = 'mixed'
            
            opponent_stats[opp_key]['episodes'].append(ep)
            if ep.won:
                opponent_stats[opp_key]['wins'] += 1
        
        # Compute statistics per opponent
        results = {}
        for opp_type, data in opponent_stats.items():
            episodes = data['episodes']
            n = len(episodes)
            
            results[opp_type] = {
                'num_episodes': n,
                'win_rate': data['wins'] / n,
                'avg_survival': np.mean([ep.survival_time for ep in episodes]),
                'avg_reward': np.mean([ep.total_reward for ep in episodes]),
                'avg_kills': np.mean([ep.opponents_killed for ep in episodes]),
            }
        
        return results
    
    def compare_by_map(self) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across different maps.
        
        Returns:
            Dictionary mapping map name to performance metrics
        """
        map_stats = defaultdict(lambda: {'episodes': [], 'wins': 0})
        
        for ep in self.episodes:
            map_stats[ep.map_name]['episodes'].append(ep)
            if ep.won:
                map_stats[ep.map_name]['wins'] += 1
        
        results = {}
        for map_name, data in map_stats.items():
            episodes = data['episodes']
            n = len(episodes)
            
            results[map_name] = {
                'num_episodes': n,
                'win_rate': data['wins'] / n,
                'avg_survival': np.mean([ep.survival_time for ep in episodes]),
                'avg_reward': np.mean([ep.total_reward for ep in episodes]),
            }
        
        return results
    
    def save(self, filename: Optional[str] = None):
        """
        Save metrics to disk.
        
        Args:
            filename: Custom filename (default: agent_name_timestamp.pkl)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.agent_name}_{self.episode_count}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'agent_name': self.agent_name,
                'episodes': [ep.to_dict() for ep in self.episodes],
                'episode_count': self.episode_count,
            }, f)
        
        # print(f"Metrics saved to {filepath}")
        
        # Also save JSON summary
        summary_path = filepath.replace('.pkl', '_summary.json')
        summary = self.get_summary_stats()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load(self, filename: str):
        """
        Load metrics from disk.
        
        Args:
            filename: File to load from
        """
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.agent_name = data['agent_name']
        self.episodes = [EpisodeMetrics.from_dict(ep) for ep in data['episodes']]
        self.episode_count = data['episode_count']
        
        # Rebuild running stats
        for ep in self.episodes:
            self.current_episode = ep
            self._update_running_stats()
        self.current_episode = None
        
        # print(f"Loaded {len(self.episodes)} episodes for agent '{self.agent_name}'")
    
    def export_to_csv(self, filename: Optional[str] = None):
        """
        Export episode data to CSV format.
        
        Args:
            filename: CSV filename (default: agent_name_timestamp.csv)
        """
        import csv
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.agent_name}_{timestamp}.csv"
        
        filepath = os.path.join(self.save_dir, filename)
        
        if not self.episodes:
            print("No episodes to export")
            return
        
        # Get all possible fields
        fieldnames = list(self.episodes[0].to_dict().keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for ep in self.episodes:
                row = ep.to_dict()
                # Convert complex types to strings
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        row[key] = str(value)
                writer.writerow(row)
        
        print(f"Exported {len(self.episodes)} episodes to {filepath}")
