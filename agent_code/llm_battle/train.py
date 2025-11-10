"""
Training callbacks for LLM Battle agent - for evaluation metrics tracking only.

The LLM Battle agent doesn't learn from rewards (it uses Claude API for decisions),
but we track metrics for evaluation and comparison with other agents.
"""

import logging
import events as e
from metrics.metrics_tracker import MetricsTracker


def setup_training(self):
    """
    Setup for LLM Battle agent evaluation tracking.

    LLM doesn't train/learn, but we track metrics for evaluation.
    """
    self.name = "LLM Battle"

    # Setup logger
    if not hasattr(self, 'logger'):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # Initialize metrics tracker for evaluation
    self.metrics_tracker = MetricsTracker(
        agent_name=self.name,
        save_dir="evaluation_metrics"
    )

    self.logger.info("LLM Battle evaluation tracking initialized.")


def game_events_occurred(self, old_game_state: dict, self_action: str,
                         new_game_state: dict, events: list):
    """
    Track events for metrics (LLM doesn't learn from them).

    Called after each game step to record what happened.
    """
    # Start episode tracking on first step
    if old_game_state and old_game_state.get("step", 0) == 1:
        opponent_names = []
        if "others" in old_game_state:
            for other in old_game_state["others"]:
                if other is not None:
                    opponent_names.append(other[0] if isinstance(other, tuple) else str(other))

        self.metrics_tracker.start_episode(
            episode_id=old_game_state.get("round", 0),
            opponent_types=opponent_names,
            scenario="evaluation"
        )

    # Record action
    if hasattr(self, 'metrics_tracker') and self.metrics_tracker.current_episode:
        self.metrics_tracker.record_action(self_action, is_valid=True)

        # Record events with evaluation rewards
        from evaluation_rewards import EVALUATION_REWARDS

        for event in events:
            event_name = event if isinstance(event, str) else str(event)
            reward = EVALUATION_REWARDS.get(event, 0.0)
            self.metrics_tracker.record_event(event_name, reward=reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each round.

    Finalize episode metrics and save.
    """
    if not hasattr(self, 'metrics_tracker'):
        return

    # Record final events
    if self.metrics_tracker.current_episode:
        from evaluation_rewards import EVALUATION_REWARDS

        for event in events:
            event_name = event if isinstance(event, str) else str(event)
            reward = EVALUATION_REWARDS.get(event, 0.0)
            self.metrics_tracker.record_event(event_name, reward=reward)

        # Determine if won (first place)
        won = e.SURVIVED_ROUND in events
        rank = 1 if won else 2  # Simplified - adjust based on actual game state

        # Get survival time
        survival_steps = last_game_state.get('step', 0)

        # End episode
        self.metrics_tracker.end_episode(
            won=won,
            rank=rank,
            survival_steps=survival_steps,
            total_steps=last_game_state.get('step', 0)
        )

        # Save metrics after every episode
        self.metrics_tracker.save()

    self.logger.info(f"Round {last_game_state.get('round', 0)} ended.")


def reward_from_events(self, events: list) -> float:
    """
    Calculate reward from events using evaluation rewards.

    Note: LLM doesn't use this for learning, only for metrics tracking.
    """
    from evaluation_rewards import get_reward_from_events
    return get_reward_from_events(events)
