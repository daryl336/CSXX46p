#!/usr/bin/env python3
"""
Quick test to verify metrics tracking is working correctly.
Run this before starting your training to ensure no double counting.
"""

import sys
from agent_code.ppo.callbacks import setup, act
from agent_code.ppo.train import setup_training, game_events_occurred, end_of_round
from metrics.metrics_tracker import MetricsTracker


class MockSelf:
    """Mock agent object for testing."""
    def __init__(self):
        self.logger = type('obj', (object,), {
            'info': lambda x: print(f"[INFO] {x}"),
            'debug': lambda x: print(f"[DEBUG] {x}"),
            'warning': lambda x: print(f"[WARNING] {x}"),
            'error': lambda x: print(f"[ERROR] {x}"),
        })()


def test_training_mode():
    """Test that training mode doesn't double-count."""
    print("\n" + "="*70)
    print("TEST 1: Training Mode - Checking for Double Counting")
    print("="*70)

    agent = MockSelf()

    # Setup
    setup(agent)
    setup_training(agent)

    # Simulate episode start (step 1)
    old_game_state = {
        'step': 1,
        'round': 1,
        'field': [[0]*17 for _ in range(17)],
        'self': ('agent', 0, True, (1, 1)),
        'others': [('opponent1', 0, True, (15, 15))],
        'coins': [(5, 5)],
        'bombs': [],
        'explosion_map': [[0]*17 for _ in range(17)],
    }

    # Call act() - this should NOT start episode (train.py will)
    action = act(agent, old_game_state)
    print(f"✓ act() called, returned action: {action}")

    # Call game_events_occurred() - this SHOULD start episode
    new_game_state = old_game_state.copy()
    new_game_state['step'] = 2
    game_events_occurred(agent, old_game_state, action, new_game_state, ['MOVED_RIGHT'])

    # Check if episode was started exactly once
    if agent.metrics_tracker.current_episode:
        print(f"✓ Episode started: ID={agent.metrics_tracker.current_episode.episode_id}")

        # Check event count
        event_count = len(agent.metrics_tracker.episode_events)
        print(f"✓ Events recorded: {event_count}")

        if event_count == 0:
            print("❌ WARNING: No events recorded! Check if MOVED_RIGHT is in GAME_REWARDS")
        elif event_count > 10:
            print("❌ WARNING: Too many events! Possible double counting!")
        else:
            print("✓ Event count looks reasonable")
    else:
        print("❌ ERROR: Episode not started!")

    # Simulate a few more steps
    for step in range(3, 6):
        old_state = new_game_state.copy()
        new_game_state = old_state.copy()
        new_game_state['step'] = step

        action = act(agent, old_state)
        game_events_occurred(agent, old_state, action, new_game_state, [])

    # Simulate coin collection
    old_state = new_game_state.copy()
    new_game_state = old_state.copy()
    new_game_state['step'] = 6
    action = act(agent, old_state)
    game_events_occurred(agent, old_state, action, new_game_state, ['COIN_COLLECTED'])

    coin_events = sum(1 for e in agent.metrics_tracker.episode_events if e == 'COIN_COLLECTED')
    print(f"\n✓ Simulated coin collection: {coin_events} COIN_COLLECTED event(s)")

    if coin_events == 1:
        print("✓ PASS: No double counting!")
    else:
        print(f"❌ FAIL: Expected 1 COIN_COLLECTED, got {coin_events}")

    # End episode
    end_of_round(agent, new_game_state, action, ['SURVIVED_ROUND'])

    survived_events = sum(1 for e in agent.metrics_tracker.episode_events if e == 'SURVIVED_ROUND')
    print(f"✓ End of round: {survived_events} SURVIVED_ROUND event(s)")

    if survived_events == 1:
        print("✓ PASS: End events not double-counted!")
    else:
        print(f"❌ FAIL: Expected 1 SURVIVED_ROUND, got {survived_events}")

    # Check episode ended
    if agent.metrics_tracker.current_episode is None:
        print("✓ Episode properly ended")
    else:
        print("❌ ERROR: Episode not ended!")

    # Check metrics
    if len(agent.metrics_tracker.episodes) == 1:
        ep = agent.metrics_tracker.episodes[0]
        print(f"\n✓ Episode recorded:")
        print(f"  - Total actions: {ep.total_actions}")
        print(f"  - Coins collected: {ep.coins_collected}")
        print(f"  - Total reward: {ep.total_reward:.2f}")
        print(f"  - Events breakdown: {ep.reward_breakdown}")

        if ep.coins_collected == 1:
            print("✓ PASS: Correct coin count!")
        else:
            print(f"❌ FAIL: Expected 1 coin, got {ep.coins_collected}")
    else:
        print(f"❌ ERROR: Expected 1 episode, got {len(agent.metrics_tracker.episodes)}")


def test_gameplay_mode():
    """Test that gameplay mode works correctly."""
    print("\n" + "="*70)
    print("TEST 2: Gameplay Mode - Checking Episode Tracking")
    print("="*70)

    agent = MockSelf()

    # Setup (no train setup - gameplay only)
    setup(agent)

    # Simulate gameplay
    game_state = {
        'step': 1,
        'round': 1,
        'field': [[0]*17 for _ in range(17)],
        'self': ('agent', 0, True, (1, 1)),
        'others': [],
        'coins': [],
        'bombs': [],
        'explosion_map': [[0]*17 for _ in range(17)],
    }

    # First action should start episode
    action = act(agent, game_state)

    if agent.metrics_tracker.current_episode:
        print("✓ Episode started in gameplay mode")
    else:
        print("❌ ERROR: Episode not started in gameplay mode!")

    # Simulate a few more steps
    for step in range(2, 10):
        game_state['step'] = step
        action = act(agent, game_state)

    print(f"✓ Actions recorded: {agent.metrics_tracker.current_episode.total_actions}")

    # End round
    from agent_code.ppo.callbacks import end_of_round as gameplay_end
    gameplay_end(agent, game_state, action, ['SURVIVED_ROUND'])

    if agent.metrics_tracker.current_episode is None:
        print("✓ Episode properly ended in gameplay mode")
    else:
        print("❌ ERROR: Episode not ended in gameplay mode!")


if __name__ == "__main__":
    try:
        test_training_mode()
        test_gameplay_mode()

        print("\n" + "="*70)
        print("TESTS COMPLETED")
        print("="*70)
        print("\nIf you see ✓ PASS messages, your metrics tracking is working correctly!")
        print("If you see ❌ FAIL messages, there may still be issues.")

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
