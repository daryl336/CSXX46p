"""
UNIVERSAL EVALUATION REWARDS
=============================
Standardized reward structure for fair evaluation across all agents.

Use this module to ensure all agents are evaluated with identical reward signals.
This eliminates reward-shaping bias and allows direct performance comparison.

Author: Generated for CS5446 AI Planning Project Evaluation
Date: 2025-11-09
"""

import events as e

# =============================================================================
# UNIVERSAL EVALUATION REWARD MAPPING
# =============================================================================
# Scale: -10 to +10 (balanced, interpretable, presentation-friendly)

EVALUATION_REWARDS = {
    # -------------------------------------------------------------------------
    # PRIMARY OBJECTIVES (Highest rewards)
    # -------------------------------------------------------------------------
    e.COIN_COLLECTED: 10.0,          # Main goal - highest positive reward
    e.KILLED_OPPONENT: 8.0,          # Combat success - secondary objective
    e.SURVIVED_ROUND: 5.0,           # Survival bonus - essential for success

    # -------------------------------------------------------------------------
    # SECONDARY OBJECTIVES
    # -------------------------------------------------------------------------
    e.CRATE_DESTROYED: 1.0,          # Environmental interaction - minor credit

    # -------------------------------------------------------------------------
    # CRITICAL PENALTIES (Symmetric with rewards)
    # -------------------------------------------------------------------------
    e.KILLED_SELF: -10.0,            # Suicide - worst outcome (matches coin value)
    e.GOT_KILLED: -5.0,              # Death by opponent (matches survival value)

    # -------------------------------------------------------------------------
    # MINOR PENALTIES
    # -------------------------------------------------------------------------
    e.INVALID_ACTION: -1.0,          # Discourage mistakes/invalid moves
    e.WAITED: -0.1,                  # Small efficiency cost for waiting

    # -------------------------------------------------------------------------
    # NEUTRAL EVENTS (Outcome-based only)
    # -------------------------------------------------------------------------
    e.BOMB_DROPPED: 0.0,             # Neutral - only outcomes matter
    e.BOMB_EXPLODED: 0.0,            # Neutral - only outcomes matter
    e.MOVED_LEFT: 0.0,               # Neutral - no directional bias
    e.MOVED_RIGHT: 0.0,              # Neutral - no directional bias
    e.MOVED_UP: 0.0,                 # Neutral - no directional bias
    e.MOVED_DOWN: 0.0,               # Neutral - no directional bias
}


# =============================================================================
# REWARD CALCULATION FUNCTION
# =============================================================================

def get_reward_from_events(events: list) -> float:
    """
    Calculate total reward from a list of game events.

    Args:
        events: List of event constants from events.py

    Returns:
        Total reward as a float

    Example:
        >>> events = [e.COIN_COLLECTED, e.MOVED_RIGHT, e.WAITED]
        >>> reward = get_reward_from_events(events)
        >>> # reward = 10.0 + 0.0 + (-0.1) = 9.9
    """
    reward_sum = 0.0
    for event in events:
        reward_sum += EVALUATION_REWARDS.get(event, 0.0)
    return reward_sum


# =============================================================================
# NORMALIZED VERSION (for PPO or agents requiring [-1, 1] range)
# =============================================================================

EVALUATION_REWARDS_NORMALIZED = {
    event: reward / 10.0
    for event, reward in EVALUATION_REWARDS.items()
}


def get_reward_from_events_normalized(events: list) -> float:
    """
    Calculate normalized reward (scaled to [-1, 1] range).

    Args:
        events: List of event constants from events.py

    Returns:
        Total reward normalized to [-1, 1] range
    """
    reward_sum = 0.0
    for event in events:
        reward_sum += EVALUATION_REWARDS_NORMALIZED.get(event, 0.0)
    return reward_sum


# =============================================================================
# DOCUMENTATION & RATIONALE
# =============================================================================

REWARD_RATIONALE = """
DESIGN PRINCIPLES:
------------------
1. BALANCED SCALE: [-10, +10] range is moderate - not too extreme, not too flat
2. SYMMETRIC RISK/REWARD: Coin (+10) = Suicide (-10), Survival (+5) = Death (-5)
3. CLEAR HIERARCHY: Coins (10) > Combat (8) > Survival (5) > Crates (1)
4. NO AGENT BIAS: Uses only standard game events, no custom/shaped events
5. INTERPRETABLE: Easy to explain and visualize in presentations

EXPECTED BEHAVIORS:
-------------------
✓ Prioritize coin collection (highest reward)
✓ Avoid suicide (equal penalty to coin value)
✓ Value survival (can't collect coins if dead)
✓ Engage in combat when advantageous
✓ Destroy crates opportunistically
✓ Minimize unnecessary waiting

COMPARISON TO AGENT-SPECIFIC REWARDS:
--------------------------------------
- Q-Learning: Uses dense shaping (MOVED_TOWARDS_COIN, etc.) - NOT USED HERE
- PPO: Uses normalized [-1,1] with shaped rewards - OPTIONAL NORMALIZATION PROVIDED
- DQN Final: Uses sophisticated policy shaping - NOT USED HERE
- Maverick: Uses extreme scale (-1000 to 500) - MODERATED HERE
- LLM: No explicit rewards - NOW HAS CLEAR SIGNAL

This standardization ensures FAIR COMPARISON across all architectures.
"""


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("UNIVERSAL EVALUATION REWARDS - Quick Reference")
    print("=" * 70)
    print("\nREWARD TABLE:")
    print("-" * 70)
    print(f"{'Event':<30} {'Reward':>10} {'Normalized':>15}")
    print("-" * 70)

    # Sort by reward value (descending)
    sorted_rewards = sorted(EVALUATION_REWARDS.items(),
                           key=lambda x: x[1],
                           reverse=True)

    for event, reward in sorted_rewards:
        if reward != 0.0:  # Only show non-zero rewards
            normalized = EVALUATION_REWARDS_NORMALIZED[event]
            print(f"{str(event):<30} {reward:>10.1f} {normalized:>15.2f}")

    print("-" * 70)
    print("\nEXAMPLE CALCULATIONS:")
    print("-" * 70)

    # Example 1: Successful coin collection
    example1 = [e.COIN_COLLECTED, e.MOVED_RIGHT]
    reward1 = get_reward_from_events(example1)
    print(f"Scenario 1: Collected coin + moved")
    print(f"  Events: {[str(ev) for ev in example1]}")
    print(f"  Reward: {reward1:.1f}")

    # Example 2: Suicide
    example2 = [e.BOMB_DROPPED, e.KILLED_SELF]
    reward2 = get_reward_from_events(example2)
    print(f"\nScenario 2: Dropped bomb + suicide")
    print(f"  Events: {[str(ev) for ev in example2]}")
    print(f"  Reward: {reward2:.1f}")

    # Example 3: Successful combat
    example3 = [e.KILLED_OPPONENT, e.SURVIVED_ROUND]
    reward3 = get_reward_from_events(example3)
    print(f"\nScenario 3: Killed opponent + survived")
    print(f"  Events: {[str(ev) for ev in example3]}")
    print(f"  Reward: {reward3:.1f}")

    print("\n" + "=" * 70)
    print("READY FOR EVALUATION!")
    print("=" * 70)
