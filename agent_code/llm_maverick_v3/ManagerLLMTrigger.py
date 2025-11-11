"""
Smart LLM Triggering System
Only calls expensive LLM when Maverick is uncertain or in critical situations
"""

import numpy as np
from collections import deque

# Action mappings
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']
OPPOSITE_ACTIONS = {
    'LEFT': 'RIGHT', 'RIGHT': 'LEFT',
    'UP': 'DOWN', 'DOWN': 'UP'
}


def should_trigger_llm(self, game_state, Q_values, top_3_actions, bomb_radius_data=None,
                        coins_collection_data=None, plant_bomb_data=None, opponents_analysis=None):
    """
    Determines if LLM should be consulted based on uncertainty and high-stakes decisions.

    FOCUSED APPROACH: Only triggers on high-impact scenarios
    1. Behavioral loops (shivering/stationary)
    2. Low Q-value confidence (uncertainty)
    3. Bomb planting decisions (high-stakes strategic choice)

    Args:
        self: Agent object with history tracking
        game_state: Current game state dict
        Q_values: Array of Q-values for all 6 actions
        top_3_actions: List of dicts with top 3 actions and their Q-values
        bomb_radius_data: (Optional) Dict with bomb danger information
        coins_collection_data: (Optional) Dict with coin collection strategy
        plant_bomb_data: (Optional) Dict with bomb planting evaluation - REQUIRED for bomb trigger
        opponents_analysis: (Optional) List of opponent states

    Returns:
        (bool, str): (should_trigger, reason)
    """

    # Initialize history if not exists
    if not hasattr(self, 'position_history'):
        self.position_history = deque(maxlen=5)
    if not hasattr(self, 'action_history'):
        self.action_history = deque(maxlen=5)
    if not hasattr(self, 'llm_call_count'):
        self.llm_call_count = 0
    if not hasattr(self, 'total_steps'):
        self.total_steps = 0

    self.total_steps += 1
    current_pos = game_state['self'][3] if game_state and 'self' in game_state else None

    # Track current position
    if current_pos:
        self.position_history.append(current_pos)

    # ============================================================
    # PRIMARY TRIGGER 0: Imminent Danger (DISABLED)
    # ============================================================
    # DISABLED: Testing shows this trigger decreased survival (136→73 steps)
    # Root cause: Either LLM making bad escape decisions OR Maverick already
    # handles danger well from training. Need to investigate LLM prompt/logic.
    #
    # is_in_danger, danger_reason = detect_imminent_danger(bomb_radius_data)
    # if is_in_danger:
    #     self.llm_call_count += 1
    #     return True, f"⚠️ DANGER: {danger_reason}"

    # ============================================================
    # PRIMARY TRIGGER 1: Behavioral Loops (Stuck/Shivering)
    # ============================================================
    is_shivering, shiver_reason = detect_shivering(self.action_history)
    if is_shivering:
        self.llm_call_count += 1
        return True, f"🔄 Behavioral loop: {shiver_reason}"

    is_stationary, stationary_reason = detect_stationary(self.position_history)
    if is_stationary:
        self.llm_call_count += 1
        return True, f"🛑 Stationary: {stationary_reason}"

    # ============================================================
    # PRIMARY TRIGGER 2: Low Q-value Confidence (Uncertainty)
    # ============================================================
    is_uncertain, uncertainty_reason = detect_uncertainty(Q_values, top_3_actions)
    if is_uncertain:
        self.llm_call_count += 1
        return True, f"❓ Uncertainty: {uncertainty_reason}"

    # ============================================================
    # PRIMARY TRIGGER 3: Bomb Planting Decision
    # ============================================================
    wants_bomb, bomb_reason = detect_bomb_intent(top_3_actions, plant_bomb_data, game_state)
    if wants_bomb:
        self.llm_call_count += 1
        return True, f"💣 Bomb decision: {bomb_reason}"

    # LLM not needed - Maverick is confident and moving normally
    return False, "✓ Maverick confident"


def detect_shivering(action_history):
    """
    Detects if agent is oscillating between opposite directions.
    Returns: (is_shivering, reason)
    """
    if len(action_history) < 4:
        return False, None

    recent_actions = list(action_history)[-4:]

    # Check for LEFT-RIGHT-LEFT-RIGHT pattern
    if (recent_actions[0] in OPPOSITE_ACTIONS and
        recent_actions[1] == OPPOSITE_ACTIONS[recent_actions[0]] and
        recent_actions[2] == recent_actions[0] and
        recent_actions[3] == recent_actions[1]):
        return True, f"Oscillating {recent_actions[0]}-{recent_actions[1]}"

    # Check for 3-step oscillation
    if len(action_history) >= 3:
        last_3 = list(action_history)[-3:]
        if (last_3[0] in OPPOSITE_ACTIONS and
            last_3[1] == OPPOSITE_ACTIONS[last_3[0]] and
            last_3[2] == last_3[0]):
            return True, f"Quick oscillation {last_3[0]}-{last_3[1]}-{last_3[0]}"

    return False, None


def detect_stationary(position_history):
    """
    Detects if agent is stuck in same position or very small area.
    Returns: (is_stationary, reason)
    """
    if len(position_history) < 4:
        return False, None

    recent_positions = list(position_history)[-4:]

    # Same position for 4 consecutive steps
    if all(pos == recent_positions[0] for pos in recent_positions):
        return True, f"Stuck at position {recent_positions[0]}"

    # Moving in very small 2x2 area (stuck in corner/loop)
    if len(position_history) >= 5:
        last_5 = list(position_history)[-5:]
        unique_positions = set(last_5)
        if len(unique_positions) <= 2:
            return True, f"Looping in small area: {unique_positions}"

    return False, None


def detect_imminent_danger(bomb_radius_data):
    """
    Detects if agent is in immediate life-threatening danger.

    This is the HIGHEST PRIORITY trigger - survival trumps all other objectives.
    LLM should be consulted when agent needs to escape bombs, even if Maverick
    is confident about a different action.

    Args:
        bomb_radius_data: Dict from check_bomb_radius_and_escape helper
            Expected format: {
                'in_danger': 'yes'/'no',
                'escape_bomb_action': 'UP'/'DOWN'/'LEFT'/'RIGHT'/'WAIT'/'none'
            }

    Returns: (is_in_danger, reason)
    """
    if not bomb_radius_data:
        return False, None

    in_danger = bomb_radius_data.get('in_danger') == 'yes'
    escape_action = bomb_radius_data.get('escape_bomb_action', 'none')

    # CRITICAL: Agent is in bomb blast radius
    if in_danger:
        if escape_action and escape_action != 'none' and escape_action != 'WAIT':
            # There's a clear escape route
            return True, f"In blast radius, must escape: {escape_action}"
        elif escape_action == 'none':
            # Trapped with no obvious escape - LLM needs to find creative solution
            return True, "Trapped in blast radius, no clear escape route"
        else:
            # In danger but helper says WAIT (might be safest option?)
            return True, "In blast radius, need to evaluate safety"

    return False, None


def detect_uncertainty(Q_values, top_3_actions):
    """
    Detects if Maverick is uncertain based on Q-value spread.

    Three clear indicators:
    1. Top 2 Q-values very close (< 0.5 difference)
    2. Top action has low probability (< 35%)
    3. All Q-values near zero (< 0.3) - no strong preference

    Returns: (is_uncertain, reason)
    """
    if len(top_3_actions) < 2:
        return False, None

    q1 = top_3_actions[0]['q_value']
    q2 = top_3_actions[1]['q_value']
    prob1 = top_3_actions[0]['probability']

    # CRITERION 1: Top 2 Q-values very close
    # When Maverick can't decide between two options
    q_diff = abs(q1 - q2)
    if q_diff < 0.5:
        return True, f"Q-values too close: Δ={q_diff:.2f} ({q1:.2f} vs {q2:.2f})"

    # CRITERION 2: Low action probability
    # Softmax probability < 35% means high entropy/uncertainty
    if prob1 < 0.35:
        return True, f"Low confidence: p={prob1:.2%} for best action"

    # CRITERION 3: Weak Q-values overall
    # All options look equally bad → need strategic thinking
    if abs(q1) < 0.3:
        return True, f"Weak preferences: best Q={q1:.2f}"

    return False, None


def detect_contradiction(best_action, bomb_radius_data, current_pos, game_state):
    """
    Detects if Maverick's suggestion contradicts safety features.
    Returns: (is_contradictory, reason)
    """
    # Suggesting WAIT while in danger
    in_danger = bomb_radius_data.get('in_danger') == 'yes'
    if in_danger and best_action == 'WAIT':
        return True, "Suggesting WAIT while in bomb danger"

    # Suggesting movement into explosion
    if best_action in ['LEFT', 'RIGHT', 'UP', 'DOWN']:
        escape_action = bomb_radius_data.get('escape_action')
        if escape_action and escape_action != 'none' and best_action != escape_action:
            # Maverick suggests different direction than escape route
            return True, f"Suggesting {best_action} but escape is {escape_action}"

    # Suggesting BOMB when we don't have one
    if best_action == 'BOMB' and game_state:
        has_bomb = game_state['self'][2]  # self[2] is bomb availability
        if not has_bomb:
            return True, "Suggesting BOMB but no bomb available"

    return False, None


def detect_bomb_intent(top_3_actions, plant_bomb_data, game_state=None):
    """
    Detects if Maverick wants to plant a bomb.

    Bomb planting is a high-stakes strategic decision that benefits from LLM reasoning:
    - Risk vs reward (crates/opponents vs safety)
    - Escape route planning
    - Timing with respect to opponents

    Args:
        top_3_actions: Top 3 Maverick recommendations
        plant_bomb_data: Strategic bomb evaluation from helper
        game_state: (Optional) Current game state to check bomb availability

    Returns: (wants_bomb, reason)
    """
    if not top_3_actions or len(top_3_actions) == 0:
        return False, None

    # Check if BOMB is in top 3 actions
    top_3_action_names = [action['action'] for action in top_3_actions]

    # CRITERION 1: BOMB is the top choice
    if top_3_actions[0]['action'] == 'BOMB':
        # FIRST: Check if agent even has a bomb available
        if game_state:
            self_info = game_state.get('self', [None, None, False])
            has_bomb = self_info[2] if len(self_info) >= 3 else False

            if not has_bomb:
                # No bomb available → Don't waste LLM call
                return False, None

        # Agent has bomb OR we don't know → Continue evaluation
        # Check if it's a strategic opportunity (from plant_bomb_data)
        if plant_bomb_data and plant_bomb_data.get('plant') == 'true':
            current_status = plant_bomb_data.get('current_status', {})

            # Handle both dict and other types (defensive programming)
            expected_crates = 0
            opponents_in_range = 0

            if isinstance(current_status, dict):
                expected_crates = current_status.get('expected_crate_destruction', 0)
                opponents_in_range = current_status.get('opponents_in_range', 0)

            # High-value target → definitely consult LLM
            if expected_crates >= 2 or opponents_in_range > 0:
                return True, f"High-value bomb: {expected_crates} crates, {opponents_in_range} opponents"

            # Any bomb planting is strategic
            return True, f"Maverick suggests bomb (Q={top_3_actions[0]['q_value']:.2f})"

        # BOMB is top choice but plant_bomb_data says it's not safe/good
        elif plant_bomb_data and plant_bomb_data.get('plant') != 'true':
            # Maverick wants to bomb but helper features say no → contradiction, needs LLM
            reason = plant_bomb_data.get('reason', 'Unknown reason')
            return True, f"Risky bomb (Maverick vs helper): {reason}"

        # No plant_bomb_data available but BOMB is top choice
        else:
            return True, f"Bomb is top action (Q={top_3_actions[0]['q_value']:.2f})"

    # CRITERION 2: BOMB is in top 3 (but not top 1) and looks promising
    if 'BOMB' in top_3_action_names:
        bomb_idx = top_3_action_names.index('BOMB')
        bomb_q = top_3_actions[bomb_idx]['q_value']
        top_q = top_3_actions[0]['q_value']

        # If BOMB is competitive with top action (within 0.7)
        if abs(bomb_q - top_q) < 0.7:
            return True, f"BOMB competitive: Q={bomb_q:.2f} vs top {top_q:.2f}"

    return False, None


def log_llm_trigger_stats(self, logger=None):
    """
    Logs statistics about LLM trigger rate.
    Call this periodically to monitor performance.
    """
    if not hasattr(self, 'llm_call_count') or not hasattr(self, 'total_steps'):
        return

    if self.total_steps == 0:
        return

    trigger_rate = (self.llm_call_count / self.total_steps) * 100

    msg = (f"LLM Trigger Stats: {self.llm_call_count}/{self.total_steps} steps "
           f"({trigger_rate:.1f}% trigger rate)")

    if logger:
        logger.info(msg)
    else:
        print(msg)

    return trigger_rate


def reset_trigger_stats(self):
    """Reset statistics for new episode."""
    self.llm_call_count = 0
    self.total_steps = 0
    if hasattr(self, 'position_history'):
        self.position_history.clear()
    if hasattr(self, 'action_history'):
        self.action_history.clear()
