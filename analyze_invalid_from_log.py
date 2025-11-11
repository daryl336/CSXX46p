#!/usr/bin/env python3
"""
Analyze invalid actions from game log.
"""

import re
from collections import Counter

log_file = "logs/game.log"

print("=" * 80)
print("INVALID ACTIONS ANALYSIS FROM GAME LOG")
print("=" * 80)
print()

# Read last 10000 lines (recent gameplay)
with open(log_file, 'r') as f:
    lines = f.readlines()[-10000:]

print(f"Analyzing last {len(lines)} log lines...")
print()

# Find all llm_maverick_v3 actions
llm_actions = []
invalid_actions = []
invalid_contexts = []

for i, line in enumerate(lines):
    # Look for llm_maverick_v3 taking actions
    if 'llm_maverick_v3' in line.lower() or 'LLM Maverick' in line:
        # Check if this is an invalid action line
        if 'INVALID_ACTION' in line:
            # Extract action if possible
            match = re.search(r"action[:\s]+([A-Z]+)", line, re.IGNORECASE)
            if match:
                action = match.group(1).upper()
                invalid_actions.append(action)

                # Get context (previous and next few lines)
                context_start = max(0, i-3)
                context_end = min(len(lines), i+2)
                context = ''.join(lines[context_start:context_end])
                invalid_contexts.append({
                    'action': action,
                    'context': context,
                    'line_num': i
                })

print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

if invalid_actions:
    print(f"Found {len(invalid_actions)} invalid actions")
    print()

    print("Invalid action breakdown:")
    action_counts = Counter(invalid_actions)
    for action, count in action_counts.most_common():
        print(f"  {action}: {count} times ({count/len(invalid_actions)*100:.1f}%)")
    print()

    print("=" * 80)
    print("SAMPLE INVALID ACTION CONTEXTS (first 3)")
    print("=" * 80)
    for i, ctx in enumerate(invalid_contexts[:3], 1):
        print(f"\nInvalid Action #{i}: {ctx['action']}")
        print("-" * 80)
        print(ctx['context'])
else:
    print("✅ No invalid actions found in recent logs!")
    print()
    print("This could mean:")
    print("  1. llm_maverick_v3 hasn't been run recently")
    print("  2. Recent runs had no invalid actions (ideal!)")
    print("  3. Logs don't contain detailed action info")

print()
print("=" * 80)
