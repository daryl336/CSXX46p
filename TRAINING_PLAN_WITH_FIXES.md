# Complete Training Plan with All Fixes

## Overview

Since the network architecture changed (6 → 9 global features), we need to **retrain from scratch** using curriculum learning.

**Training Phases:**
1. **Phase 1:** Coin collection + Survival (10k rounds) - **START HERE**
2. **Phase 2:** More coins (optional, 5k rounds)
3. **Phase 3:** Crate destruction (10-15k rounds)
4. **Phase 4:** 1v1 combat (10k rounds)
5. **Phase 5:** 4-player competition (15k rounds)

---

## Phase 1: Coin Collection & Survival (START HERE)

### Goal
Learn basic navigation, survival, and coin collection **without crates**.

### Configuration

**Already set up for you:**
- ✅ `settings.py`: `CRATE_DENSITY = 0.0` (no crates)
- ✅ `train.py`: Phase 1 rewards configured

**Current rewards:**
```python
COIN_COLLECTED: 150.0      # Primary goal
SURVIVED_ROUND: 200.0      # Important
BOMB_DROPPED: -5.0         # Discourage (no crates)
KILLED_SELF: -400.0        # Strong penalty
APPROACHING_COIN: 8.0      # Shaped reward
STEP_ALIVE: 0.2            # Basic survival
```

### Training Command

```bash
# Already running! Just let it continue
python -m main play --agents ppo --train 1 --n-round 10000 --no-gui
```

### Expected Results

| Milestone | Coins/Game | Survival | Bombs/Game | Avg Reward |
|-----------|------------|----------|------------|------------|
| **Round 1k** | 0.5-1.0 | 60-70% | 0.5-1.0 | -50 to 0 |
| **Round 3k** | 2.0-3.0 | 80-90% | 0.2-0.5 | +50 to +100 |
| **Round 5k** | 3.0-4.0 | 90-95% | 0.1-0.3 | +100 to +150 |
| **Round 10k** | 4.0-5.0 | 95-99% | 0.1-0.2 | +150 to +250 |

**Success criteria for Phase 1:**
- ✅ Coins collected: ≥4.0 per game
- ✅ Survival rate: ≥95%
- ✅ Self-kill rate: <5%
- ✅ Avg reward: ≥+150

### When to Check
- **Round 3000:** Should see 2+ coins/game, 80%+ survival
- **Round 5000:** Should see 3+ coins/game, 90%+ survival
- **Round 10000:** Should meet success criteria

### After Phase 1
```bash
# Save Phase 1 checkpoint
cp agent_code/ppo/models/ppo_agent.pth agent_code/ppo/models/ppo_agent_phase1_final.pth
```

---

## Phase 2: More Coins (Optional)

### Goal
Improve coin collection to 5-6 coins/game before introducing crates.

### Configuration
```bash
# Keep CRATE_DENSITY = 0.0
# Adjust rewards to be more aggressive about coins
```

**Optional step** - only if Phase 1 results are < 4 coins/game.

---

## Phase 3: Crate Destruction

### Goal
Learn strategic bombing and crate destruction while maintaining survival skills.

### Configuration Changes Needed

**1. Update settings.py:**
```python
CRATE_DENSITY = 0.75  # Add crates
```

**2. Update train.py rewards:**
```python
GAME_REWARDS = {
    "COIN_COLLECTED": 100.0,         # Maintain skill
    "CRATE_DESTROYED": 90.0,         # NEW PRIMARY GOAL
    "BOMB_DROPPED": 0.0,             # Neutral - reward outcomes
    "KILLED_SELF": -600.0,           # Strong penalty
    "SURVIVED_ROUND": 400.0,         # Still very important
}

# Phase 3 shaped rewards
STEP_ALIVE_REWARD = 0.5
DANGER_PENALTY = -2.0
ESCAPED_DANGER_REWARD = 35.0
SAFE_BOMB_REWARD = 70.0
BOMB_STAY_PENALTY = -30.0
MOVING_TO_SAFETY_REWARD = 20.0
CRATE_IN_RANGE_REWARD = 18.0
APPROACHING_COIN_REWARD = 6.0
```

**3. Training command:**
```bash
python -m main play --agents ppo --train 1 --n-round 15000 --no-gui
```

### Expected Results

| Milestone | Crates/Game | Survival | Coins/Game | Avg Reward |
|-----------|-------------|----------|------------|------------|
| **Round 3k** | 2-4 | 75-85% | 1-2 | +20 to +80 |
| **Round 6k** | 4-6 | 80-90% | 2-3 | +80 to +150 |
| **Round 10k** | 6-10 | 85-92% | 2-4 | +150 to +250 |
| **Round 15k** | 8-12 | 90-95% | 3-5 | +200 to +350 |

**Success criteria for Phase 3:**
- ✅ Crates destroyed: ≥8 per game
- ✅ Survival rate: ≥85%
- ✅ Self-kill rate: <20%
- ✅ Avg reward: ≥+200

### After Phase 3
```bash
# Save Phase 3 checkpoint
cp agent_code/ppo/models/ppo_agent.pth agent_code/ppo/models/ppo_agent_phase3_final.pth
```

---

## Phase 4: 1v1 Combat (Future)

### Goal
Learn to compete against one opponent.

### Configuration
```bash
# Keep CRATE_DENSITY = 0.75
# Train against rule_based_agent opponent
python -m main play --agents ppo rule_based_agent --train 1 --n-round 10000 --no-gui
```

---

## Phase 5: 4-Player Competition (Future)

### Goal
Learn to compete in full 4-player games.

### Configuration
```bash
# Train against 3 opponents
python -m main play --agents ppo rule_based_agent rule_based_agent rule_based_agent --train 1 --n-round 15000 --no-gui
```

---

## Current Status

✅ **Phase 1 is currently running!**

The training just started with:
- ✅ New architecture (9 global features with escape info)
- ✅ Fixed action masking (safe bomb dropping)
- ✅ Phase 1 rewards (coin collection)
- ✅ No crates (CRATE_DENSITY = 0.0)

---

## What You Should Do Now

### 1. Let Phase 1 Continue Training

Let the current training run until round 10,000.

### 2. Monitor at Round 3000

Check these metrics:
```json
{
  "avg_coins_per_episode": ?,
  "avg_survival_time": ?,
  "self_kill_rate": ?,
  "avg_reward": ?,
  "avg_bombs_per_episode": ?
}
```

**Expected at round 3000:**
- Coins: 2-3 per game
- Survival: 320+ steps (80%+)
- Self-kill: <10%
- Reward: +50 to +100
- Bombs: <0.5 per game (should be low, no crates!)

### 3. If Results Look Good at 10k

Save checkpoint and move to Phase 3:
```bash
# Save Phase 1 checkpoint
cp agent_code/ppo/models/ppo_agent.pth agent_code/ppo/models/ppo_agent_phase1_final.pth

# Update settings.py
# Change: CRATE_DENSITY = 0.0 → CRATE_DENSITY = 0.75

# Update train.py rewards to Phase 3 (see above)

# Start Phase 3 training
python -m main play --agents ppo --train 1 --n-round 15000 --no-gui
```

---

## Timeline Estimate

- **Phase 1:** ~10k rounds = ~2-3 hours (depending on hardware)
- **Phase 3:** ~15k rounds = ~3-4 hours
- **Total to Phase 3 completion:** ~5-7 hours

---

## Key Differences from Old Training

### Old Approach (Failed)
❌ Jumped straight to Phase 3 with crates
❌ Agent tried to learn everything at once
❌ No basic skills to build on
❌ Result: 42% self-kill rate, negative rewards

### New Approach (Will Succeed)
✅ Start with Phase 1: coin collection (easier)
✅ Build basic navigation + survival skills
✅ Then add complexity (crates) in Phase 3
✅ Agent builds on solid foundation
✅ Expected: <20% self-kill, positive rewards

---

## Summary

**Current:** Phase 1 training in progress with new architecture
**Next:** Continue Phase 1 until round 10k
**Then:** Move to Phase 3 with crates
**Goal:** Learn strategic bombing while maintaining survival skills

Let Phase 1 run and check back at round 3000! 🚀
