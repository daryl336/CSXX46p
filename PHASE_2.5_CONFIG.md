# Phase 2.5 Configuration - Gentle Crate Introduction

## Overview
Phase 2.5 is an **intermediate training phase** designed to bridge the gap between Phase 1 (coin collection, no crates) and Phase 3 (full crate density). This phase addresses the 67% self-kill rate observed in direct Phase 1 → Phase 3 transition.

## Why Phase 2.5?

### Problem Identified
After fixing the critical escape check bug, Phase 3 training showed:
- ✅ Crate destruction improved: 1.89 → 4.40 per game
- ❌ Self-kill rate increased: 41% → 67%
- Root cause: **Reward imbalance** - agent learned "get crates even if I die"

### Solution
Three-pronged structural approach:
1. **Rebalance rewards** - Make survival dominate
2. **Reduce crate density** - Gentler learning curve
3. **Stricter escape check** - Only allow very safe bombs

---

## Configuration Changes

### 1. Game Settings ([settings.py](settings.py))

```python
CRATE_DENSITY = 0.3  # REDUCED from 0.75 (Phase 3) or 0.0 (Phase 1)
```

**Rationale:**
- Phase 1: 0.0 crates (pure coin collection)
- **Phase 2.5: 0.3 crates (gentle introduction)**  ← YOU ARE HERE
- Phase 3: 0.75 crates (full density)

This creates a smoother curriculum with smaller jumps.

---

### 2. Reward Rebalancing ([agent_code/ppo/train.py:19-49](agent_code/ppo/train.py#L19-L49))

#### Game Event Rewards
```python
GAME_REWARDS = {
    # Basic rewards
    "COIN_COLLECTED": 80.0,        # DOWN from 100 (Phase 3)
    "CRATE_DESTROYED": 50.0,       # DOWN from 90 - LESS ATTRACTIVE ✓
    "KILLED_SELF": -800.0,         # DOWN from -500 - STRONGER PENALTY ✓
    "SURVIVED_ROUND": 600.0,       # UP from 300 - MUCH MORE VALUABLE ✓

    # Movement penalties (unchanged)
    "INVALID_ACTION": -30.0,
    "MOVED_IN_CIRCLE": -8.0,
    "WAITED": -2.0,
}
```

#### Shaped Rewards
```python
STEP_ALIVE_REWARD = 0.7           # UP from 0.4 - Higher value per step ✓
ESCAPED_DANGER_REWARD = 80.0      # UP from 40 - MASSIVE escape reward ✓
MOVING_TO_SAFETY_REWARD = 50.0    # UP from 25 - DOUBLE escape progress ✓
APPROACHING_COIN_REWARD = 8.0     # Unchanged
```

---

### 3. Reward Math Analysis

#### Phase 3 (OLD - Broken):
```
Survive + 4 crates: 300 + (4×90) + (400×0.4) = +820
Die with 4 crates:  -500 + (4×90)            = -140
Survive, 0 crates:  300 + (400×0.4)          = +460
```
**Problem:** Dying with crates (-140) better than surviving without (+460)? NO!
But the negative value is small enough that stochastic exploration leads to deaths.

#### Phase 2.5 (NEW - Fixed):
```
Survive + 4 crates: 600 + (4×50) + (400×0.7) = +1,080 ✓
Die with 4 crates:  -800 + (4×50)            = -600   ✗
Survive, 0 crates:  600 + (400×0.7)          = +880   ✓
```
**Solution:** Survival ALWAYS dominates death, even with crates!

**Key Insight:**
- Dying with 4 crates: **-600** (terrible)
- Surviving with 0 crates: **+880** (great!)
- Survival is now **1,480 reward points better** than dying!

---

### 4. Stricter Escape Check ([agent_code/ppo/callbacks.py:224-226](agent_code/ppo/callbacks.py#L224-L226))

#### Change Made
```python
# OLD (Phase 3):
if simulated_danger[cx, cy] < danger_threshold and explosion_map[cx, cy] == 0:
    return True

# NEW (Phase 2.5):
if simulated_danger[cx, cy] < (danger_threshold * 0.5) and explosion_map[cx, cy] == 0:
    return True
```

**Effect:** Agent can only drop bombs if it can reach a position with danger < 0.5 (very safe), not just < 1.0 (barely safe).

**Rationale:** Prevent "barely escaping" situations that often lead to death due to:
- Explosion timing uncertainty
- Other agents' bombs appearing
- Path getting blocked

---

## Training Plan

### Starting Point
```bash
# Restore Phase 1 checkpoint
cp checkpoints/checkpoint_phase1_10000.pth checkpoints/latest.pth
```

**Why start from Phase 1?**
- Phase 1 achieved 95.9% survival, 2.03 coins/game
- Already learned safe movement and coin collection
- Now learn bombing while maintaining survival skills

### Expected Behavior

#### Early Training (Rounds 0-3000)
- Survival should stay high (>90%)
- Crate destruction gradually increases (0 → 2 per game)
- Agent learns to drop bombs in safe situations only

#### Mid Training (Rounds 3000-7000)
- Crate destruction continues increasing (2 → 4 per game)
- Survival maintained (>85%)
- Agent learns escape patterns with stricter safety margin

#### Late Training (Rounds 7000-10000)
- Crate destruction plateaus (4-5 per game expected with 0.3 density)
- Survival high (>90%)
- Agent masters safe bombing with escape

### Success Metrics (Target for 10k rounds)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Survival Rate | >90% | Much better than Phase 3's 33% |
| Crates/Game | 3-5 | Reasonable with 0.3 density |
| Coins/Game | >1.5 | Maintain Phase 1 skill |
| Self-Kill Rate | <5% | Down from 67% |
| Avg Reward | >500 | Positive learning signal |
| Invalid Actions | <1% | Already achieved |

---

## After Phase 2.5

### If Successful (Metrics meet targets)
**Proceed to Phase 3:**
1. Save checkpoint: `checkpoint_phase2.5_10000.pth`
2. Update `CRATE_DENSITY = 0.75` in [settings.py](settings.py)
3. Keep same rewards (already optimized for survival-first)
4. Train another 10k rounds
5. Expected: ~8-12 crates/game with >80% survival

### If Unsuccessful (High self-kill persists)

**Option A: Extend Phase 2.5**
- Train another 5-10k rounds
- Agent may need more time to learn escape patterns

**Option B: Further reward tuning**
- Increase `SURVIVED_ROUND` to 800
- Increase `ESCAPED_DANGER_REWARD` to 100
- Reduce `CRATE_DESTROYED` to 40

**Option C: Add even gentler Phase 2.25**
- `CRATE_DENSITY = 0.15` (very sparse)
- Helps if 0.3 is still too hard

---

## Implementation Summary

### Files Modified

1. **[settings.py](settings.py#L8)**
   - `CRATE_DENSITY = 0.3`

2. **[agent_code/ppo/train.py](agent_code/ppo/train.py#L19-L49)**
   - Rebalanced all reward values
   - Survival-first approach

3. **[agent_code/ppo/callbacks.py](agent_code/ppo/callbacks.py#L224-L226)**
   - Stricter escape check (50% threshold)

### Files to Review Before Training

- Verify `CRATE_DENSITY = 0.3` in [settings.py](settings.py)
- Verify Phase 1 checkpoint exists: `checkpoints/checkpoint_phase1_10000.pth`
- Check training hyperparameters in [agent_code/ppo/train.py](agent_code/ppo/train.py#L53-L61)

---

## Theoretical Analysis

### Why This Should Work

1. **Reward Hierarchy (by value per game)**
   - Survive + crates: ~1,000 reward ✓ (best outcome)
   - Survive + 0 crates: ~880 reward ✓ (safe fallback)
   - Die with crates: -600 reward ✗ (worst outcome)

2. **Safety First Learning**
   - Stricter escape check forces conservative bombing
   - High escape rewards reinforce safe behavior
   - Massive death penalty discourages risky plays

3. **Gradual Complexity**
   - 0.3 density = ~3-6 crates per map
   - Enough to learn but not overwhelming
   - Smooth transition from 0.0 → 0.3 → 0.75

### Potential Issues to Watch

1. **Over-conservative bombing**
   - Agent may drop very few bombs (high survival, low crates)
   - Solution: If crates/game < 2 after 10k, slightly relax escape threshold

2. **Catastrophic forgetting of coins**
   - Coins/game may drop below 1.0
   - Solution: If this happens, increase `COIN_COLLECTED` reward to 120

3. **Escape reward exploitation**
   - Agent may intentionally get near danger to farm escape rewards
   - Monitor: If agent circles bombs without destroying crates, reduce escape rewards

---

## Quick Start

```bash
# 1. Restore Phase 1 checkpoint
cp checkpoints/checkpoint_phase1_10000.pth checkpoints/latest.pth

# 2. Verify configuration
python -c "import settings; print(f'Crate density: {settings.CRATE_DENSITY}')"
# Should print: Crate density: 0.3

# 3. Start training
python main.py play --agents ppo ppo ppo ppo --train 1 --n-rounds 10000

# 4. Monitor metrics every 1000 rounds
# Watch for: survival_rate, crates_destroyed_avg, coins_collected_avg, self_kill_rate
```

---

## Expected Timeline

- **Rounds 0-1000:** Learning to drop bombs safely (~1-2 crates/game)
- **Rounds 1000-5000:** Improving escape patterns (~2-4 crates/game)
- **Rounds 5000-10000:** Mastering safe bombing (~3-5 crates/game)

**Total estimated training time:** ~2-3 hours (depending on hardware)

---

## Success Criteria

Phase 2.5 is considered **successful** if after 10k rounds:

✅ Survival rate > 90%
✅ Crates destroyed > 3/game
✅ Coins collected > 1.5/game
✅ Self-kill rate < 5%
✅ Average reward > 500

If all criteria met → **Proceed to Phase 3 with confidence!**
