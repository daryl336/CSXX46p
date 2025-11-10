# Current Training Status - Phase 2.5 Ready

## Executive Summary

**Status:** Ready to start Phase 2.5 training with comprehensive fixes
**Problem Fixed:** 67% self-kill rate in Phase 3
**Solution:** Three structural changes + intermediate Phase 2.5

---

## What Happened - Chronological Timeline

### Phase 1: Coin Collection (COMPLETED ✓)
- **Rounds:** 10,000
- **Results:**
  - Survival: 95.9% ✓
  - Coins/game: 2.03 ✓
  - Invalid actions: 0% ✓
- **Checkpoint:** `checkpoint_phase1_10000.pth`
- **Status:** Acceptable (could be better but good enough to move on)

### Phase 3 First Attempt (FAILED ✗)
- **Rounds:** 10,000
- **Configuration:** CRATE_DENSITY=0.75, Phase 3 rewards
- **Results:**
  - Survival: 59%
  - Self-kill: 41% ✗
  - Crates/game: 1.89
- **Problem Found:** Critical bug in `can_safely_escape_bomb()`
  - Function didn't simulate NEW bomb danger
  - Returned True immediately if current position was safe
  - Agent dropped bombs without checking if it could escape the NEW blast

### Critical Fix: Escape Check Bug
**File:** [agent_code/ppo/callbacks.py:166-254](agent_code/ppo/callbacks.py#L166-L254)

**What was fixed:**
```python
# BEFORE (BUGGY):
if danger_map[cx, cy] < danger_threshold:  # Uses OLD danger map
    return True

# AFTER (FIXED):
simulated_danger = danger_map.copy()
# Add NEW bomb danger at position and blast zones
for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
    for step in range(1, BOMB_POWER + 1):
        # ... add blast danger ...

if steps > 0:  # Skip starting position
    if simulated_danger[cx, cy] < danger_threshold:
        return True
```

### Phase 3 Second Attempt (WORSE ✗✗)
- **Rounds:** 10,000
- **Configuration:** Same as first attempt but with fixed escape check
- **Results:**
  - Survival: 33% (WORSE!)
  - Self-kill: 67% ✗✗
  - Crates/game: 4.40 ✓ (much better!)
- **Paradox:** Escape check works (more bombs, more crates) but deaths increased!

### Root Cause Analysis

**The escape check fix WORKED** - agent now correctly identifies when it can escape. This allowed more bombing, leading to 4.4 crates/game.

**But reward imbalance was exposed:**

Phase 3 reward math:
```
Die with 4 crates:  -500 + (4×90) = -140
Survive, 0 crates:  300 + (400×0.4) = +460
```

While dying is negative, it's "not negative enough" compared to the crate rewards. Agent learned: **"Get crates even if I die"** because exploration noise + 360 crate reward made death seem acceptable.

---

## The Solution - Three Structural Changes

### 1. Reward Rebalancing (Survival-First Approach)

**File:** [agent_code/ppo/train.py:19-49](agent_code/ppo/train.py#L19-L49)

| Reward | Phase 3 (Old) | Phase 2.5 (New) | Change |
|--------|---------------|-----------------|--------|
| CRATE_DESTROYED | 90 | 50 | ↓ 44% less attractive |
| SURVIVED_ROUND | 300 | 600 | ↑ 100% more valuable |
| KILLED_SELF | -500 | -800 | ↑ 60% stronger penalty |
| STEP_ALIVE | 0.4 | 0.7 | ↑ 75% more per step |
| ESCAPED_DANGER | 40 | 80 | ↑ 100% massive escape reward |
| MOVING_TO_SAFETY | 25 | 50 | ↑ 100% double escape progress |

**New reward math:**
```
Survive + 4 crates: 600 + (4×50) + (400×0.7) = +1,080 ✓
Die with 4 crates:  -800 + (4×50)            = -600  ✗
Survive, 0 crates:  600 + (400×0.7)          = +880  ✓
```

**Result:** Dying is now **1,480 points worse** than surviving! Survival always dominates.

### 2. Add Phase 2.5 (Gentle Crate Introduction)

**File:** [settings.py:8](settings.py#L8)

```python
CRATE_DENSITY = 0.3  # Intermediate phase
```

**Curriculum progression:**
- Phase 1: 0.0 crates (completed)
- **Phase 2.5: 0.3 crates** ← YOU ARE HERE
- Phase 3: 0.75 crates (coming next)

**Rationale:** Jump from 0.0 → 0.75 too large. Need intermediate step.

### 3. Stricter Escape Check

**File:** [agent_code/ppo/callbacks.py:224-226](agent_code/ppo/callbacks.py#L224-L226)

```python
# OLD: Accept any position with danger < 1.0 (threshold)
if simulated_danger[cx, cy] < danger_threshold:

# NEW: Only accept VERY safe positions (danger < 0.5)
if simulated_danger[cx, cy] < (danger_threshold * 0.5):
```

**Effect:** Agent can't drop bombs unless it can reach a very safe zone, not just barely safe.

---

## Current Configuration

### Game Settings
- **Crate Density:** 0.3 (Phase 2.5)
- **Max Steps:** 400
- **Bomb Power:** 3
- **Bomb Timer:** 4

### Network Architecture
- **Input:** 9 spatial planes (17×17) + 9 global features = 2610 total
- **New features added:** safe_neighbors, min_safety_distance, crates_in_bomb_range
- **Hidden layers:** [512, 512] for both actor and critic
- **Action masking:** Enabled (prevents invalid/unsafe moves)

### Training Hyperparameters
- **Batch size:** 256
- **Learning rate:** 3e-4
- **Gamma:** 0.99
- **GAE lambda:** 0.95
- **Clip epsilon:** 0.2
- **Entropy bonus:** 0.01

---

## What You Should Do Next

### Step 1: Restore Phase 1 Checkpoint
```bash
cp checkpoints/checkpoint_phase1_10000.pth checkpoints/latest.pth
```

**Why?** Phase 1 already learned safe movement (95.9% survival) and coin collection (2.03/game). Start from this solid foundation.

### Step 2: Verify Configuration
```bash
# Check crate density
python -c "import settings; print(f'Crate density: {settings.CRATE_DENSITY}')"
# Should print: 0.3

# Check checkpoint exists
ls -lh checkpoints/checkpoint_phase1_10000.pth
```

### Step 3: Start Phase 2.5 Training
```bash
python main.py play --agents ppo ppo ppo ppo --train 1 --n-rounds 10000
```

### Step 4: Monitor Metrics

**Check every 1000 rounds for:**

| Metric | Target | Warning Sign |
|--------|--------|--------------|
| Survival rate | >90% | <80% = too aggressive |
| Crates/game | 3-5 | <2 = too conservative |
| Coins/game | >1.5 | <1.0 = catastrophic forgetting |
| Self-kill rate | <5% | >10% = reward imbalance persists |
| Avg reward | >500 | <300 = not learning properly |

---

## Expected Results

### Optimistic Scenario (10k rounds)
- Survival: 92% ✓
- Crates: 4.2/game ✓
- Coins: 1.8/game ✓
- Self-kill: 3% ✓
- **Result:** Proceed to Phase 3 with confidence

### Realistic Scenario (10k rounds)
- Survival: 88% ✓
- Crates: 3.5/game ✓
- Coins: 1.5/game ✓
- Self-kill: 6% ✓
- **Result:** Good enough to proceed to Phase 3

### Pessimistic Scenario (10k rounds)
- Survival: <80% ✗
- Self-kill: >15% ✗
- **Result:** Need to extend training or tune rewards further

---

## Files Modified (All Changes Complete ✓)

1. ✓ [settings.py](settings.py) - CRATE_DENSITY = 0.3
2. ✓ [agent_code/ppo/train.py](agent_code/ppo/train.py) - Rebalanced all rewards
3. ✓ [agent_code/ppo/callbacks.py](agent_code/ppo/callbacks.py) - Fixed escape bug + stricter check

---

## Reference Documents

- **[PHASE_2.5_CONFIG.md](PHASE_2.5_CONFIG.md)** - Comprehensive Phase 2.5 guide
- **[FEATURE_ANALYSIS.md](FEATURE_ANALYSIS.md)** - Network architecture and features
- **[IMPLEMENTED_FIXES.md](IMPLEMENTED_FIXES.md)** - All fixes applied
- **[TRAINING_PLAN.md](TRAINING_PLAN.md)** - Overall curriculum plan

---

## Key Insights

1. **The escape check bug was critical** - Agent was dropping bombs without checking if it could escape the NEW blast zone

2. **Fixing the bug exposed reward imbalance** - Agent could now bomb more effectively, but rewards incentivized suicidal behavior

3. **Survival must dominate mathematically** - Not just "negative when you die" but "massively worse than surviving"

4. **Curriculum learning needs small jumps** - 0.0 → 0.75 crate density too large, need 0.3 intermediate

5. **Conservative is better than aggressive** - Stricter escape check (50% threshold) prevents risky situations

---

## Quick Troubleshooting

### If self-kill rate stays high (>10%)
1. Increase `SURVIVED_ROUND` to 800
2. Reduce `CRATE_DESTROYED` to 40
3. Train another 5k rounds

### If crate collection too low (<2/game)
1. Relax escape threshold to 0.6 (from 0.5)
2. Increase `CRATE_DESTROYED` to 60
3. Check if agent is dropping any bombs at all

### If coin collection drops (<1.0/game)
1. Increase `COIN_COLLECTED` to 120
2. Increase `APPROACHING_COIN_REWARD` to 12
3. May need to alternate training (coins vs crates)

---

## Bottom Line

**You are ready to start Phase 2.5 training!**

All structural fixes are implemented:
- ✓ Escape check bug fixed
- ✓ Rewards rebalanced (survival-first)
- ✓ Crate density reduced (0.3)
- ✓ Escape check stricter (50% threshold)

**Expected outcome:** Agent learns safe bombing with high survival (>90%) and reasonable crate destruction (3-5/game).

**Training command:**
```bash
cp checkpoints/checkpoint_phase1_10000.pth checkpoints/latest.pth
python main.py play --agents ppo ppo ppo ppo --train 1 --n-rounds 10000
```

Good luck! Monitor metrics every 1000 rounds and report back if you see concerning patterns.
