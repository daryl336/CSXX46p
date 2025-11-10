# Phase 3 Configuration - Strategic Bombing

## Configuration Complete! ✅

All settings updated for Phase 3 training.

---

## What Changed

### 1. Settings (✅ Complete)
```python
# settings.py
CRATE_DENSITY = 0.75  # Changed from 0.0 → 0.75
```

### 2. Game Event Rewards (✅ Complete)
```python
# train.py - GAME_REWARDS
COIN_COLLECTED: 100.0      # Reduced from 150 (still important)
CRATE_DESTROYED: 90.0      # NEW PRIMARY GOAL (was 50)
BOMB_DROPPED: 0.0          # Neutral (was -5)
KILLED_SELF: -500.0        # Increased from -400
SURVIVED_ROUND: 300.0      # Increased from 200
WAITED: -1.0               # Reduced from -2 (less harsh)
```

### 3. Shaped Rewards (✅ Complete)
```python
# train.py - Shaped rewards
STEP_ALIVE_REWARD: 0.4           # Doubled from 0.2
DANGER_PENALTY: -1.5             # Reduced from -5 (allow risks)
ESCAPED_DANGER_REWARD: 40.0      # Increased from 15
SAFE_BOMB_REWARD: 80.0           # Increased from 30
BOMB_STAY_PENALTY: -25.0         # Increased from -10
MOVING_TO_SAFETY_REWARD: 25.0    # Increased from 15
CRATE_IN_RANGE_REWARD: 15.0      # Increased from 5
APPROACHING_COIN_REWARD: 8.0     # Same (maintain skill)
SAFE_POSITION_REWARD: 0.3        # Increased from 0.1
EXPLORATION_BONUS: 2.5           # Increased from 2.0
```

---

## Phase 3 Reward Philosophy

### Survival First, Bombing Second
**Total survival value per round:**
- Steps alive: 400 × 0.4 = **+160**
- Survived round: **+300**
- Safe position: 400 × 0.3 = **+120**
- **Total survival: +580**

**Successful bombing run (3 bombs, 9 crates):**
- Survival base: **+580**
- Crates in range: 3 × 15 = **+45**
- Escaped danger: 3 × 40 = **+120**
- Moving to safety: 3 × 25 = **+75**
- Crates destroyed: 9 × 90 = **+810**
- Safe bombs: 3 × 80 = **+240**
- Coins: 2 × 100 = **+200**
- Danger penalty: ~15 steps × -1.5 = **-23**
- **Total: +2,047** ✅✅✅

**Suicide scenario:**
- Steps alive: 150 × 0.4 = **+60**
- Killed self: **-500**
- Some crates: 3 × 90 = **+270**
- Waits: 50 × -1.0 = **-50**
- **Total: -220** ❌

**Survival clearly dominates!**

---

## What the Agent Will Learn

### Phase 1 Skills (Already Learned) ✅
1. Navigate the map
2. Avoid walls and obstacles
3. Collect coins
4. Survive full rounds (95.9% survival!)

### Phase 3 New Skills (To Learn)
1. **Identify good bombing positions** (crates in range)
2. **Drop bombs safely** (escape check ensures safe drops)
3. **Escape after bombing** (+25 per step toward safety!)
4. **Destroy crates** (+90 each)
5. **Balance coins + crates** (both rewarded)

---

## Expected Training Progression

### Round 3,000 (Early Phase 3)
- **Self-kill:** 25-35% (learning bomb escape)
- **Crates/game:** 2-4 (starting to bomb)
- **Coins/game:** 1-2 (maintaining skill)
- **Bombs/game:** 1.5-2.5
- **Avg reward:** +50 to +150
- **Survival:** 75-85%

### Round 6,000 (Mid Phase 3)
- **Self-kill:** 20-28% (improving)
- **Crates/game:** 4-7 (getting better)
- **Coins/game:** 1.5-2.5
- **Bombs/game:** 2.5-3.5
- **Avg reward:** +150 to +250
- **Survival:** 80-88%

### Round 10,000 (Late Phase 3)
- **Self-kill:** 15-22% (much better)
- **Crates/game:** 6-10 (good performance)
- **Coins/game:** 2-3
- **Bombs/game:** 3-4
- **Avg reward:** +250 to +400
- **Survival:** 85-92%

### Round 15,000 (Phase 3 Complete)
- **Self-kill:** <20% ✅
- **Crates/game:** 8-12 ✅
- **Coins/game:** 2-4 ✅
- **Bombs/game:** 3-5 ✅
- **Avg reward:** +350 to +500 ✅
- **Survival:** 88-95% ✅

---

## Key Improvements from Previous Attempts

### Old Phase 3 (Failed - 42% self-kill)
❌ No escape check (dropped bombs in death traps)
❌ No escape features (couldn't evaluate safety)
❌ No escape progress reward (no learning signal)
❌ Started from Phase 1 model (wrong architecture)

### New Phase 3 (Will Succeed)
✅ **BFS escape check** (only drops safe bombs)
✅ **3 new escape features** (safe_neighbors, min_safety_dist, crates_in_range)
✅ **Escape progress reward** (+25 per step to safety)
✅ **Balanced rewards** (survival = +580, successful bombing = +2047)
✅ **Starting from Phase 1 skills** (95.9% survival!)

---

## Training Command

```bash
# Clean old Phase 3 metrics (optional)
rm -rf agent_code/ppo/metrics/PPO_*.pkl
rm -rf agent_code/ppo/metrics/PPO_*_summary.json

# Start Phase 3 training
python -m main play --agents ppo --train 1 --n-round 15000 --no-gui
```

---

## Monitoring Checklist

### ✅ Check at Round 3,000
Expected signs of progress:
- [ ] Self-kill rate dropping below 35%
- [ ] Crates/game increasing (2+)
- [ ] Avg reward positive (+50+)
- [ ] Bombs placed: 1.5-2.5 per game

**If not meeting targets:** Agent may need reward adjustments

### ✅ Check at Round 6,000
Expected improvement:
- [ ] Self-kill rate below 30%
- [ ] Crates/game: 4-7
- [ ] Avg reward: +150+
- [ ] Survival: 80%+

**If not meeting targets:** Continue to 10k, agent still learning

### ✅ Check at Round 10,000
Expected near-completion:
- [ ] Self-kill rate below 25%
- [ ] Crates/game: 6-10
- [ ] Avg reward: +250+
- [ ] Survival: 85%+

**If meeting targets:** Can stop at 12k instead of 15k

### ✅ Final Check at Round 15,000
Success criteria:
- [ ] Self-kill rate: <20%
- [ ] Crates/game: ≥8
- [ ] Avg reward: ≥+350
- [ ] Survival: ≥88%

**If all met:** Phase 3 complete! Ready for Phase 4 (1v1)

---

## Troubleshooting

### If self-kill rate stays >35% after 6k rounds:
**Problem:** Escape mechanism not working
**Solution:** Check if `can_safely_escape_bomb()` is too restrictive

### If crates/game <4 after 6k rounds:
**Problem:** Not bombing enough
**Solution:** Increase `CRATE_IN_RANGE_REWARD` and `SAFE_BOMB_REWARD`

### If reward stays negative after 3k rounds:
**Problem:** Penalties too harsh or not enough positive rewards
**Solution:** Reduce `KILLED_SELF` penalty or increase `CRATE_DESTROYED` reward

### If survival <75% after 6k rounds:
**Problem:** Agent too aggressive
**Solution:** Increase `SURVIVED_ROUND` and `STEP_ALIVE_REWARD`

---

## After Phase 3 Success

### Save Final Checkpoint
```bash
cp agent_code/ppo/models/ppo_agent.pth agent_code/ppo/models/ppo_agent_phase3_final.pth
```

### Move to Phase 4 (1v1 Combat)
Update configuration for opponent training:
```bash
# Update rewards for combat
# Add opponent in training command
python -m main play --agents ppo rule_based_agent --train 1 --n-round 10000 --no-gui
```

---

## Summary

✅ **Phase 3 is configured and ready!**

**Key features:**
- Better escape checking (BFS pathfinding)
- Escape route features (agent knows when safe)
- Escape progress rewards (learn step-by-step)
- Balanced rewards (survival + strategic bombing)
- Starting from good foundation (95.9% Phase 1 survival)

**Expected outcome:**
- Agent learns to bomb strategically
- Destroys 8-12 crates per game
- Maintains 88-95% survival
- Gets +350 to +500 avg reward

**Start training now!** 🚀

```bash
python -m main play --agents ppo --train 1 --n-round 15000 --no-gui
```
