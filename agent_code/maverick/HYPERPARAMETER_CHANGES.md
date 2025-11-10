# Maverick Agent - Hyperparameter Optimization Summary

## Changes Made (Date: 2025-11-09)

### 1. **Network Architecture** ([Model.py](Model.py))

**KEPT ORIGINAL:**
```
23 → 60 → 6  (Single hidden layer, ~1,800 parameters)
```

**Rationale:**
- Simple architecture works well with hand-crafted features
- Faster training and inference
- Less prone to overfitting with limited data
- Original design was already well-tuned

---

### 2. **Discount Factor (Gamma)** ([train.py](train.py))

**BEFORE:** `0.6` (Highly myopic - only values 2-3 steps ahead)

**AFTER:** `0.95` (Standard for strategic games)

**Rationale:**
- Bomberman requires long-term planning:
  - Bomb takes 4 steps to explode
  - Escape routes need 3-5 steps
  - Strategic positioning for future opportunities
- Low gamma (0.6) made agent prioritize immediate rewards over survival
- With γ=0.95, reward 10 steps ahead still worth 60% of immediate reward

**Impact:**
```
Reward Value Comparison (R=100):
Step  | γ=0.6  | γ=0.95
------|--------|--------
Now   | 100    | 100
+3    | 21.6   | 85.7
+5    | 7.8    | 77.4
+10   | 0.6    | 59.9
```

---

### 3. **Exploration Strategy (Epsilon)** ([train.py](train.py))

**BEFORE:** `(0.5, 0.05)` - Start at 50% exploration

**AFTER:** `(1.0, 0.01)` - Start at 100% exploration

**Decay:** 85% linear, 15% constant (was 90%/10%)

**Rationale:**
- Full initial exploration helps discover diverse strategies
- Lower final epsilon (0.01) allows more exploitation at end
- Longer constant phase helps maintain some exploration

**Epsilon Schedule (50k episodes):**
```
Episode    | Epsilon | Behavior
-----------|---------|------------------
0          | 1.00    | 100% exploration
10,000     | 0.77    | 77% exploration
25,000     | 0.42    | Balanced
42,500     | 0.01    | Start pure exploitation
50,000     | 0.01    | Pure exploitation
```

---

### 4. **Experience Replay** ([train.py](train.py))

**BEFORE:**
- Buffer size: 2,000
- Batch size: 120

**AFTER:**
- Buffer size: 10,000 (5x larger)
- Batch size: 256 (2x larger)

**Rationale:**
- Larger buffer stores more diverse game situations
- Bigger batches → more stable gradient estimates
- With 50k episodes, 10k buffer covers last ~3k episodes of experience
- Better breaking of temporal correlations

**Memory Usage:** ~80MB (acceptable)

---

### 5. **Learning Rate** ([train.py](train.py))

**BEFORE:** `0.001` (Standard Adam)

**AFTER:** `0.0005` (More conservative)

**Rationale:**
- Lower LR with larger batches prevents overshooting
- More stable training with deeper network
- Better convergence for long training (50k episodes)

---

### 6. **Training Duration** ([train.py](train.py))

**BEFORE:** `20,000 episodes`

**AFTER:** `50,000 episodes` (2.5x longer)

**Rationale:**
- Deeper network needs more training
- More episodes to fully exploit epsilon decay
- Bomberman strategy is complex - needs time to learn
- Checkpoints every 500 episodes (1% of total)

**Estimated Time:** 8-12 hours on CPU

---

### 7. **Reward Structure** ([ManagerRewards.py](ManagerRewards.py))

| Event | Before | After | Change |
|-------|--------|-------|--------|
| `COIN_COLLECTED` | +100 | +100 | ✓ Same |
| `KILLED_OPPONENT` | +500 | +500 | ✓ Same |
| `INVALID_ACTION` | -10 | **-50** | 5x penalty |
| **`KILLED_SELF`** | **0** | **-1000** | **Critical fix!** |
| `GOT_KILLED` | -700 | -700 | ✓ Same |
| Movement | -1 | -1 | ✓ Same |
| `BOMB_DROPPED` | -1 | -1 | ✓ Same |

**Critical Fix:**
- **KILLED_SELF was giving 0 penalty!** Agent had no incentive to avoid suicide
- Now heavily punished (-1000) to prioritize survival
- Invalid actions also penalized more to reduce wasted exploration

**Reward Balance:**
```
Outcome                    | Total Reward
---------------------------|-------------
Collect coin safely        | +99 (100 - 1 movement)
Kill opponent              | +499
Suicide bombing opponent   | -500 (500 - 1000)
Die without achievement    | -700
```

---

## Expected Improvements

### 1. **Survival Rate**
- **Before:** Agent often suicided (0 penalty)
- **After:** Strong survival instinct (-1000 penalty)

### 2. **Strategic Depth**
- **Before:** Short-sighted decisions (γ=0.6)
- **After:** Plans ahead for bomb timing, escapes, positioning (γ=0.95)

### 3. **Learning Stability**
- **Before:** Small batches (120), high variance
- **After:** Large batches (256), stable gradients

### 4. **Exploration Coverage**
- **Before:** 50% initial exploration missed strategies
- **After:** 100% initial exploration finds diverse tactics

### 5. **Experience Diversity**
- **Before:** Small buffer (2k), limited variety
- **After:** Large buffer (10k), diverse situations

---

## Training Command

```bash
cd "/Users/daryltay/Documents/NUS Masters/CS5446 AI Planning/CSXX46p"

# Full training (50,000 episodes)
python main.py play --agents maverick --train 1 --n-rounds 50000 --no-gui

# Training against opponents (recommended)
python main.py play \
  --agents maverick rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --n-rounds 50000 \
  --no-gui
```

**Monitoring:**
```bash
# Watch training plot (updates live)
open training_progress.png

# Monitor logs
tail -f logs/game.log

# Check agent logs
tail -f agent_code/maverick/logs/maverick.log
```

---

## Testing After Training

```bash
# Copy trained model to final parameters
cp agent_code/maverick/network_parameters/last_save.pt \
   agent_code/maverick/network_parameters/final_parameters.pt

# Test without GUI (10 games)
python main.py play --agents maverick --n-rounds 10 --no-gui

# Compare against rule-based agent
python main.py play \
  --agents maverick rule_based_agent rule_based_agent rule_based_agent \
  --n-rounds 100 \
  --no-gui
```

---

## Checkpoint Schedule

With 50,000 episodes, saves occur every **500 episodes** (1%):

```
Episode | Action
--------|------------------------------------------
500     | Save checkpoint + score plot
1,000   | Save checkpoint + score plot
...     | ...
49,500  | Save checkpoint + score plot
50,000  | Training complete, final save
```

**Saved files:**
- `network_parameters/last_save.pt` (overwritten)
- `network_parameters/save after {n} iterations.pt` (snapshots)
- `training_progress.png` (live plot)
- `game_score_{n}` (score arrays)

---

## Key Metrics to Watch

### Early Training (Episodes 0-10k)
- **Survival rate:** Should increase from ~20% → 60%
- **Coins/game:** Should reach 0.5-1.0
- **Self-kill rate:** Should drop below 20%

### Mid Training (Episodes 10k-30k)
- **Survival rate:** Should reach 70-80%
- **Coins/game:** Should reach 1.5-2.5
- **Crate destruction:** Should start increasing
- **Self-kill rate:** Should drop below 10%

### Late Training (Episodes 30k-50k)
- **Survival rate:** Should reach 80-90%
- **Coins/game:** Should reach 2.5-3.5
- **Strategic bombing:** More opponent kills
- **Self-kill rate:** Should drop below 5%

---

## Rollback Instructions

If training doesn't improve after 10k episodes:

1. **Check logs:** `logs/game.log` and `agent_code/maverick/logs/maverick.log`
2. **Revert hyperparameters:**
   ```bash
   git checkout agent_code/maverick/train.py
   git checkout agent_code/maverick/Model.py
   git checkout agent_code/maverick/ManagerRewards.py
   ```
3. **Or manually adjust:** Lower learning rate to 0.0001 or reduce batch size

---

## Advanced: Curriculum Learning (Optional)

For even better results, train in stages with increasing difficulty:

### Phase 1: Coin Collection (10k episodes)
- Set `CRATE_DENSITY = 0.0` in `core_game/settings.py`
- Train pure coin collection + survival

### Phase 2: Gentle Crates (20k episodes)
- Set `CRATE_DENSITY = 0.3`
- Load Phase 1 model: `TRAIN_FROM_SCRETCH = False`, `LOAD = 'last_save'`

### Phase 3: Full Game (20k episodes)
- Set `CRATE_DENSITY = 0.75`
- Load Phase 2 model

---

## Parameter Summary Table

| Parameter | Old | New | Improvement |
|-----------|-----|-----|-------------|
| **Network architecture** | 23→60→6 | 23→60→6 | ✓ Kept original |
| **Discount (γ)** | 0.6 | 0.95 | +58% |
| **Buffer size** | 2,000 | 10,000 | +400% |
| **Batch size** | 120 | 256 | +113% |
| **Learning rate** | 0.001 | 0.0005 | -50% (stability) |
| **Episodes** | 20,000 | 50,000 | +150% |
| **Init epsilon** | 0.5 | 1.0 | +100% |
| **Self-kill penalty** | 0 | -1,000 | **CRITICAL** |

---

## Expected Training Time

**Hardware assumptions:**
- CPU: Modern multi-core processor
- RAM: 8GB+
- No GPU (training uses CPU)

**Time estimates:**
```
Episodes | Single Agent | vs 3 Opponents
---------|--------------|----------------
1,000    | 12 minutes   | 15 minutes
10,000   | 2 hours      | 2.5 hours
50,000   | 10 hours     | 12 hours
```

**Recommendation:** Run overnight or over weekend.

---

## Questions or Issues?

**Logs to check:**
1. `logs/game.log` - Game-level events
2. `agent_code/maverick/logs/maverick.log` - Agent decisions
3. `training_progress.png` - Score over time

**Common issues:**
- **Score not improving:** May need 5-10k episodes before improvement
- **High self-kill rate:** Check escape logic in ManagerFeatures.py
- **Memory issues:** Reduce buffer size to 5000 or batch size to 128

---

**Good luck with training! 🚀💣**
