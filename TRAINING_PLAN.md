# PPO Agent Training Plan for Bomberman

## Overview
This document outlines the complete training strategy using curriculum learning for your PPO agent. The plan is divided into 5 phases, each building on the previous one.

---

## Pre-Training Checklist

### 1. Verify Setup
```bash
# Test metrics tracking (should show no double counting)
python test_metrics_tracking.py

# Expected output: All ✓ PASS messages
```

### 2. Clean Old Metrics (Optional)
```bash
# Backup old metrics if they exist
mkdir -p metrics_backup
mv metrics/*.pkl metrics_backup/ 2>/dev/null || true
mv metrics/*.json metrics_backup/ 2>/dev/null || true
```

### 3. Verify Model Directory
```bash
mkdir -p models
ls -la models/  # Check if ppo_agent.pth exists

# If you want to start fresh:
# mv models/ppo_agent.pth models/ppo_agent_backup.pth
```

---

## Training Phases

## Phase 1: Survival & Movement Training
**Duration:** 5,000 rounds (~2-3 hours)
**Goal:** Learn to move safely and avoid danger

### Configuration
```bash
# Edit settings.py
# Set: CRATE_DENSITY = 0  # Empty arena for learning movement
```

### Run Training
```bash
python -m main play \
  --agents ppo \
  --train 1 \
  --n-round 5000 \
  --no-gui
```

### Expected Outcomes
- ✅ Agent survives >70% of rounds
- ✅ Learns to escape danger zones
- ✅ Reduces KILLED_SELF rate to <10%
- ✅ Moves efficiently (not stuck in corners)

### Monitor Progress
```bash
# Every 1000 rounds, check:
tail -100 logs/game.log | grep "survived"

# Check metrics (after round 1000):
python -c "
from metrics.metrics_tracker import MetricsTracker
tracker = MetricsTracker('PPO')
tracker.load('metrics/PPO_1000.pkl')
stats = tracker.get_summary_stats(last_n=100)
print(f'Survival Rate: {stats[\"avg_survival_time\"]/400:.2%}')
print(f'Self-Kill Rate: {stats[\"self_kill_rate\"]:.2f}')
"
```

### Troubleshooting Phase 1
| Problem | Solution |
|---------|----------|
| Agent keeps dying to own bombs | Increase UNSAFE_BOMB_PENALTY to -30.0 |
| Agent gets stuck in corners | Check action masking is working |
| Survival rate < 50% after 3000 rounds | Reduce learning rate to 1e-4 |

---

## Phase 2: Coin Collection Training
**Duration:** 10,000 rounds (~4-5 hours)
**Goal:** Learn efficient coin navigation

### Configuration
```bash
# Keep settings.py:
# CRATE_DENSITY = 0  # Still empty arena
```

### Run Training
```bash
python -m main play \
  --agents ppo \
  --train 1 \
  --n-round 10000 \
  --no-gui
```

### Expected Outcomes
- ✅ Collects 5+ coins per game on average
- ✅ Maintains >70% survival rate
- ✅ Shows directed movement toward coins

### Monitor Progress
```bash
# Check coin collection rate every 2000 rounds:
python -c "
from metrics.metrics_tracker import MetricsTracker
tracker = MetricsTracker('PPO')
tracker.load('metrics/PPO_10000.pkl')
stats = tracker.get_summary_stats(last_n=500)
print(f'Avg Coins: {stats[\"avg_coins_per_episode\"]:.2f}')
print(f'Survival: {stats[\"avg_survival_time\"]:.1f}/400')
"
```

### Troubleshooting Phase 2
| Problem | Solution |
|---------|----------|
| Agent ignores coins | Increase COIN_COLLECTED to 80.0 |
| Coin collection < 3 per game | Increase APPROACHING_COIN_REWARD to 3.0 |
| Agent wanders aimlessly | Check feature vector includes coin positions |

---

## Phase 3: Crate Destruction Training
**Duration:** 15,000 rounds (~6-8 hours)
**Goal:** Learn strategic bomb placement

### Configuration
```bash
# Edit settings.py
# Set: CRATE_DENSITY = 0.75  # NOW add crates
```

### Run Training
```bash
python -m main play \
  --agents ppo \
  --train 1 \
  --n-round 15000 \
  --no-gui
```

### Expected Outcomes
- ✅ Destroys 10+ crates per game
- ✅ Places bombs strategically (near crates)
- ✅ Successfully escapes from own bombs >95%
- ✅ Maintains survival rate >60%

### Monitor Progress
```bash
# Check bombing effectiveness:
python -c "
from metrics.metrics_tracker import MetricsTracker
tracker = MetricsTracker('PPO')
tracker.load('metrics/PPO_25000.pkl')  # 5k + 10k + 10k so far
stats = tracker.get_summary_stats(last_n=1000)
print(f'Crates/Game: {stats[\"avg_crates_per_episode\"]:.2f}')
print(f'Bomb Effectiveness: {stats[\"bomb_effectiveness\"]:.2%}')
print(f'Self-Kills: {stats[\"self_kill_rate\"]:.3f}')
"
```

### Troubleshooting Phase 3
| Problem | Solution |
|---------|----------|
| Agent doesn't place bombs | Increase CRATE_DESTROYED to 50.0 |
| Dies to own bombs frequently | Increase KILLED_SELF penalty to -500.0 |
| <5 crates destroyed per game | Increase CRATE_IN_RANGE_REWARD to 10.0 |

---

## Phase 4: Single Opponent Combat
**Duration:** 20,000 rounds (~8-10 hours)
**Goal:** Learn basic combat strategies

### Configuration
```bash
# Keep settings.py:
# CRATE_DENSITY = 0.75
```

### Run Training
```bash
python -m main play \
  --agents ppo rule_based_agent \
  --train 1 \
  --n-round 20000 \
  --no-gui
```

### Expected Outcomes
- ✅ 40%+ win rate vs rule_based_agent
- ✅ Kills opponent 30%+ of games
- ✅ Avoids being killed by opponent 60%+
- ✅ Strategic bomb placement near opponent

### Monitor Progress
```bash
# Check combat performance:
python -c "
from metrics.metrics_tracker import MetricsTracker
tracker = MetricsTracker('PPO')
tracker.load('metrics/PPO_45000.pkl')
stats = tracker.get_summary_stats(last_n=2000)
print(f'Win Rate: {stats[\"win_rate\"]:.2%}')
print(f'Kills/Game: {stats[\"avg_kills_per_episode\"]:.2f}')
print(f'Deaths/Game: {1 - stats[\"win_rate\"]:.2f}')

# Compare vs opponent type:
opp_stats = tracker.compare_by_opponent()
for opp, data in opp_stats.items():
    print(f'{opp}: {data[\"win_rate\"]:.2%} WR')
"
```

### Troubleshooting Phase 4
| Problem | Solution |
|---------|----------|
| Win rate < 25% after 10k rounds | Reduce opponent difficulty (use random agent) |
| Agent too passive | Increase KILLED_OPPONENT to 300.0 |
| Agent too aggressive (dies often) | Increase GOT_KILLED penalty to -200.0 |

---

## Phase 5: Multi-Agent Combat
**Duration:** 30,000+ rounds (~12-15 hours)
**Goal:** Master 4-player competition

### Configuration
```bash
# Keep settings.py:
# CRATE_DENSITY = 0.75
```

### Run Training
```bash
# Start with 2 opponents
python -m main play \
  --agents ppo rule_based_agent rule_based_agent \
  --train 1 \
  --n-round 15000 \
  --no-gui

# Then 3 opponents
python -m main play \
  --agents ppo rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --n-round 15000 \
  --no-gui
```

### Expected Outcomes
- ✅ 25%+ win rate in 4-player games (random = 25%)
- ✅ 30%+ win rate in 3-player games (random = 33%)
- ✅ Strategic positioning away from multiple threats
- ✅ Opportunistic kills when safe

### Monitor Progress
```bash
# Final evaluation:
python -c "
from metrics.metrics_tracker import MetricsTracker
tracker = MetricsTracker('PPO')
tracker.load('metrics/PPO_75000.pkl')  # Total rounds
stats = tracker.get_summary_stats(last_n=5000)
print('=== FINAL PERFORMANCE ===')
print(f'Win Rate: {stats[\"win_rate\"]:.2%}')
print(f'Avg Rank: {stats[\"avg_rank\"]:.2f}')
print(f'Avg Survival: {stats[\"avg_survival_time\"]:.1f}/400 steps')
print(f'Kills/Game: {stats[\"avg_kills_per_episode\"]:.2f}')
print(f'Coins/Game: {stats[\"avg_coins_per_episode\"]:.2f}')
print(f'Crates/Game: {stats[\"avg_crates_per_episode\"]:.2f}')
"
```

---

## Hyperparameter Schedule

### Current Settings (Good for Phases 1-3)
```python
lr = 3e-4
entropy_coef = 0.04
batch_size = 1024
clip_eps = 0.2
```

### Recommended Adjustments

#### After Phase 3 (Round 25,000):
```python
# In callbacks.py, modify setup():
lr = 2e-4  # Reduce learning rate for stability
entropy_coef = 0.03  # Reduce exploration
```

#### After Phase 4 (Round 45,000):
```python
lr = 1e-4  # Further reduce for fine-tuning
entropy_coef = 0.02  # More exploitation
```

#### Phase 5 (Final refinement):
```python
lr = 5e-5  # Very conservative
entropy_coef = 0.01  # Mostly exploitation
```

---

## Reward Tuning Guidelines

### If Agent is Too Passive
```python
STEP_ALIVE_REWARD = 0.05  # Reduce (was 0.1)
SAFE_POSITION_REWARD = 0.1  # Reduce (was 0.2)
EXPLORATION_BONUS = 1.0  # Increase (was 0.8)
KILLED_OPPONENT = 300.0  # Increase (was 200.0)
```

### If Agent is Too Aggressive (Dies Often)
```python
KILLED_SELF = -500.0  # Increase penalty (was -300.0)
GOT_KILLED = -250.0  # Increase penalty (was -150.0)
SURVIVED_ROUND = 150.0  # Increase reward (was 100.0)
STEP_ALIVE_REWARD = 0.15  # Increase (was 0.1)
```

### If Agent Ignores Objectives
```python
COIN_COLLECTED = 80.0  # Increase (was 50.0)
CRATE_DESTROYED = 50.0  # Increase (was 30.0)
APPROACHING_COIN_REWARD = 3.0  # Increase (was 2.0)
```

---

## Evaluation Protocol

### Every 5,000 Rounds
1. Save checkpoint:
   ```bash
   cp models/ppo_agent.pth models/ppo_agent_round_${ROUND}.pth
   ```

2. Run evaluation games (no training):
   ```bash
   python -m main play --agents ppo rule_based_agent rule_based_agent rule_based_agent --n-round 100
   ```

3. Analyze metrics:
   ```bash
   python metrics/metrics_visualization.py  # Generate plots
   ```

---

## Success Criteria

### Phase 1 Success
- [ ] 70%+ survival rate
- [ ] <10% self-kill rate
- [ ] Average survival > 280 steps

### Phase 2 Success
- [ ] 5+ coins per game
- [ ] Maintains Phase 1 survival
- [ ] Shows directed coin-seeking

### Phase 3 Success
- [ ] 10+ crates destroyed per game
- [ ] <5% self-kill rate with bombs
- [ ] 60%+ survival rate

### Phase 4 Success
- [ ] 40%+ win rate vs 1 opponent
- [ ] 0.3+ kills per game
- [ ] Strategic bomb usage

### Phase 5 Success (FINAL GOAL)
- [ ] 25%+ win rate in 4-player
- [ ] 1.5+ average rank (1=best, 4=worst)
- [ ] 250+ average survival steps
- [ ] Balanced play (coins + combat + survival)

---

## Next Steps

1. **Run the test:**
   ```bash
   python test_metrics_tracking.py
   ```

2. **If tests pass, start Phase 1:**
   ```bash
   # Set CRATE_DENSITY = 0 in settings.py
   python -m main play --agents ppo --train 1 --n-round 5000 --no-gui
   ```

3. **Monitor training:** Check logs every 1000 rounds

4. **Save checkpoints:** After each phase completes

5. **Adjust rewards:** Based on observed behavior

---

## Troubleshooting

### Training is unstable (rewards oscillating)
- Reduce learning rate by 50%
- Increase batch_size to 2048
- Reduce clip_eps to 0.15

### Agent not improving after 10k rounds
- Check if action masking is too restrictive
- Verify feature vector contains necessary info
- Try different opponent types for diversity

### Metrics showing weird values
- Run test_metrics_tracking.py again
- Check for double counting in logs
- Verify no manual edits broke the tracking

---

## Estimated Timeline

| Phase | Rounds | CPU Time | GPU Time* | Total Time |
|-------|--------|----------|-----------|------------|
| 1 | 5,000 | 2-3h | 0.5-1h | 3h |
| 2 | 10,000 | 4-5h | 1-2h | 8h |
| 3 | 15,000 | 6-8h | 2-3h | 16h |
| 4 | 20,000 | 8-10h | 3-4h | 26h |
| 5 | 30,000 | 12-15h | 4-6h | 41h |
| **Total** | **80,000** | **32-41h** | **10-16h** | **~2 days** |

*GPU times are estimates - actual may vary

---

## Contact & Support

If you encounter issues:
1. Check logs in `logs/game.log`
2. Verify metrics in `metrics/` directory
3. Review this training plan
4. Adjust hyperparameters based on guidelines above

Good luck with training! 🎮💣
