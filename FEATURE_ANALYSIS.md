# Feature & Data Analysis for PPO Bomberman Agent

## Current Feature Representation

### Spatial Features (9 planes of 17x17 = 2601 values)
1. **Walls** - Static obstacles (-1 in field)
2. **Crates** - Destructible obstacles (1 in field)
3. **Free tiles** - Walkable spaces (0 in field)
4. **Coins** - Collectible items
5. **Others** - Opponent positions
6. **Bombs map** - Active bombs with timer decay (1/(1+timer))
7. **Explosion map** - Current explosions (normalized by EXPLOSION_TIMER)
8. **Danger map** - Computed bomb blast zones with timer decay
9. **Self map** - Agent's current position

### Global Features (6 scalar values)
1. **Has bomb available** - Binary (bombs_left > 0)
2. **Current danger level** - danger_map[x, y]
3. **Total coins remaining** - Normalized by MAX_COIN_COUNT
4. **Active bomb count** - Normalized by MAX_AGENTS
5. **Game step** - Normalized by MAX_STEPS (400)
6. **Round number** - Normalized by 1000

**Total: 2607 features**

---

## Missing Features Critical for Bombing Behavior

### 1. **Escape Route Information** ⚠️ CRITICAL
**Current:** Agent only knows danger map
**Missing:**
- Number of safe escape routes from current position
- Distance to nearest safe tile
- Whether escape routes exist in each direction
- Minimum steps needed to reach safety

**Why it matters:** Agent can't evaluate if a bomb drop is safe without knowing escape options

**Proposed addition:**
```python
# Add to global features:
safe_neighbors = count_safe_adjacent_tiles(game_state, x, y, danger_map)
min_distance_to_safety = compute_min_safety_distance(game_state, x, y, danger_map)
can_escape_up/right/down/left = directional_escape_flags(game_state, x, y, danger_map)
```

### 2. **Crate Density Information** ⚠️ IMPORTANT
**Current:** Agent sees individual crates but no density info
**Missing:**
- Number of crates in bomb range from current position
- Crate density in each cardinal direction
- Optimal bombing positions relative to agent

**Why it matters:** Agent needs to know if bombing current position is worthwhile

**Proposed addition:**
```python
# Add to global features:
crates_in_bomb_range = count_crates_in_bomb_range(game_state, (x, y))
crates_nearby_count = count_crates_in_radius(game_state, (x, y), radius=5)
```

### 3. **Temporal Information** ⚠️ MODERATE
**Current:** Bomb timers encoded as 1/(1+timer)
**Missing:**
- Explicit time-to-explosion for nearest bomb
- Whether agent just dropped a bomb (post-bomb state)
- Time since last bomb drop

**Why it matters:** Agent needs to understand urgency of escape

**Proposed addition:**
```python
# Add to global features:
nearest_bomb_timer = get_nearest_bomb_timer(game_state, (x, y))
just_placed_bomb = float(hasattr(self, 'post_bomb_timer') and self.post_bomb_timer > 0)
```

### 4. **Historical Context** ⚠️ LOW PRIORITY
**Current:** Only current frame
**Missing:**
- Previous positions (frame stacking)
- Movement direction/velocity
- Previous actions

**Why it matters:** Helps with temporal reasoning

**Not recommended:** Adds complexity, current features should be sufficient

---

## Current Action Masking Analysis

### What's Working ✅
- Invalid movements masked (walls, crates, out of bounds)
- Explosions masked
- High danger zones masked (threshold = 0.75)
- Bombs masked if no bombs left
- Simple escape check: at least one adjacent safe tile

### Potential Issues ❌

#### 1. **Escape Check is Too Simple**
```python
# Current implementation (line 112-121):
has_escape = False
for dx, dy in DIRECTIONS.values():
    ex, ey = x + dx, y + dy
    if 0 <= ex < field.shape[0] and 0 <= ey < field.shape[1]:
        if field[ex, ey] == 0 and (ex, ey) not in bomb_positions:
            if danger_map[ex, ey] < danger_threshold and explosion_map[ex, ey] == 0:
                has_escape = True
                break
if not has_escape:
    mask[ACTION_IDX['BOMB']] = 0.0
```

**Problem:** Only checks **immediate adjacent tiles**! Doesn't check if agent can actually **reach** safety within bomb timer.

**Example failure case:**
```
Agent at position with crates around:
W W W W W
W C C C W
W C A C W  (A = Agent, C = Crate)
W . . . W  (. = free tile)
W W W W W
```
Agent has one free tile adjacent (down), but **that tile is ALSO in bomb blast!**
Current check passes (has adjacent free tile), but agent will die!

**Fix needed:**
```python
def can_safely_escape_bomb(game_state, position, danger_map, bomb_timer=4):
    """Check if agent can reach safety before bomb explodes."""
    # BFS to find path to safety within bomb_timer steps
    from collections import deque

    field = np.array(game_state["field"])
    x, y = position
    queue = deque([(x, y, 0)])  # (x, y, steps)
    visited = {(x, y)}

    while queue:
        cx, cy, steps = queue.popleft()

        # Found safe tile within time limit?
        if danger_map[cx, cy] < DANGER_THRESHOLD and steps <= bomb_timer:
            return True

        if steps >= bomb_timer:
            continue

        for dx, dy in DIRECTIONS.values():
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and
                field[nx, ny] == 0 and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))

    return False
```

#### 2. **DANGER_THRESHOLD May Be Too High**
Current: 0.75 (line 32)
- This means only mask when danger is VERY imminent (timer < 1.33 steps)
- Agent might walk into slightly lower danger zones

---

## Reward Shaping Analysis

### Current Shaped Rewards (from train.py)
```python
STEP_ALIVE_REWARD = 0.8              # Every step alive
DANGER_PENALTY = -1.0                # Per step in danger
ESCAPED_DANGER_REWARD = 60.0         # Escaped from danger zone
SAFE_BOMB_REWARD = 120.0             # Bomb exploded, destroyed crates, survived
BOMB_STAY_PENALTY = -15.0            # Stayed on bomb position
UNSAFE_BOMB_PENALTY = -10.0          # (Not currently used?)
CRATE_IN_RANGE_REWARD = 12.0         # Bomb placed with crates in range
APPROACHING_COIN_REWARD = 10.0       # Moving closer to coin
SAFE_POSITION_REWARD = 1.0           # Per step in safe position
EXPLORATION_BONUS = 2.0              # Visiting new tile
```

### Missing Shaped Rewards

#### 1. **Escape Progress Reward** ⚠️ CRITICAL
**Missing:** Reward for moving toward safety after placing bomb

**Proposed:**
```python
MOVING_TO_SAFETY_REWARD = 15.0  # Reward each step that moves away from danger

# In game_events_occurred():
if hasattr(self, 'post_bomb_timer') and self.post_bomb_timer > 0:
    if old_game_state and new_game_state:
        old_danger = compute_danger_map(old_game_state)
        new_danger = compute_danger_map(new_game_state)
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        # Reward moving to lower danger
        if new_danger[new_pos] < old_danger[old_pos]:
            reward += MOVING_TO_SAFETY_REWARD
```

#### 2. **Multi-Crate Bonus** ⚠️ MODERATE
**Missing:** Extra reward for hitting multiple crates with one bomb

**Proposed:**
```python
if 'BOMB_EXPLODED' in events and 'CRATE_DESTROYED' in events:
    crates_destroyed_this_explosion = count_crates_destroyed_recently()
    if crates_destroyed_this_explosion >= 3:
        reward += MULTI_CRATE_BONUS * crates_destroyed_this_explosion
```

#### 3. **Positioning Reward** ⚠️ LOW PRIORITY
**Missing:** Reward for positioning near crates (without bombing yet)

**Proposed:**
```python
NEAR_CRATES_REWARD = 2.0  # Small reward for being near destructible crates

crates_adjacent = count_adjacent_crates(game_state, (x, y))
if crates_adjacent > 0:
    reward += NEAR_CRATES_REWARD * crates_adjacent
```

---

## Recommendations

### HIGH PRIORITY (Implement Now) 🔴

1. **Fix escape check in action masking**
   - Replace simple adjacent tile check with proper BFS path finding
   - Ensure agent can actually reach safety before bomb explodes
   - Location: `callbacks.py` line 111-121

2. **Add escape route features**
   - Add to global features: `safe_neighbors_count`, `min_distance_to_safety`
   - Helps agent learn when bombing is safe
   - Location: `callbacks.py` `build_feature_vector()` line 168-178

3. **Add escape progress reward**
   - Reward each step that moves toward safety after bomb placement
   - Critical for learning the escape behavior
   - Location: `train.py` `game_events_occurred()` around line 240-250

### MEDIUM PRIORITY (Consider After 3k More Rounds) 🟡

4. **Add crate density features**
   - `crates_in_bomb_range`, `crates_nearby_count`
   - Helps agent learn to target high-value positions
   - Location: `callbacks.py` `build_feature_vector()` line 168-178

5. **Add explicit bomb timer**
   - `nearest_bomb_timer`, `just_placed_bomb` flag
   - Improves temporal understanding
   - Location: `callbacks.py` `build_feature_vector()` line 168-178

### LOW PRIORITY (Maybe Later) 🟢

6. **Frame stacking** - Adds complexity, likely not needed
7. **Multi-crate bonus** - Nice-to-have but not critical
8. **Positioning rewards** - Could encourage loitering

---

## Data Analysis

### Training Data Quality

**Current training:** 6000 rounds with 42.5% self-kill rate
- ✅ **Sufficient volume:** 6000+ episodes is good sample size
- ❌ **Poor quality:** 42.5% deaths means agent learning wrong behaviors
- ❌ **Negative reward:** -58.7 average means policy is suboptimal

**Recommendation:** Continue training with new rewards, but **critically** need better escape mechanism

### Data Distribution Issues

**Observed behavior:**
- Bombs placed: 1.07 per game (very low)
- Crates destroyed: 1.53 per game (very low)
- Survival time: 260 steps (65% of round)

**Problem:** Agent is **under-exploring** the bombing action space
- Not enough positive examples of successful bombing
- Too many negative examples of failed bombing
- Agent converges to "safe" policy (no bombing)

**Solution:**
1. Fix escape check (HIGH PRIORITY #1)
2. Add escape progress reward (HIGH PRIORITY #3)
3. These will enable agent to **safely explore** bombing more

---

## Summary

### Critical Missing Features:
1. ⚠️ **Better escape route checking** (action masking bug)
2. ⚠️ **Escape route information** (features)
3. ⚠️ **Escape progress reward** (shaped reward)

### Implementation Priority:
1. **Fix action masking** (30 min) - MUST DO NOW
2. **Add escape features** (15 min) - SHOULD DO NOW
3. **Add escape reward** (10 min) - SHOULD DO NOW
4. **Continue training** with fixed code

### Expected Impact:
- Self-kill rate: 42.5% → 25-30% (after 3k rounds)
- Bombs placed: 1.07 → 2-3 per game
- Crates destroyed: 1.53 → 5-8 per game
- Avg reward: -58.7 → +50 to +150 (positive!)

**Bottom line:** The current feature set is mostly sufficient, but the **action masking has a critical bug** that allows the agent to drop bombs without safe escape routes. Fix this first, then add escape-focused features and rewards.
