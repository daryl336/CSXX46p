# Sequential Q-Learning → LLM Agent

A hybrid Bomberman agent that combines Q-learning's tactical expertise with LLM's strategic reasoning.

## Architecture

```
Game State → Q-Learning → LLM → Final Action
```

1. **Q-Learning**: Analyzes state and suggests action based on learned Q-values
2. **LLM**: Reviews suggestion and makes final decision (accept/modify/override)
3. **Safety Check**: Validates final action is safe

## Files

- `callbacks.py` - Main agent logic (setup, action selection, decision flow)
- `train.py` - Q-learning training (updates Q-table based on rewards)
- `helper.py` - Tactical analysis functions (used by LLM)

## Quick Start

### Play with LLM
```bash
python main.py play --agents q_llm --no-gui
```

### Play without LLM (Q-learning only)
Edit `callbacks.py` line 92:
```python
self.use_llm = False
```

### Train Q-learning component
```bash
python main.py play --agents q_llm --train 1 --n-rounds 1000 --no-gui
```

## LLM Integration

**Endpoint**: `http://0.0.0.0:6000` (POST)

**Request Format**:
```json
{
  "valid_movement": {...},
  "nearest_crate": {...},
  "check_bomb_radius": {...},
  "plant_bomb_available": {...},
  "coins_collection_policy": {...},
  "movement_history": [...],
  "q_learning_suggestion": {
    "recommended_action": "RIGHT",
    "q_values": {
      "UP": 2.5,
      "RIGHT": 5.2,
      "DOWN": 1.8,
      "LEFT": 0.5,
      "WAIT": 0.0,
      "BOMB": 3.1
    },
    "confidence": 4.7
  }
}
```

**Response Format**:
```json
{
  "action": "RIGHT",
  "reasoning": "Accepting Q-learning suggestion to collect nearby coin"
}
```

## Model Loading

Loads Q-learning models in priority order:
1. `agent_code/q_learning/my-saved-model.pt` (preferred)
2. `agent_code/q_llm/my-saved-model.pt` (fallback)
3. `my-saved-model.pt` (local)

## Key Features

- **Transparent Decision Making**: Logs show when LLM accepts vs overrides Q-learning
- **Fallback Safety**: Works even if LLM endpoint is unavailable
- **Complementary Strengths**: Q-learning speed + LLM reasoning
- **Metrics Tracking**: Full episode tracking for both training and play modes

## Example Output

```
[Q-LEARNING] Suggested action: RIGHT
[Q-LEARNING] Q-values: {'RIGHT': 5.2, 'UP': 2.5, ...}
[LLM] Received Q-suggestion: RIGHT
[LLM] Reasoning: Accepting coin collection strategy
[LLM] ✅ Accepted Q-learning suggestion
✅ Final action: RIGHT
```

When LLM overrides:
```
[LLM] 🔄 Modified suggestion: BOMB → WAIT
```
