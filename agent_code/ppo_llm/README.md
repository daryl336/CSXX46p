# Sequential PPO → LLM Agent

A hybrid Bomberman agent that combines PPO's deep learning policy with LLM's strategic reasoning.

## Architecture

```
Game State → PPO Network → LLM → Final Action
```

1. **PPO**: Neural network analyzes state and suggests action with probability distribution
2. **LLM**: Reviews PPO's suggestion and action probabilities, makes final decision
3. **Safety Check**: Validates final action is safe

## Files

- `callbacks.py` - Main agent logic (setup, action selection, sequential decision flow)
- `train.py` - PPO training (updates neural network based on rewards and LLM's choices)
- `helper.py` - Tactical analysis functions (used by LLM)

## Quick Start

### Play with LLM
```bash
python main.py play --agents ppo_llm --no-gui
```

### Play without LLM (PPO only)
Edit `callbacks.py` line 92:
```python
self.use_llm = False
```

### Train PPO component
```bash
python main.py play --agents ppo_llm --train 1 --n-rounds 1000 --no-gui
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
  "ppo_suggestion": {
    "recommended_action": "RIGHT",
    "action_probabilities": {
      "UP": 0.05,
      "RIGHT": 0.68,
      "DOWN": 0.12,
      "LEFT": 0.03,
      "WAIT": 0.02,
      "BOMB": 0.10
    },
    "value_estimate": 0.45,
    "confidence": 0.82,
    "top_3_actions": [
      {"action": "RIGHT", "probability": 0.68},
      {"action": "DOWN", "probability": 0.12},
      {"action": "BOMB", "probability": 0.10}
    ]
  }
}
```

**Response Format**:
```json
{
  "action": "RIGHT",
  "reasoning": "PPO shows 68% confidence in RIGHT for coin collection. Accepting neural network's learned policy."
}
```

## Model Loading

Loads PPO models in priority order:
1. `models/ppo_llm_agent.pth` (ppo_llm model - preferred)
2. `agent_code/ppo/models/ppo_agent.pth` (base PPO model - fallback)
3. `models/ppo_agent.pth` (alternative location)

## Key Features

- **Neural Network Policy**: PPO provides learned action probabilities
- **Confidence Metrics**: Entropy-based confidence scores for LLM
- **Transparent Decisions**: Logs show acceptance vs override rates
- **Training on LLM Choices**: PPO learns from LLM's final decisions
- **Fallback Safety**: Works even if LLM endpoint is unavailable

## PPO vs Q-Learning

Unlike `q_llm` (Q-values), this agent provides:
- **Action Probabilities**: Full distribution instead of single Q-values
- **Value Estimates**: State value prediction from critic network
- **Confidence Scores**: Entropy-based measure of policy certainty
- **Continuous Learning**: Neural network adapts smoothly

## Example Output

```
[PPO] Suggested action: RIGHT
[PPO] Action probabilities: {UP=5%, RIGHT=68%, DOWN=12%, LEFT=3%, WAIT=2%, BOMB=10%}
[PPO] Value estimate: 0.450
[PPO] Selected RIGHT with confidence 68%
[LLM] Received PPO suggestion: RIGHT (conf: 82%)
[LLM] Reasoning: PPO's high confidence aligns with coin collection. Accepting.
[LLM] ✅ Accepted PPO suggestion (accept rate: 42/50)
✅ Final action: RIGHT
```

When LLM overrides:
```
[LLM] 🔄 Overrode PPO: BOMB → WAIT (override rate: 8/50)
```

## Training Benefits

Training with LLM's decisions allows PPO to:
1. Learn from strategic overrides
2. Align policy with high-level reasoning
3. Improve beyond pure reward signals
4. Bootstrap from LLM's knowledge
