Setup environment:

```python
pip install -r requirements.txt
```

Train agent:

```python
python -m main play \
  --agents dqn_torch rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --n-round 50000 \
  --no-gui

python -m main play \
  --agents ppo rule_based_agent rule_based_agent rule_based_agent \
  --train 1 \
  --n-round 50000 \
  --no-gui

python -m main play \
  --agents ppo \
  --train 1 \
  --n-round 1000 \
  --no-gui

python -m main play \
  --agents q_learning \
  --train 1 \
  --n-round 2000 \
  --no-gui

python -m main play \
  --agents q_learning \
  --train 1 \
  --n-round 8000 \
  --no-gui

python -m main play \
  --agents q_learning rule_based_agent \
  --train 1 \
  --n-round 20000 \
  --no-gui
```

```
Coin Collection Phase:
python -m main play \
  --agents bbman \
  --train 1 \
  --n-round 10000 \
  --no-gui

Bomb Placement Phase:
python -m main play \
  --agents bbman \
  --train 1 \
  --n-round 1000 \
  --no-gui
```

Play agent:

```python
python -m main play --agents rule_based_agent
python -m main play --agents llm
python -m main play --agents q_learning
python -m main play --agents ppo
python -m main play --agents maverick_enhanced
python -m main play --agents llm_maverick_v3
```

python -m main play \
  --agents llm_maverick_v2 rule_based_agent rule_based_agent rule_based_agent\
  --train 1 \
  --n-round 1000 \
  --no-gui

python -m main play --agents q_learning ppo dqn_final maverick

python run_individual_evaluation.py --agents q_learning ppo dqn_final maverick --n-rounds 10
python run_individual_evaluation.py --agents q_llm ppo_llm llm_maverick --n-rounds 1