#!/bin/bash

# Simple Single-Agent PPO Training Script
# Trains one PPO agent against rule-based opponents

set -e

# Configuration
ROUNDS=${1:-1000}
NO_GUI=${2:---no-gui}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PPO Single-Agent Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Training Rounds: $ROUNDS"
echo "  GUI: $([ "$NO_GUI" == "--no-gui" ] && echo "Disabled (faster)" || echo "Enabled")"
echo "  Opponents: 3x rule_based_agent"
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start training
echo -e "${GREEN}Starting single-agent training...${NC}"
echo ""

python -m main play \
    --agents ppo_final rule_based_agent rule_based_agent rule_based_agent \
    --train 1 \
    --n-rounds "$ROUNDS" \
    $NO_GUI

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Check the following for results:"
echo "  - Model: agent_code/ppo_final/models/ppo_agent.pth"
echo "  - Logs: agent_code/ppo_final/logs/"
echo "  - Plots: agent_code/ppo_final/logs/training_progress.png"
echo ""
