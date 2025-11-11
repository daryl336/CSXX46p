#!/bin/bash

# Easy Mode PPO Training Script
# Trains one PPO agent against no opponents
# Good for: Testing, debugging, learning basic survival and coin collection

set -e

# Configuration
ROUNDS=${1:-2000}
NO_GUI=${2:---no-gui}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PPO Easy Mode Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Training Rounds: $ROUNDS"
echo "  GUI: $([ "$NO_GUI" == "--no-gui" ] && echo "Disabled (faster)" || echo "Enabled")"
echo "  No Opponents (easy mode)"
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start training
echo -e "${GREEN}Starting easy mode training...${NC}"
echo ""

python -m main play \
    --agents ppo_final \
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
echo -e "${YELLOW}Next steps:${NC}"
echo "  Once the agent performs well in easy mode, train against"
echo "  rule_based_agent opponents using: ./train_single.sh"
echo ""
