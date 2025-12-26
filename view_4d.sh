#!/bin/bash
# ============================================================================
# 4D Gaussian Splatting Interactive Viewer Launcher
# ============================================================================
# Quick launcher for the 4D interactive viewer.
#
# Usage:
#   ./view_4d.sh <experiment_name>
#   ./view_4d.sh my_experiment
#   ./view_4d.sh my_experiment --time 0.5
#
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage:${NC} $0 <experiment_name> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 my_experiment"
    echo "  $0 my_experiment --time 0.5"
    echo "  $0 my_experiment --width 1920 --height 1080"
    echo ""
    echo "Options:"
    echo "  --time, -t      Initial time (0.0-1.0)"
    echo "  --width         Window width"
    echo "  --height        Window height"
    echo ""
    exit 1
fi

EXPERIMENT_NAME="$1"
shift

# Find model path
MODEL_PATH="data/outputs/$EXPERIMENT_NAME/model"

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}[ERROR]${NC} Model not found: $MODEL_PATH"
    echo ""
    echo "Available experiments:"
    ls -1 data/outputs/ 2>/dev/null || echo "  (none)"
    echo ""
    echo "Train a model first with:"
    echo "  ./run_pipeline.sh data/inputs/my_video $EXPERIMENT_NAME"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Starting 4D Viewer for: $EXPERIMENT_NAME"
echo -e "${GREEN}[INFO]${NC} Model path: $MODEL_PATH"
echo ""

# Run viewer
python3 src/viewer_4d.py --model "$MODEL_PATH" "$@"
