#!/bin/bash
# ============================================================================
# 4DGS Quick Start Script
# ============================================================================
# 간단한 실행 스크립트 - 빠른 시작용
#
# Usage:
#   ./quick_start.sh <dataset_name> [experiment_name]
#
# Examples:
#   ./quick_start.sh sample_synthetic
#   ./quick_start.sh coffee my_coffee_test
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [experiment_name]"
    echo ""
    echo "Available datasets:"
    ls -1 data/inputs/ 2>/dev/null | grep -v "^\.gitkeep$" || echo "  (no datasets found)"
    echo ""
    echo "Examples:"
    echo "  $0 sample_synthetic"
    echo "  $0 coffee my_coffee_test"
    exit 1
fi

DATASET_NAME="$1"
EXPERIMENT_NAME="${2:-${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)}"
INPUT_FOLDER="data/inputs/$DATASET_NAME"

# Validate input
if [ ! -d "$INPUT_FOLDER" ]; then
    echo -e "${YELLOW}Dataset not found: $INPUT_FOLDER${NC}"
    echo ""
    echo "Available datasets:"
    ls -1 data/inputs/ 2>/dev/null | grep -v "^\.gitkeep$" || echo "  (no datasets found)"
    echo ""
    echo "To download a dataset, run:"
    echo "  ./download_sample_data.sh <dataset_name>"
    exit 1
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE} 4DGS Quick Start${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "  Dataset:     $DATASET_NAME"
echo "  Input:       $INPUT_FOLDER"
echo "  Experiment:  $EXPERIMENT_NAME"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if transforms.json exists
if [ -f "$INPUT_FOLDER/transforms.json" ]; then
    echo -e "${GREEN}✓ Found transforms.json - will skip COLMAP${NC}"
    SKIP_COLMAP="--skip_colmap"
else
    echo -e "${YELLOW}⚠ No transforms.json found - will run COLMAP${NC}"
    SKIP_COLMAP=""
fi

echo ""
echo "Starting pipeline..."
echo ""

# Run pipeline
./run_pipeline.sh "$INPUT_FOLDER" "$EXPERIMENT_NAME" $SKIP_COLMAP

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Pipeline Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Output locations:"
echo "  Model:  data/outputs/$EXPERIMENT_NAME/model/"
echo "  Video:  data/outputs/$EXPERIMENT_NAME/renders/"
echo ""
echo "To render custom camera path, see RENDER_GUIDE.md"
echo ""

