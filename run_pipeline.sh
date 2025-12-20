#!/bin/bash
# ============================================================================
# 4D Gaussian Splatting Pipeline - All-in-One Execution Script
# ============================================================================
# This script runs the complete 4DGS pipeline with resume support.
# If the pipeline fails at any step, it can be resumed from where it left off.
#
# Usage:
#   ./run_pipeline.sh <input_folder> <experiment_name> [options]
#
# Examples:
#   # Run full pipeline
#   ./run_pipeline.sh data/inputs/my_video my_experiment
#
#   # Resume from training step
#   ./run_pipeline.sh data/inputs/my_video my_experiment --resume
#
#   # Skip COLMAP (for datasets with existing poses)
#   ./run_pipeline.sh data/inputs/n3dv_sample n3dv_test --skip_colmap
#
#   # Run with custom config
#   ./run_pipeline.sh data/inputs/my_video my_experiment --config configs/custom.yaml
#
# ============================================================================

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Pipeline steps
STEPS=("preprocess" "colmap" "convert" "train" "render")

# State file location
STATE_DIR="$SCRIPT_DIR/.pipeline_state"

# ============================================================================
# Logging Functions
# ============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN} STEP: $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

# ============================================================================
# State Management Functions
# ============================================================================
init_state_dir() {
    mkdir -p "$STATE_DIR"
}

get_state_file() {
    local experiment_name="$1"
    echo "$STATE_DIR/${experiment_name}.state"
}

save_state() {
    local experiment_name="$1"
    local current_step="$2"
    local status="$3"  # "completed" or "failed"
    local state_file=$(get_state_file "$experiment_name")

    cat > "$state_file" << EOF
EXPERIMENT_NAME=$experiment_name
LAST_STEP=$current_step
STATUS=$status
TIMESTAMP=$(date -Iseconds)
EOF
    log_info "Pipeline state saved: step=$current_step, status=$status"
}

load_state() {
    local experiment_name="$1"
    local state_file=$(get_state_file "$experiment_name")

    if [ -f "$state_file" ]; then
        source "$state_file"
        echo "$LAST_STEP:$STATUS"
    else
        echo ""
    fi
}

clear_state() {
    local experiment_name="$1"
    local state_file=$(get_state_file "$experiment_name")

    if [ -f "$state_file" ]; then
        rm "$state_file"
        log_info "Pipeline state cleared for experiment: $experiment_name"
    fi
}

get_resume_step_index() {
    local experiment_name="$1"
    local state=$(load_state "$experiment_name")

    if [ -z "$state" ]; then
        echo "0"
        return
    fi

    local last_step="${state%:*}"
    local status="${state#*:}"

    # Find index of the step
    for i in "${!STEPS[@]}"; do
        if [ "${STEPS[$i]}" = "$last_step" ]; then
            if [ "$status" = "completed" ]; then
                # Start from next step
                echo $((i + 1))
            else
                # Retry the failed step
                echo "$i"
            fi
            return
        fi
    done

    echo "0"
}

# ============================================================================
# Step Detection Functions
# ============================================================================
check_transforms_exists() {
    local experiment_dir="$1"
    local transforms_file="$experiment_dir/transforms.json"

    if [ -f "$transforms_file" ]; then
        return 0
    fi
    return 1
}

check_model_exists() {
    local model_dir="$1"

    # Check for checkpoint directories
    if ls "$model_dir/point_cloud/iteration_"* 1> /dev/null 2>&1; then
        return 0
    fi
    return 1
}

check_video_exists() {
    local output_dir="$1"
    local experiment_name="$2"

    if ls "$output_dir/$experiment_name/"*.mp4 1> /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ============================================================================
# Pipeline Execution
# ============================================================================
run_step() {
    local step="$1"
    local input_folder="$2"
    local experiment_name="$3"
    local extra_args="$4"

    log_step "$step"

    case "$step" in
        preprocess)
            python3 main.py \
                --input "$input_folder" \
                --experiment "$experiment_name" \
                --step preprocess \
                $extra_args
            ;;
        colmap)
            python3 main.py \
                --experiment "$experiment_name" \
                --step colmap \
                $extra_args
            ;;
        convert)
            python3 main.py \
                --experiment "$experiment_name" \
                --step convert \
                $extra_args
            ;;
        train)
            python3 main.py \
                --experiment "$experiment_name" \
                --step train \
                $extra_args
            ;;
        render)
            python3 main.py \
                --experiment "$experiment_name" \
                --step render \
                $extra_args
            ;;
        *)
            log_error "Unknown step: $step"
            return 1
            ;;
    esac
}

run_pipeline() {
    local input_folder="$1"
    local experiment_name="$2"
    local start_index="$3"
    local extra_args="$4"
    local skip_colmap="$5"

    local end_index=${#STEPS[@]}

    log_info "Starting pipeline from step index $start_index to $((end_index - 1))"

    for ((i=start_index; i<end_index; i++)); do
        local step="${STEPS[$i]}"

        # Skip COLMAP-related steps if requested
        if [ "$skip_colmap" = "true" ]; then
            if [ "$step" = "preprocess" ] || [ "$step" = "colmap" ] || [ "$step" = "convert" ]; then
                log_info "Skipping $step (--skip_colmap enabled)"
                save_state "$experiment_name" "$step" "completed"
                continue
            fi
        fi

        # Run the step
        log_info "Executing step: $step"
        save_state "$experiment_name" "$step" "running"

        if run_step "$step" "$input_folder" "$experiment_name" "$extra_args"; then
            save_state "$experiment_name" "$step" "completed"
            log_info "Step $step completed successfully"
        else
            save_state "$experiment_name" "$step" "failed"
            log_error "Step $step failed!"
            log_warn "To resume, run: $0 $input_folder $experiment_name --resume"
            return 1
        fi
    done

    # Clear state on successful completion
    clear_state "$experiment_name"
    log_info "Pipeline completed successfully!"
}

# ============================================================================
# Post-Processing
# ============================================================================
compress_outputs() {
    local experiment_name="$1"
    local output_dir="data/outputs/$experiment_name"

    if [ -d "$output_dir" ]; then
        log_info "Compressing outputs for download..."

        # Use the Python utility for compression
        python3 -c "
from src.utils import compress_output
compress_output('$output_dir', '$output_dir.tar.gz')
print('Compressed to: $output_dir.tar.gz')
"
        if [ $? -eq 0 ]; then
            log_info "Output compressed to: $output_dir.tar.gz"
        else
            log_warn "Compression failed, using tar directly..."
            tar -czvf "$output_dir.tar.gz" "$output_dir"
        fi
    fi
}

# ============================================================================
# Usage
# ============================================================================
print_usage() {
    echo "Usage: $0 <input_folder> <experiment_name> [options]"
    echo ""
    echo "Arguments:"
    echo "  input_folder      Path to folder containing input videos or images"
    echo "  experiment_name   Name for this experiment/run"
    echo ""
    echo "Options:"
    echo "  --resume          Resume from last failed/incomplete step"
    echo "  --restart         Clear state and start from beginning"
    echo "  --skip_colmap     Skip preprocessing and COLMAP (for datasets with poses)"
    echo "  --config FILE     Use custom configuration file"
    echo "  --mock            Use mock training (for testing)"
    echo "  --compress        Compress outputs after completion"
    echo "  --gpu INDEX       Specify GPU index (default: 0)"
    echo "  --iterations N    Training iterations (default: 30000)"
    echo "  --verbose         Enable verbose logging"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Full pipeline"
    echo "  $0 data/inputs/my_video my_experiment"
    echo ""
    echo "  # Resume from failure"
    echo "  $0 data/inputs/my_video my_experiment --resume"
    echo ""
    echo "  # Use existing camera poses"
    echo "  $0 data/inputs/n3dv_coffee coffee_test --skip_colmap"
    echo ""
    echo "  # Quick test with mock training"
    echo "  $0 data/inputs/test test_run --mock --compress"
}

# ============================================================================
# Main
# ============================================================================
main() {
    # Initialize state directory
    init_state_dir

    # Parse arguments
    if [ $# -lt 2 ]; then
        print_usage
        exit 1
    fi

    INPUT_FOLDER="$1"
    EXPERIMENT_NAME="$2"
    shift 2

    # Default options
    RESUME=false
    RESTART=false
    SKIP_COLMAP=false
    COMPRESS=false
    EXTRA_ARGS=""

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --resume)
                RESUME=true
                shift
                ;;
            --restart)
                RESTART=true
                shift
                ;;
            --skip_colmap)
                SKIP_COLMAP=true
                EXTRA_ARGS="$EXTRA_ARGS --skip_colmap"
                shift
                ;;
            --config)
                EXTRA_ARGS="$EXTRA_ARGS --config $2"
                shift 2
                ;;
            --mock)
                EXTRA_ARGS="$EXTRA_ARGS --mock"
                shift
                ;;
            --compress)
                COMPRESS=true
                shift
                ;;
            --gpu)
                export CUDA_VISIBLE_DEVICES="$2"
                shift 2
                ;;
            --iterations)
                EXTRA_ARGS="$EXTRA_ARGS --iterations $2"
                shift 2
                ;;
            --verbose|-v)
                EXTRA_ARGS="$EXTRA_ARGS --verbose"
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate input folder exists
    if [ ! -d "$INPUT_FOLDER" ] && [ "$SKIP_COLMAP" = "false" ]; then
        log_error "Input folder does not exist: $INPUT_FOLDER"
        exit 1
    fi

    # Handle restart
    if [ "$RESTART" = "true" ]; then
        log_info "Clearing previous state for experiment: $EXPERIMENT_NAME"
        clear_state "$EXPERIMENT_NAME"
    fi

    # Determine starting step
    START_INDEX=0

    if [ "$RESUME" = "true" ]; then
        START_INDEX=$(get_resume_step_index "$EXPERIMENT_NAME")

        if [ "$START_INDEX" -ge "${#STEPS[@]}" ]; then
            log_info "Pipeline already completed for experiment: $EXPERIMENT_NAME"
            log_info "Use --restart to run again from the beginning"
            exit 0
        fi

        if [ "$START_INDEX" -gt 0 ]; then
            log_info "Resuming from step: ${STEPS[$START_INDEX]}"
        else
            log_info "No previous state found, starting from beginning"
        fi
    fi

    # Print configuration
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE} 4D Gaussian Splatting Pipeline${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo "  Input:        $INPUT_FOLDER"
    echo "  Experiment:   $EXPERIMENT_NAME"
    echo "  Start Step:   ${STEPS[$START_INDEX]}"
    echo "  Skip COLMAP:  $SKIP_COLMAP"
    echo "  Compress:     $COMPRESS"
    echo "  GPU:          ${CUDA_VISIBLE_DEVICES:-0}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""

    # Record start time
    PIPELINE_START=$(date +%s)

    # Run the pipeline
    if run_pipeline "$INPUT_FOLDER" "$EXPERIMENT_NAME" "$START_INDEX" "$EXTRA_ARGS" "$SKIP_COLMAP"; then
        # Calculate elapsed time
        PIPELINE_END=$(date +%s)
        ELAPSED=$((PIPELINE_END - PIPELINE_START))
        HOURS=$((ELAPSED / 3600))
        MINUTES=$(((ELAPSED % 3600) / 60))
        SECONDS=$((ELAPSED % 60))

        echo ""
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN} PIPELINE COMPLETED SUCCESSFULLY${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo "  Experiment: $EXPERIMENT_NAME"
        echo "  Elapsed:    ${HOURS}h ${MINUTES}m ${SECONDS}s"
        echo -e "${GREEN}============================================================${NC}"

        # Compress outputs if requested
        if [ "$COMPRESS" = "true" ]; then
            compress_outputs "$EXPERIMENT_NAME"
        fi

        # Print output locations
        echo ""
        echo "Output locations:"
        echo "  Model:  data/outputs/$EXPERIMENT_NAME/model/"
        echo "  Video:  data/outputs/$EXPERIMENT_NAME/renders/"
        if [ "$COMPRESS" = "true" ]; then
            echo "  Archive: data/outputs/$EXPERIMENT_NAME.tar.gz"
        fi

    else
        log_error "Pipeline failed!"
        log_warn "To resume from the failed step, run:"
        log_warn "  $0 $INPUT_FOLDER $EXPERIMENT_NAME --resume"
        exit 1
    fi
}

# Run main function
main "$@"
