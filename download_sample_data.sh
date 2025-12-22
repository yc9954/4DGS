#!/bin/bash
# ============================================================================
# 4D Gaussian Splatting Pipeline - Sample Data Downloader
# ============================================================================
# This script downloads public 4D datasets for testing the pipeline.
# It supports multiple datasets and automatically organizes them for our
# pipeline format.
#
# Supported Datasets:
#   - N3DV (Neural 3D Video): Multi-view video sequences (coffee, flame_salmon, etc.)
#   - DyNeRF: Dynamic NeRF sequences
#   - D-NeRF: Synthetic dynamic scenes
#
# Usage:
#   ./download_sample_data.sh [dataset_name] [options]
#
# Examples:
#   ./download_sample_data.sh                    # Downloads default sample
#   ./download_sample_data.sh coffee             # Downloads N3DV coffee scene
#   ./download_sample_data.sh flame_salmon       # Downloads N3DV flame_salmon scene
#   ./download_sample_data.sh --list             # List available datasets
#
# ============================================================================

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/inputs"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# ============================================================================
# Dataset Definitions
# ============================================================================

# N3DV Dataset URLs (Google Drive / Dropbox mirrors)
# Note: These are placeholder URLs - replace with actual dataset URLs
declare -A N3DV_DATASETS=(
    ["coffee"]="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip"
    ["flame_salmon"]="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon.zip"
    ["cook_spinach"]="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cook_spinach.zip"
    ["cut_beef"]="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cut_beef.zip"
    ["sear_steak"]="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/sear_steak.zip"
)

# D-NeRF synthetic datasets (commonly available)
declare -A DNERF_DATASETS=(
    ["bouncing_balls"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/bouncing_balls.zip"
    ["lego"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/lego.zip"
    ["jumping_jacks"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/jumping_jacks.zip"
    ["hook"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/hook.zip"
    ["hellwarrior"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/hellwarrior.zip"
    ["mutant"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/mutant.zip"
    ["standup"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/standup.zip"
    ["trex"]="https://github.com/albertpumarola/D-NeRF/releases/download/data/trex.zip"
)

# Sample mini datasets for quick testing
declare -A SAMPLE_DATASETS=(
    ["sample_synthetic"]="synthetic"  # Will generate a small test dataset
)

# ============================================================================
# Download Functions
# ============================================================================

download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if wget -q --show-progress -O "$output" "$url"; then
            return 0
        elif curl -L -o "$output" "$url" 2>/dev/null; then
            return 0
        fi

        retry=$((retry + 1))
        log_warn "Download failed, attempt $retry/$max_retries..."
        sleep 2
    done

    return 1
}

# ============================================================================
# Dataset Processing Functions
# ============================================================================

convert_n3dv_format() {
    local src_dir="$1"
    local dst_dir="$2"
    local dataset_name="$3"

    log_info "Converting N3DV format to pipeline format..."

    # N3DV structure: dataset/cam{00-20}/images/*.png + poses_bounds.npy
    # Our structure: dataset/transforms.json + images/

    mkdir -p "$dst_dir"

    # Check if poses_bounds.npy exists (LLFF format)
    if [ -f "$src_dir/poses_bounds.npy" ]; then
        # This is LLFF format - convert to transforms.json
        log_info "Found LLFF format (poses_bounds.npy), converting..."

        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from src.preprocess import DatasetAdapter

adapter = DatasetAdapter()
adapter.convert_llff_to_transforms(
    '$src_dir',
    '$dst_dir'
)
print('Conversion complete!')
"
    elif [ -d "$src_dir/cam00" ]; then
        # Multi-camera N3DV format
        log_info "Found multi-camera N3DV format, adapting..."

        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from src.preprocess import DatasetAdapter

adapter = DatasetAdapter()
adapter.adapt_n3dv_dataset(
    '$src_dir',
    '$dst_dir'
)
print('Conversion complete!')
"
    else
        # Unknown format - just copy
        log_warn "Unknown N3DV format, copying as-is..."
        cp -r "$src_dir"/* "$dst_dir/"
    fi

    # Create a config file for this dataset
    create_dataset_config "$dst_dir" "$dataset_name" "n3dv"
}

convert_dnerf_format() {
    local src_dir="$1"
    local dst_dir="$2"
    local dataset_name="$3"

    log_info "Converting D-NeRF format to pipeline format..."

    # D-NeRF already has transforms_train.json format
    mkdir -p "$dst_dir"

    # Check for D-NeRF format (transforms_train.json)
    if [ -f "$src_dir/transforms_train.json" ]; then
        log_info "Found D-NeRF format (transforms_train.json), adapting..."

        python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from src.preprocess import DatasetAdapter

adapter = DatasetAdapter()
adapter.adapt_dnerf_dataset(
    '$src_dir',
    '$dst_dir'
)
print('Conversion complete!')
"
    else
        log_warn "Standard D-NeRF transforms not found, copying as-is..."
        cp -r "$src_dir"/* "$dst_dir/"
    fi

    create_dataset_config "$dst_dir" "$dataset_name" "dnerf"
}

create_dataset_config() {
    local dataset_dir="$1"
    local dataset_name="$2"
    local dataset_type="$3"

    local config_file="$SCRIPT_DIR/configs/${dataset_name}.yaml"

    log_info "Creating config file: $config_file"

    cat > "$config_file" << EOF
# 4DGS Pipeline Configuration for ${dataset_name}
# Auto-generated by download_sample_data.sh
# Dataset type: ${dataset_type}

# === Paths ===
input_dir: data/inputs/${dataset_name}
processed_dir: data/processed
output_dir: data/outputs

# === Experiment ===
experiment_name: ${dataset_name}
skip_existing: true
verbose: true

# === Frame Extraction ===
# Skip frame extraction for pre-extracted datasets
frame_extraction:
  fps: 30.0
  quality: 95
  format: png
  start_time: 0.0
  duration: null
  resize_width: null
  resize_height: null

# === COLMAP Settings ===
# Most downloaded datasets come with poses, so COLMAP may be skipped
colmap:
  camera_model: OPENCV
  single_camera: false
  single_camera_per_folder: true
  matcher_type: sequential
  use_gpu: true
  gpu_index: 0
  max_image_size: 1600
  num_threads: -1

# === Training Settings ===
training:
  iterations: 30000
  save_iterations:
    - 7000
    - 15000
    - 30000
  test_iterations:
    - 7000
    - 15000
    - 30000
  resolution: -1
  white_background: false

# === Rendering Settings ===
render:
  output_format: mp4
  fps: 30
  codec: libx264
  crf: 18
  camera_path_type: interpolate
  num_frames: 300
  width: null
  height: null
EOF
}

# ============================================================================
# Generate Synthetic Test Data
# ============================================================================

generate_synthetic_sample() {
    local output_dir="$1"

    log_info "Generating synthetic test dataset..."

    mkdir -p "$output_dir/images"

    # Generate a simple synthetic dataset using Python
    python3 << EOF
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image

output_dir = Path("$output_dir")
images_dir = output_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Generate 50 frames with 4 camera views
num_frames = 50
num_cameras = 4

# Camera parameters
fov = 50.0
width, height = 400, 400

# Generate camera transforms
frames = []

for cam_idx in range(num_cameras):
    # Camera position on a circle around the origin
    angle = (cam_idx / num_cameras) * 2 * np.pi
    radius = 4.0

    cam_x = radius * np.cos(angle)
    cam_y = 0.5
    cam_z = radius * np.sin(angle)

    # Look at origin
    forward = np.array([0, 0, 0]) - np.array([cam_x, cam_y, cam_z])
    forward = forward / np.linalg.norm(forward)

    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Build 4x4 transform matrix (camera-to-world)
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = -forward
    transform[:3, 3] = [cam_x, cam_y, cam_z]

    for frame_idx in range(num_frames):
        # Generate a simple colored gradient image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Time-varying color
        t = frame_idx / num_frames
        r = int(128 + 127 * np.sin(2 * np.pi * t + cam_idx * 0.5))
        g = int(128 + 127 * np.sin(2 * np.pi * t * 1.5 + cam_idx * 0.7))
        b = int(128 + 127 * np.sin(2 * np.pi * t * 2 + cam_idx * 0.3))

        # Create gradient
        for y in range(height):
            for x in range(width):
                img[y, x] = [
                    int(r * (1 - y/height) + 64),
                    int(g * (x/width)),
                    int(b * (y/height))
                ]

        # Add a moving circle
        cx = int(width/2 + 100 * np.sin(2 * np.pi * t))
        cy = int(height/2 + 50 * np.cos(2 * np.pi * t * 2))
        for y in range(max(0, cy-30), min(height, cy+30)):
            for x in range(max(0, cx-30), min(width, cx+30)):
                if (x-cx)**2 + (y-cy)**2 < 30**2:
                    img[y, x] = [255, 255, 255]

        # Save image
        img_name = f"cam{cam_idx:02d}_frame{frame_idx:04d}.png"
        Image.fromarray(img).save(images_dir / img_name)

        # Add to frames list
        frames.append({
            "file_path": f"images/{img_name}",
            "transform_matrix": transform.tolist(),
            "time": t,
            "camera_id": cam_idx
        })

# Create transforms.json
transforms = {
    "camera_angle_x": fov * np.pi / 180,
    "camera_angle_y": fov * np.pi / 180 * height / width,
    "fl_x": width / (2 * np.tan(fov * np.pi / 360)),
    "fl_y": height / (2 * np.tan(fov * np.pi / 360)),
    "cx": width / 2,
    "cy": height / 2,
    "w": width,
    "h": height,
    "frames": frames
}

with open(output_dir / "transforms.json", 'w') as f:
    json.dump(transforms, f, indent=2)

print(f"Generated {len(frames)} frames in {output_dir}")
EOF

    create_dataset_config "$output_dir" "sample_synthetic" "synthetic"
    log_info "Synthetic sample dataset created at: $output_dir"
}

# ============================================================================
# Main Download Logic
# ============================================================================

list_datasets() {
    log_section "Available Datasets"

    echo ""
    echo "N3DV (Neural 3D Video) Datasets:"
    for dataset in "${!N3DV_DATASETS[@]}"; do
        echo "  - $dataset"
    done

    echo ""
    echo "D-NeRF Synthetic Datasets:"
    for dataset in "${!DNERF_DATASETS[@]}"; do
        echo "  - $dataset"
    done

    echo ""
    echo "Sample/Test Datasets:"
    echo "  - sample_synthetic (generates synthetic test data)"
    echo ""
    echo "Usage: $0 <dataset_name>"
}

download_dataset() {
    local dataset_name="$1"
    local url=""
    local dataset_type=""

    # Check which collection the dataset belongs to
    if [[ -v N3DV_DATASETS[$dataset_name] ]]; then
        url="${N3DV_DATASETS[$dataset_name]}"
        dataset_type="n3dv"
    elif [[ -v DNERF_DATASETS[$dataset_name] ]]; then
        url="${DNERF_DATASETS[$dataset_name]}"
        dataset_type="dnerf"
    elif [ "$dataset_name" = "sample_synthetic" ]; then
        dataset_type="synthetic"
    else
        log_error "Unknown dataset: $dataset_name"
        echo ""
        list_datasets
        exit 1
    fi

    # Create data directory
    mkdir -p "$DATA_DIR"

    local output_dir="$DATA_DIR/$dataset_name"

    # Check if already exists
    if [ -d "$output_dir" ] && [ -f "$output_dir/transforms.json" ]; then
        log_info "Dataset already exists at: $output_dir"
        log_info "Use --force to re-download"
        return 0
    fi

    # Handle synthetic sample
    if [ "$dataset_type" = "synthetic" ]; then
        generate_synthetic_sample "$output_dir"
        return 0
    fi

    # Download the dataset
    log_section "Downloading $dataset_name"

    local archive_path="$DATA_DIR/${dataset_name}.zip"
    local temp_dir="$DATA_DIR/${dataset_name}_temp"

    log_info "URL: $url"
    log_info "Destination: $output_dir"

    if ! download_with_retry "$url" "$archive_path"; then
        log_error "Failed to download dataset"
        log_warn "The dataset may require manual download."
        log_warn "Please visit the dataset website and download manually."

        # Provide fallback instructions
        echo ""
        echo "Manual download instructions:"
        if [ "$dataset_type" = "n3dv" ]; then
            echo "  1. Visit: https://github.com/facebookresearch/Neural_3D_Video"
            echo "  2. Download the $dataset_name dataset"
            echo "  3. Extract to: $output_dir"
        elif [ "$dataset_type" = "dnerf" ]; then
            echo "  1. Visit: https://github.com/albertpumarola/D-NeRF"
            echo "  2. Download the $dataset_name dataset"
            echo "  3. Extract to: $output_dir"
        fi
        exit 1
    fi

    # Extract archive
    log_info "Extracting archive..."
    mkdir -p "$temp_dir"

    if [[ "$archive_path" == *.zip ]]; then
        unzip -q "$archive_path" -d "$temp_dir"
    elif [[ "$archive_path" == *.tar.gz ]] || [[ "$archive_path" == *.tgz ]]; then
        tar -xzf "$archive_path" -C "$temp_dir"
    elif [[ "$archive_path" == *.tar ]]; then
        tar -xf "$archive_path" -C "$temp_dir"
    fi

    # Find the actual data directory (might be nested)
    local src_dir="$temp_dir"
    if [ $(ls -1 "$temp_dir" | wc -l) -eq 1 ]; then
        src_dir="$temp_dir/$(ls -1 "$temp_dir")"
    fi

    # Convert to our format
    log_info "Converting dataset format..."

    if [ "$dataset_type" = "n3dv" ]; then
        convert_n3dv_format "$src_dir" "$output_dir" "$dataset_name"
    elif [ "$dataset_type" = "dnerf" ]; then
        convert_dnerf_format "$src_dir" "$output_dir" "$dataset_name"
    fi

    # Cleanup
    log_info "Cleaning up temporary files..."
    rm -rf "$temp_dir"
    rm -f "$archive_path"

    log_info "Dataset ready at: $output_dir"
    log_info "Config file created at: $SCRIPT_DIR/configs/${dataset_name}.yaml"
}

# ============================================================================
# Usage
# ============================================================================

print_usage() {
    echo "Usage: $0 [dataset_name] [options]"
    echo ""
    echo "Download and prepare public 4D datasets for the 4DGS pipeline."
    echo ""
    echo "Arguments:"
    echo "  dataset_name    Name of the dataset to download (see --list)"
    echo ""
    echo "Options:"
    echo "  --list          List available datasets"
    echo "  --force         Re-download even if dataset exists"
    echo "  --output DIR    Custom output directory"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Download default synthetic sample"
    echo "  $0 sample_synthetic       # Generate synthetic test data"
    echo "  $0 bouncing_balls         # Download D-NeRF bouncing_balls"
    echo "  $0 coffee                 # Download N3DV coffee scene"
    echo "  $0 --list                 # Show all available datasets"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Default dataset
    DATASET_NAME="sample_synthetic"
    FORCE=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --list|-l)
                list_datasets
                exit 0
                ;;
            --force|-f)
                FORCE=true
                shift
                ;;
            --output|-o)
                DATA_DIR="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
            *)
                DATASET_NAME="$1"
                shift
                ;;
        esac
    done

    log_section "4DGS Sample Data Downloader"

    # Handle force flag
    if [ "$FORCE" = true ]; then
        rm -rf "$DATA_DIR/$DATASET_NAME"
    fi

    # Download the dataset
    download_dataset "$DATASET_NAME"

    echo ""
    log_info "Next steps:"
    echo "  1. Run the pipeline with:"
    echo "     ./run_pipeline.sh data/inputs/$DATASET_NAME $DATASET_NAME --skip_colmap"
    echo ""
    echo "  2. Or use the config file:"
    echo "     python main.py --config configs/${DATASET_NAME}.yaml --skip_colmap"
}

main "$@"
