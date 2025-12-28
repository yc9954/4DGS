#!/bin/bash
# ============================================================================
# 4D Gaussian Splatting Pipeline - RunPod Environment Setup
# ============================================================================
# This script automates the setup of a RunPod/Ubuntu docker container for
# running the 4DGS pipeline. It installs all required dependencies and
# prepares the environment for training.
#
# Usage:
#   chmod +x setup_runpod.sh
#   ./setup_runpod.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Determine if we need sudo (RunPod runs as root, so no sudo needed)
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        log_error "Not running as root and sudo is not available."
        exit 1
    fi
fi

# Check if running as root or with sudo
check_permissions() {
    if [ "$EUID" -eq 0 ]; then
        log_info "Running as root (no sudo needed)"
    elif [ -n "$SUDO" ]; then
        log_info "Using sudo for privileged operations"
    fi
}

# ============================================================================
# System Package Installation
# ============================================================================
install_system_packages() {
    log_section "Installing System Packages"

    # Update package lists
    log_info "Updating apt package lists..."
    $SUDO apt-get update -qq

    # Install core dependencies
    log_info "Installing core dependencies..."
    $SUDO apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        p7zip-full \
        software-properties-common

    # Install FFmpeg
    log_info "Installing FFmpeg..."
    $SUDO apt-get install -y ffmpeg

    # Install OpenCV dependencies (headless)
    log_info "Installing OpenCV/OpenGL dependencies..."
    $SUDO apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1

    # Install COLMAP and its dependencies
    log_info "Installing COLMAP..."
    $SUDO apt-get install -y colmap

    log_info "System packages installed successfully."
}

# ============================================================================
# CUDA Version Check
# ============================================================================
check_cuda() {
    log_section "Checking CUDA Configuration"

    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Is NVIDIA driver installed?"
        return 1
    fi

    # Get CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    log_info "NVIDIA Driver Version: $CUDA_VERSION"

    # Check nvcc (CUDA compiler)
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
        log_info "NVCC Version: $NVCC_VERSION"
    else
        log_warn "nvcc not found in PATH. You may need to set CUDA_HOME."
    fi

    # Print GPU info
    log_info "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

    # Check COLMAP GPU support
    log_info "Checking COLMAP GPU support..."
    if colmap -h 2>&1 | grep -q "SiftExtraction.use_gpu"; then
        log_info "COLMAP has GPU support available."
    else
        log_warn "COLMAP may not have GPU support compiled. Feature matching will use CPU."
    fi

    return 0
}

# ============================================================================
# Python Environment Setup
# ============================================================================
setup_python_env() {
    log_section "Setting Up Python Environment"

    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"

    # Ensure pip is up to date
    log_info "Upgrading pip..."
    python3 -m pip install --upgrade pip setuptools wheel --ignore-installed 2>/dev/null || \
    python3 -m pip install --upgrade pip setuptools wheel

    # Detect CUDA version and install appropriate PyTorch
    log_info "Detecting CUDA version for PyTorch installation..."

    CUDA_VERSION=""
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    elif [ -f /usr/local/cuda/version.txt ]; then
        CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed -n 's/.*CUDA Version \([0-9]*\.[0-9]*\).*/\1/p')
    fi

    log_info "Detected CUDA version: ${CUDA_VERSION:-unknown}"

    # Install PyTorch based on CUDA version
    # RTX 4090 typically uses CUDA 11.8 or 12.x
    if [[ "$CUDA_VERSION" == 12.* ]]; then
        log_info "Installing PyTorch for CUDA 12.1..."
        python3 -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
            --extra-index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.8* ]] || [[ "$CUDA_VERSION" == "" ]]; then
        # Default: CUDA 11.8 (most compatible with RTX 4090)
        log_info "Installing PyTorch for CUDA 11.8 (RTX 4090 compatible)..."
        python3 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
            --extra-index-url https://download.pytorch.org/whl/cu118
    elif [[ "$CUDA_VERSION" == 11.7* ]]; then
        log_info "Installing PyTorch for CUDA 11.7..."
        python3 -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
            --extra-index-url https://download.pytorch.org/whl/cu117
    elif [[ "$CUDA_VERSION" == 11.6* ]]; then
        log_info "Installing PyTorch 1.13.1 for CUDA 11.6 (4DGaussians native)..."
        python3 -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 \
            --extra-index-url https://download.pytorch.org/whl/cu116
    else
        log_warn "Unknown CUDA version, installing latest PyTorch with CUDA 11.8..."
        python3 -m pip install torch torchvision torchaudio \
            --extra-index-url https://download.pytorch.org/whl/cu118
    fi

    # Install other Python dependencies
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

    if [ -f "$REQUIREMENTS_FILE" ]; then
        log_info "Installing Python dependencies from requirements.txt..."
        # Use --ignore-installed to avoid distutils package conflicts
        # (e.g., blinker, pyparsing that come with system Python)
        python3 -m pip install --ignore-installed -r "$REQUIREMENTS_FILE" || {
            log_warn "Some packages failed with --ignore-installed, trying without..."
            python3 -m pip install -r "$REQUIREMENTS_FILE" --no-deps || true
        }
    else
        log_warn "requirements.txt not found at $REQUIREMENTS_FILE"
        log_info "Installing core dependencies manually..."
        python3 -m pip install --ignore-installed numpy scipy opencv-python-headless pillow tqdm loguru pyyaml || true
        python3 -m pip install --ignore-installed plyfile imageio || true
    fi

    # Verify PyTorch CUDA availability
    log_info "Verifying PyTorch CUDA availability..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA is not available!')
" || log_warn "Could not verify PyTorch CUDA"

    log_info "Python environment setup complete."
}

# ============================================================================
# Git Submodules Initialization
# ============================================================================
setup_submodules() {
    log_section "Initializing Git Submodules"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"

    # Check if this is a git repository
    if [ ! -d ".git" ]; then
        log_warn "Not a git repository. Skipping submodule initialization."
        return 0
    fi

    # Check if .gitmodules exists
    if [ -f ".gitmodules" ]; then
        log_info "Initializing git submodules..."
        git submodule init
        git submodule update --init --recursive
        log_info "Git submodules initialized successfully."
    else
        log_warn ".gitmodules not found. Creating submodules directory for manual setup..."
        mkdir -p submodules

        # Clone the default 4DGS repository if not present
        if [ ! -d "submodules/4dgs" ]; then
            log_info "Cloning hustvl/4DGaussians repository..."
            git clone https://github.com/hustvl/4DGaussians.git submodules/4dgs

            # Install 4DGS dependencies
            if [ -f "submodules/4dgs/requirements.txt" ]; then
                log_info "Installing 4DGS dependencies..."
                python3 -m pip install -r submodules/4dgs/requirements.txt
            fi

            # Build custom CUDA extensions if setup.py exists
            if [ -f "submodules/4dgs/setup.py" ]; then
                log_info "Building 4DGS CUDA extensions..."
                cd submodules/4dgs
                python3 setup.py develop
                cd "$SCRIPT_DIR"
            fi
        fi
    fi

    cd "$SCRIPT_DIR"
}

# ============================================================================
# Build Custom CUDA Extensions
# ============================================================================
build_cuda_extensions() {
    log_section "Building CUDA Extensions"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Check for gaussian-splatting submodule with depth-diff-gaussian-rasterization
    # Note: hustvl/4DGaussians uses depth-diff-gaussian-rasterization (not diff-gaussian-rasterization)
    RASTERIZER_DIR="$SCRIPT_DIR/submodules/4dgs/submodules/depth-diff-gaussian-rasterization"
    if [ -d "$RASTERIZER_DIR" ]; then
        log_info "Building depth-diff-gaussian-rasterization..."
        cd "$RASTERIZER_DIR"
        # Use --no-build-isolation to prevent pip from creating isolated build environments
        # This is required for CUDA extensions that need access to system CUDA installation
        python3 -m pip install -e . --no-build-isolation
        cd "$SCRIPT_DIR"
    else
        # Fallback to diff-gaussian-rasterization (older naming)
        RASTERIZER_DIR="$SCRIPT_DIR/submodules/4dgs/submodules/diff-gaussian-rasterization"
        if [ -d "$RASTERIZER_DIR" ]; then
            log_info "Building diff-gaussian-rasterization..."
            cd "$RASTERIZER_DIR"
            python3 -m pip install -e . --no-build-isolation
            cd "$SCRIPT_DIR"
        fi
    fi

    # Check for simple-knn
    SIMPLE_KNN_DIR="$SCRIPT_DIR/submodules/4dgs/submodules/simple-knn"
    if [ -d "$SIMPLE_KNN_DIR" ]; then
        log_info "Building simple-knn..."
        cd "$SIMPLE_KNN_DIR"
        python3 -m pip install -e . --no-build-isolation
        cd "$SCRIPT_DIR"
    fi

    log_info "CUDA extensions built successfully."
}

# ============================================================================
# Patch 4DGS Code for RunPod Compatibility
# ============================================================================
patch_4dgs_code() {
    log_section "Patching 4DGS Code for RunPod/Headless Environment"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DGS_DIR="$SCRIPT_DIR/submodules/4dgs"

    if [ ! -d "$DGS_DIR" ]; then
        log_warn "4DGS directory not found, skipping patches"
        return 0
    fi

    # ========================================================================
    # Patch 1: Remove tkinter import from deformation.py (headless fix)
    # ========================================================================
    DEFORMATION_FILE="$DGS_DIR/scene/deformation.py"
    if [ -f "$DEFORMATION_FILE" ]; then
        if grep -q "from tkinter import" "$DEFORMATION_FILE"; then
            log_info "Patching deformation.py: removing tkinter import..."
            # Replace 'from tkinter import W' with 'W = "w"' (the constant is just a string)
            sed -i 's/from tkinter import W/W = "w"  # Patched: removed tkinter for headless/' "$DEFORMATION_FILE"
            log_info "  ✓ Removed tkinter import from deformation.py"
        else
            log_info "  ✓ deformation.py already patched or doesn't need patching"
        fi
    fi

    # ========================================================================
    # Patch 2: Add transforms.json support to scene/__init__.py
    # ========================================================================
    SCENE_INIT_FILE="$DGS_DIR/scene/__init__.py"
    if [ -f "$SCENE_INIT_FILE" ]; then
        if ! grep -q "transforms.json" "$SCENE_INIT_FILE"; then
            log_info "Patching scene/__init__.py: adding transforms.json support..."
            # Add transforms.json check before the scene type error
            sed -i 's/raise Exception("Could not recognize scene type!")/# Check for transforms.json (NeRF synthetic format)\n            elif os.path.exists(os.path.join(args.source_path, "transforms.json")):\n                print("Found transforms.json, using NeRF synthetic format")\n                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)\n            else:\n                raise Exception("Could not recognize scene type!")/' "$SCENE_INIT_FILE"
            log_info "  ✓ Added transforms.json support to scene/__init__.py"
        else
            log_info "  ✓ scene/__init__.py already has transforms.json support"
        fi
    fi

    # ========================================================================
    # Patch 3: Fix double extension bug in dataset_readers.py
    # ========================================================================
    DATASET_READERS_FILE="$DGS_DIR/scene/dataset_readers.py"
    if [ -f "$DATASET_READERS_FILE" ]; then
        log_info "Patching dataset_readers.py: checking for double extension bug..."
        # This is a complex fix, we'll use Python for precise patching
        export DATASET_READERS_FILE
        python3 << 'PYEOF'
import re
import os

file_path = os.environ.get('DATASET_READERS_FILE')
if not os.path.exists(file_path):
    print(f"  File not found: {file_path}")
    exit(0)

with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if "# Patched: avoid double extension" in content:
    print("  ✓ dataset_readers.py already patched for double extension")
    exit(0)

# Pattern to find image saving that might cause double extension
# Original: image_path = os.path.join(images_folder, image_name + extension)
# Should be: check if extension already in image_name

modified = False

# Fix 1: In readCamerasFromTransforms - check if file_path already has extension
old_pattern = r"file_path = os\.path\.join\(path, frame\[\"file_path\"\]\)"
new_pattern = '''file_path = os.path.join(path, frame["file_path"])
        # Patched: avoid double extension - check if extension exists
        if not os.path.exists(file_path) and not file_path.endswith(extension):
            file_path = file_path + extension'''

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_pattern, content)
    modified = True

if modified:
    with open(file_path, 'w') as f:
        f.write(content)
    print("  ✓ Patched dataset_readers.py for double extension bug")
else:
    print("  ✓ dataset_readers.py pattern not found (may be different version)")
PYEOF
    fi

    # ========================================================================
    # Patch 4: Ensure final iteration saves checkpoint in train.py
    # ========================================================================
    TRAIN_FILE="$DGS_DIR/train.py"
    if [ -f "$TRAIN_FILE" ]; then
        log_info "Patching train.py: ensuring final iteration checkpoint save..."
        # Use Python for complex patching
        export TRAIN_FILE
        python3 << 'PYEOF'
import os
import re

file_path = os.environ.get('TRAIN_FILE')
if not os.path.exists(file_path):
    print(f"  File not found: {file_path}")
    exit(0)

with open(file_path, 'r') as f:
    content = f.read()

# Check if already patched
if "# Patched: also save at final iteration" in content:
    print("  ✓ train.py already patched for final iteration save")
    exit(0)

# Pattern: "if (iteration in saving_iterations):"
# Should become: "if (iteration in saving_iterations) or (iteration == opt.iterations):"

old_pattern = r"if\s*\(\s*iteration\s+in\s+saving_iterations\s*\)\s*:"
new_pattern = "if (iteration in saving_iterations) or (iteration == opt.iterations):  # Patched: also save at final iteration"

if re.search(old_pattern, content):
    content = re.sub(old_pattern, new_pattern, content)
    with open(file_path, 'w') as f:
        f.write(content)
    print("  ✓ Patched train.py for final iteration checkpoint save")
else:
    print("  ! Could not find save iteration pattern in train.py (may be different version)")
PYEOF
    fi

    log_info "4DGS code patching completed."
}

# ============================================================================
# Create Directory Structure
# ============================================================================
create_directories() {
    log_section "Creating Directory Structure"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Create required directories
    directories=(
        "data/inputs"
        "data/processed"
        "data/outputs"
        "logs"
        "configs"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "$SCRIPT_DIR/$dir"
        log_info "Created directory: $dir"
    done
}

# ============================================================================
# Environment Variables Setup
# ============================================================================
setup_environment_vars() {
    log_section "Setting Up Environment Variables"

    # Create a shell script to source for environment variables
    ENV_FILE="/tmp/4dgs_env.sh"

    cat > "$ENV_FILE" << 'EOF'
# 4DGS Pipeline Environment Variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable GUI for headless operation
export DISPLAY=
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# COLMAP settings for headless operation
export COLMAP_DATABASE_CACHE_SIZE=256
EOF

    log_info "Environment variables written to $ENV_FILE"
    log_info "To activate, run: source $ENV_FILE"

    # Source it for current session
    source "$ENV_FILE"
}

# ============================================================================
# Verify Installation
# ============================================================================
verify_installation() {
    log_section "Verifying Installation"

    ERRORS=0

    # Check FFmpeg
    if command -v ffmpeg &> /dev/null; then
        log_info "✓ FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
    else
        log_error "✗ FFmpeg not found"
        ((ERRORS++))
    fi

    # Check FFprobe
    if command -v ffprobe &> /dev/null; then
        log_info "✓ FFprobe: available"
    else
        log_error "✗ FFprobe not found"
        ((ERRORS++))
    fi

    # Check COLMAP
    if command -v colmap &> /dev/null; then
        COLMAP_VERSION=$(colmap -h 2>&1 | head -1)
        log_info "✓ COLMAP: $COLMAP_VERSION"
    else
        log_error "✗ COLMAP not found"
        ((ERRORS++))
    fi

    # Check Python packages
    log_info "Checking Python packages..."
    python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || { log_error "  ✗ PyTorch"; ((ERRORS++)); }
    python3 -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')" 2>/dev/null || { log_error "  ✗ NumPy"; ((ERRORS++)); }
    python3 -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" 2>/dev/null || { log_error "  ✗ OpenCV"; ((ERRORS++)); }
    python3 -c "import yaml; print('  ✓ PyYAML')" 2>/dev/null || { log_error "  ✗ PyYAML"; ((ERRORS++)); }
    python3 -c "import loguru; print('  ✓ Loguru')" 2>/dev/null || { log_error "  ✗ Loguru"; ((ERRORS++)); }
    python3 -c "import tqdm; print('  ✓ tqdm')" 2>/dev/null || { log_error "  ✗ tqdm"; ((ERRORS++)); }

    # Check CUDA availability
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        log_info "✓ CUDA available: $GPU_NAME"
    else
        log_warn "⚠ CUDA not available - training will be slow!"
    fi

    echo ""
    if [ $ERRORS -eq 0 ]; then
        log_info "All checks passed! Environment is ready."
        return 0
    else
        log_error "$ERRORS check(s) failed. Please review the errors above."
        return 1
    fi
}

# ============================================================================
# Print Usage Instructions
# ============================================================================
print_usage() {
    log_section "Setup Complete - Usage Instructions"

    echo ""
    echo "The 4DGS pipeline environment is now configured."
    echo ""
    echo "Quick Start:"
    echo "  1. Download sample data:"
    echo "     ./download_sample_data.sh"
    echo ""
    echo "  2. Run the full pipeline:"
    echo "     ./run_pipeline.sh data/inputs/sample my_experiment"
    echo ""
    echo "  3. Or run individual steps:"
    echo "     python main.py --input data/inputs --experiment test --step preprocess"
    echo "     python main.py --experiment test --step train"
    echo ""
    echo "  4. For public datasets with existing poses (skip COLMAP):"
    echo "     python main.py --input data/inputs/n3dv_sample --experiment n3dv --skip_colmap"
    echo ""
    echo "Environment Variables (already set for this session):"
    echo "  source /tmp/4dgs_env.sh"
    echo ""
}

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log_section "4DGS Pipeline - RunPod Environment Setup"

    # Parse arguments
    SKIP_PACKAGES=false
    SKIP_SUBMODULES=false
    SKIP_VERIFY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-packages)
                SKIP_PACKAGES=true
                shift
                ;;
            --skip-submodules)
                SKIP_SUBMODULES=true
                shift
                ;;
            --skip-verify)
                SKIP_VERIFY=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-packages    Skip system package installation"
                echo "  --skip-submodules  Skip git submodule initialization"
                echo "  --skip-verify      Skip installation verification"
                echo "  -h, --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Run setup steps
    check_permissions

    if [ "$SKIP_PACKAGES" = false ]; then
        install_system_packages
    fi

    check_cuda || log_warn "CUDA check failed, continuing anyway..."

    setup_python_env

    if [ "$SKIP_SUBMODULES" = false ]; then
        setup_submodules
        build_cuda_extensions
        patch_4dgs_code
    fi

    create_directories
    setup_environment_vars

    if [ "$SKIP_VERIFY" = false ]; then
        verify_installation
    fi

    print_usage

    log_info "Setup completed successfully!"
}

# Run main function
main "$@"
