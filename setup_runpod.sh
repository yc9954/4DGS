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

# Check if running as root or with sudo
check_permissions() {
    if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
        log_warn "This script may require sudo for some operations."
        log_info "You may be prompted for your password."
    fi
}

# ============================================================================
# System Package Installation
# ============================================================================
install_system_packages() {
    log_section "Installing System Packages"

    # Update package lists
    log_info "Updating apt package lists..."
    sudo apt-get update -qq

    # Install core dependencies
    log_info "Installing core dependencies..."
    sudo apt-get install -y --no-install-recommends \
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
    sudo apt-get install -y ffmpeg

    # Install OpenCV dependencies (headless)
    log_info "Installing OpenCV/OpenGL dependencies..."
    sudo apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1

    # Install COLMAP and its dependencies
    log_info "Installing COLMAP..."
    sudo apt-get install -y colmap

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
        python3 -m pip install -r "$REQUIREMENTS_FILE"
    else
        log_warn "requirements.txt not found at $REQUIREMENTS_FILE"
        log_info "Installing core dependencies manually..."
        python3 -m pip install numpy scipy opencv-python-headless pillow tqdm loguru pyyaml
        python3 -m pip install mmcv==1.6.0 lpips plyfile pytorch_msssim open3d imageio[ffmpeg]
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

    # Check for gaussian-splatting submodule with diff-gaussian-rasterization
    RASTERIZER_DIR="$SCRIPT_DIR/submodules/4dgs/submodules/diff-gaussian-rasterization"
    if [ -d "$RASTERIZER_DIR" ]; then
        log_info "Building diff-gaussian-rasterization..."
        cd "$RASTERIZER_DIR"
        python3 -m pip install -e .
        cd "$SCRIPT_DIR"
    fi

    # Check for simple-knn
    SIMPLE_KNN_DIR="$SCRIPT_DIR/submodules/4dgs/submodules/simple-knn"
    if [ -d "$SIMPLE_KNN_DIR" ]; then
        log_info "Building simple-knn..."
        cd "$SIMPLE_KNN_DIR"
        python3 -m pip install -e .
        cd "$SCRIPT_DIR"
    fi

    log_info "CUDA extensions built successfully."
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
