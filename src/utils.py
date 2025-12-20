"""
Utility functions for the 4DGS Pipeline.

Common helper functions used across the pipeline modules.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import subprocess

from loguru import logger


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root
    """
    # Try to find project root by looking for main.py or pyproject.toml
    current = Path(__file__).resolve().parent.parent

    markers = ["main.py", "pyproject.toml", "requirements.txt", ".git"]
    for _ in range(5):  # Max 5 levels up
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent

    # Fallback to current working directory
    return Path.cwd()


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_symlink(src: Path, dst: Path, force: bool = False) -> Path:
    """
    Create a symbolic link safely.

    Args:
        src: Source path (target of the link)
        dst: Destination path (the link itself)
        force: If True, remove existing file/link

    Returns:
        Path to the created symlink
    """
    if dst.exists() or dst.is_symlink():
        if force:
            dst.unlink()
        else:
            return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())
    return dst


def file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    Compute hash of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)

    Returns:
        Hex digest of the hash
    """
    h = hashlib.new(algorithm)

    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)

    return h.hexdigest()


def count_files(directory: Path, pattern: str = "*") -> int:
    """
    Count files matching a pattern in a directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern

    Returns:
        Number of matching files
    """
    return len(list(directory.glob(pattern)))


def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """
    Find the latest training checkpoint.

    Args:
        model_dir: Model directory

    Returns:
        Path to latest checkpoint, or None if not found
    """
    checkpoint_dir = model_dir / "point_cloud"
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("iteration_*"))
    if not checkpoints:
        return None

    # Sort by iteration number
    def get_iteration(p: Path) -> int:
        try:
            return int(p.name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return max(checkpoints, key=get_iteration)


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    capture: bool = True,
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a command with logging.

    Args:
        cmd: Command and arguments
        cwd: Working directory
        env: Environment variables
        capture: Capture stdout/stderr
        check: Raise exception on non-zero return code

    Returns:
        CompletedProcess object
    """
    logger.debug(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=capture,
        text=True if capture else None
    )

    if check and result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        if result.stderr:
            logger.error(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)

    return result


def save_metadata(
    path: Path,
    metadata: Dict[str, Any],
    append: bool = False
):
    """
    Save metadata to a JSON file.

    Args:
        path: Output path
        metadata: Metadata dictionary
        append: If True, merge with existing metadata
    """
    if append and path.exists():
        with open(path, 'r') as f:
            existing = json.load(f)
        existing.update(metadata)
        metadata = existing

    # Add timestamp
    metadata["_updated_at"] = datetime.now().isoformat()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_metadata(path: Path) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.

    Args:
        path: Path to metadata file

    Returns:
        Metadata dictionary
    """
    if not path.exists():
        return {}

    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get GPU information using nvidia-smi.

    Returns:
        List of GPU info dictionaries
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3])
                })

        return gpus

    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("nvidia-smi not available")
        return []


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required external dependencies are available.

    Returns:
        Dictionary mapping dependency name to availability
    """
    import shutil

    dependencies = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "colmap": shutil.which("colmap") is not None,
        "nvidia-smi": shutil.which("nvidia-smi") is not None,
    }

    return dependencies


def print_system_info():
    """Print system information for debugging."""
    import platform

    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working Directory: {Path.cwd()}")

    # Check dependencies
    deps = check_dependencies()
    print("\nExternal Dependencies:")
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {name}")

    # GPU info
    gpus = get_gpu_info()
    if gpus:
        print(f"\nGPUs ({len(gpus)} detected):")
        for gpu in gpus:
            print(f"  [{gpu['index']}] {gpu['name']} "
                  f"({gpu['memory_free_mb']}/{gpu['memory_total_mb']} MB free)")
    else:
        print("\nNo NVIDIA GPUs detected")

    print("=" * 60)


class ProgressTracker:
    """Simple progress tracker for multi-step operations."""

    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.step_times: List[float] = []
        self._start_time: Optional[float] = None

    def start(self):
        """Start tracking."""
        import time
        self._start_time = time.time()
        logger.info(f"{self.description}: Starting ({self.total_steps} steps)")

    def step(self, message: str = ""):
        """Record a step completion."""
        import time

        self.current_step += 1
        if self._start_time:
            elapsed = time.time() - self._start_time
            self.step_times.append(elapsed)

        percent = (self.current_step / self.total_steps) * 100
        logger.info(
            f"{self.description}: Step {self.current_step}/{self.total_steps} "
            f"({percent:.0f}%) {message}"
        )

    def finish(self):
        """Finish tracking and report."""
        import time

        if self._start_time:
            total_time = time.time() - self._start_time
            logger.info(
                f"{self.description}: Completed in {format_time(total_time)}"
            )


if __name__ == "__main__":
    # Print system info when run directly
    print_system_info()
