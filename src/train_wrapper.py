"""
Training Wrapper for 4DGS Pipeline.

Provides a wrapper class to orchestrate training of 4D Gaussian Splatting models.
Handles subprocess execution, logging, checkpointing, and progress monitoring.

This wrapper is designed to work with various 4DGS implementations:
- fudan-zvg/4d-gaussian-splatting
- hustvl/4DGaussians
- Custom implementations

The actual training is delegated to the external training script.
"""

import subprocess
import sys
import os
import time
import threading
import json
from pathlib import Path
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import shutil

from loguru import logger

from .config import PipelineConfig, TrainingConfig


@dataclass
class TrainingProgress:
    """Container for training progress information."""

    current_iteration: int = 0
    total_iterations: int = 0
    elapsed_time: float = 0.0
    loss: Optional[float] = None
    psnr: Optional[float] = None
    num_gaussians: Optional[int] = None
    last_save_iteration: Optional[int] = None

    @property
    def progress_percent(self) -> float:
        """Get training progress as percentage."""
        if self.total_iterations == 0:
            return 0.0
        return (self.current_iteration / self.total_iterations) * 100

    @property
    def eta(self) -> Optional[timedelta]:
        """Estimate time remaining."""
        if self.current_iteration == 0 or self.elapsed_time == 0:
            return None

        rate = self.current_iteration / self.elapsed_time
        remaining = self.total_iterations - self.current_iteration
        return timedelta(seconds=remaining / rate)


class TrainWrapper:
    """
    Wrapper for 4D Gaussian Splatting training.

    This class handles:
    - Preparing training data and configuration
    - Executing the training script via subprocess
    - Monitoring training progress
    - Handling checkpoints and outputs
    - Graceful error handling and recovery

    Default engine: hustvl/4DGaussians (installed in submodules/4dgs/)
    """

    # Supported 4DGS implementations
    SUPPORTED_METHODS = {
        "4dgs": {
            "repo": "hustvl/4DGaussians",
            "train_script": "train.py",
            "required_args": ["source_path", "model_path"],
            "arg_mapping": {
                "source_path": "-s",
                "model_path": "-m",
            },
            "extra_args": [],
        },
        "4dgaussians": {
            "repo": "hustvl/4DGaussians",
            "train_script": "train.py",
            "required_args": ["source_path", "model_path"],
            "arg_mapping": {
                "source_path": "-s",
                "model_path": "-m",
            },
            "extra_args": [],
        },
        "fudan": {
            "repo": "fudan-zvg/4d-gaussian-splatting",
            "train_script": "train.py",
            "required_args": ["source_path", "model_path"],
            "arg_mapping": {
                "source_path": "--source_path",
                "model_path": "--model_path",
            },
            "extra_args": [],
        },
        "custom": {
            "repo": None,
            "train_script": "train.py",
            "required_args": ["source_path", "model_path"],
            "arg_mapping": {
                "source_path": "--source_path",
                "model_path": "--model_path",
            },
            "extra_args": [],
        }
    }

    def __init__(
        self,
        config: PipelineConfig,
        method: str = "4dgs",
        method_path: Optional[Path] = None
    ):
        """
        Initialize the training wrapper.

        Args:
            config: Pipeline configuration object
            method: Training method to use ('4dgs', '4dgaussians', 'fudan', 'custom')
            method_path: Path to the training method code (if custom/local)
        """
        self.config = config
        self.train_config = config.training
        self.method = method
        self.method_path = method_path

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported: {list(self.SUPPORTED_METHODS.keys())}"
            )

        self.method_info = self.SUPPORTED_METHODS[method]
        self.progress = TrainingProgress()
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[Path] = None
        self._method_dir: Optional[Path] = None

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        # Start from this file and go up to find project root
        current = Path(__file__).resolve().parent.parent
        markers = ["main.py", "requirements.txt", ".git"]
        for _ in range(5):
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent
        return Path.cwd()

    def _find_train_script(self) -> Path:
        """
        Find the training script for the selected method.

        For hustvl/4DGaussians (default), looks in:
        - submodules/4dgs/train.py
        - submodules/4DGaussians/train.py
        - external/4dgs/train.py

        Returns:
            Path to the training script
        """
        project_root = self._get_project_root()

        if self.method_path:
            script = self.method_path / self.method_info["train_script"]
            if script.exists():
                self._method_dir = self.method_path
                return script
            raise FileNotFoundError(f"Training script not found: {script}")

        # Check common locations for the 4DGS submodule
        possible_paths = [
            # Primary location for hustvl/4DGaussians
            project_root / "submodules" / "4dgs" / self.method_info["train_script"],
            project_root / "submodules" / "4DGaussians" / self.method_info["train_script"],
            project_root / "submodules" / "4d-gaussian-splatting" / self.method_info["train_script"],
            # Alternative locations
            project_root / "external" / "4dgs" / self.method_info["train_script"],
            project_root / "external" / "4DGaussians" / self.method_info["train_script"],
            project_root / "third_party" / "4dgs" / self.method_info["train_script"],
            project_root / "third_party" / "4DGaussians" / self.method_info["train_script"],
            # Method-specific subdirectory
            project_root / "submodules" / self.method / self.method_info["train_script"],
            # Current directory fallback
            project_root / self.method_info["train_script"],
        ]

        for path in possible_paths:
            if path.exists():
                self._method_dir = path.parent
                logger.info(f"Found training script at: {path}")
                return path

        # Provide helpful error message
        error_msg = (
            f"Could not find training script for {self.method}.\n"
            f"Searched locations:\n"
        )
        for p in possible_paths[:6]:  # Show first 6 paths
            error_msg += f"  - {p}\n"

        error_msg += (
            f"\nTo fix this, either:\n"
            f"1. Clone the 4DGS repository:\n"
            f"   git clone https://github.com/hustvl/4DGaussians.git submodules/4dgs\n"
            f"2. Or specify a custom path:\n"
            f"   --method_path /path/to/4dgs"
        )

        raise FileNotFoundError(error_msg)

    def _build_train_command(
        self,
        source_path: Path,
        model_path: Path,
        **kwargs
    ) -> List[str]:
        """
        Build the training command with all arguments.

        Uses the arg_mapping from method_info to generate the correct
        command-line arguments for the specific 4DGS implementation.

        For hustvl/4DGaussians:
            python train.py -s <source_path> -m <model_path> --iterations <n>

        For fudan/4d-gaussian-splatting:
            python train.py --source_path <path> --model_path <path>

        Args:
            source_path: Path to the training data
            model_path: Path to save the trained model
            **kwargs: Additional training arguments

        Returns:
            List of command arguments
        """
        train_script = self._find_train_script()

        # Get argument mapping for this method
        arg_mapping = self.method_info.get("arg_mapping", {})
        source_arg = arg_mapping.get("source_path", "--source_path")
        model_arg = arg_mapping.get("model_path", "--model_path")

        cmd = [
            sys.executable,  # Use current Python interpreter
            str(train_script),
            source_arg, str(source_path),
            model_arg, str(model_path),
        ]

        # Add training configuration
        cmd.extend([
            "--iterations", str(self.train_config.iterations),
        ])

        # Save iterations - hustvl uses space-separated list
        if self.train_config.save_iterations:
            if self.method in ["4dgs", "4dgaussians"]:
                # hustvl/4DGaussians format
                for iteration in self.train_config.save_iterations:
                    cmd.extend(["--save_iterations", str(iteration)])
            else:
                save_iters = " ".join(str(i) for i in self.train_config.save_iterations)
                cmd.extend(["--save_iterations", save_iters])

        # Test iterations
        if self.train_config.test_iterations:
            if self.method in ["4dgs", "4dgaussians"]:
                for iteration in self.train_config.test_iterations:
                    cmd.extend(["--test_iterations", str(iteration)])
            else:
                test_iters = " ".join(str(i) for i in self.train_config.test_iterations)
                cmd.extend(["--test_iterations", test_iters])

        # Resolution
        if self.train_config.resolution != -1:
            cmd.extend(["--resolution", str(self.train_config.resolution)])

        # Background color
        if self.train_config.white_background:
            cmd.append("--white_background")

        # Add method-specific extra arguments
        for extra_arg in self.method_info.get("extra_args", []):
            cmd.append(extra_arg)

        # Add any additional keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

        return cmd

    def _get_method_env(self) -> Dict[str, str]:
        """
        Get environment variables for the training subprocess.

        Adds the method directory to PYTHONPATH for proper imports.

        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()

        # Add the method directory to PYTHONPATH
        if self._method_dir:
            existing_path = env.get("PYTHONPATH", "")
            paths_to_add = [str(self._method_dir)]
            
            # Add CUDA extension paths for 4DGS
            if "4dgs" in str(self._method_dir).lower():
                method_path = Path(self._method_dir)
                # Add depth-diff-gaussian-rasterization
                rasterizer_path = method_path / "submodules" / "depth-diff-gaussian-rasterization"
                if rasterizer_path.exists():
                    paths_to_add.append(str(rasterizer_path))
                # Add simple-knn
                knn_path = method_path / "submodules" / "simple-knn"
                if knn_path.exists():
                    paths_to_add.append(str(knn_path))
            
            if existing_path:
                env["PYTHONPATH"] = ":".join(paths_to_add + [existing_path])
            else:
                env["PYTHONPATH"] = ":".join(paths_to_add)

        return env

    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a log line from the training output.

        Different training implementations have different log formats.
        This method tries to extract common metrics.

        Args:
            line: Log line to parse

        Returns:
            Dictionary of parsed metrics, or None if not a metrics line
        """
        metrics = {}

        # Common patterns in 4DGS implementations
        # Pattern: "Iteration: 1000, Loss: 0.0123, PSNR: 25.5"
        import re

        # Try to extract iteration
        iter_match = re.search(r'[Ii]teration[:\s]+(\d+)', line)
        if iter_match:
            metrics['iteration'] = int(iter_match.group(1))

        # Try to extract loss
        loss_match = re.search(r'[Ll]oss[:\s]+([0-9.e+-]+)', line)
        if loss_match:
            try:
                metrics['loss'] = float(loss_match.group(1))
            except ValueError:
                pass

        # Try to extract PSNR
        psnr_match = re.search(r'PSNR[:\s]+([0-9.]+)', line)
        if psnr_match:
            try:
                metrics['psnr'] = float(psnr_match.group(1))
            except ValueError:
                pass

        # Try to extract number of Gaussians
        gaussian_match = re.search(r'[Nn]um[_\s]*[Gg]aussians?[:\s]+(\d+)', line)
        if gaussian_match:
            metrics['num_gaussians'] = int(gaussian_match.group(1))

        return metrics if metrics else None

    def _monitor_output(
        self,
        process: subprocess.Popen,
        log_file: Optional[Path] = None,
        callback: Optional[Callable[[TrainingProgress], None]] = None
    ):
        """
        Monitor training process output in a separate thread.

        Args:
            process: The subprocess to monitor
            log_file: Optional file to write logs to
            callback: Optional callback function for progress updates
        """
        start_time = time.time()
        log_handle = None

        if log_file:
            log_handle = open(log_file, 'w')

        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                line = line.strip()

                # Write to log file
                if log_handle:
                    log_handle.write(line + '\n')
                    log_handle.flush()

                # Parse metrics
                metrics = self._parse_log_line(line)
                if metrics:
                    if 'iteration' in metrics:
                        self.progress.current_iteration = metrics['iteration']
                    if 'loss' in metrics:
                        self.progress.loss = metrics['loss']
                    if 'psnr' in metrics:
                        self.progress.psnr = metrics['psnr']
                    if 'num_gaussians' in metrics:
                        self.progress.num_gaussians = metrics['num_gaussians']

                    self.progress.elapsed_time = time.time() - start_time

                    # Call progress callback
                    if callback:
                        callback(self.progress)

                # Log important lines
                if any(kw in line.lower() for kw in ['error', 'warning', 'saving', 'checkpoint']):
                    logger.info(f"[Training] {line}")

        finally:
            if log_handle:
                log_handle.close()

    def train(
        self,
        source_path: Path,
        model_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        **kwargs
    ) -> Path:
        """
        Start training and wait for completion.

        Args:
            source_path: Path to the training data directory
            model_path: Path to save the trained model (uses config if None)
            progress_callback: Optional callback for progress updates
            **kwargs: Additional training arguments

        Returns:
            Path to the trained model
        """
        if model_path is None:
            model_path = self.config.get_model_dir()

        model_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._log_file = model_path / "training.log"

        # Initialize progress
        self.progress = TrainingProgress(
            total_iterations=self.train_config.iterations
        )

        # Build command
        cmd = self._build_train_command(source_path, model_path, **kwargs)

        logger.info(f"Starting training: {self.method}")
        logger.info(f"Source: {source_path}")
        logger.info(f"Output: {model_path}")
        logger.info(f"Iterations: {self.train_config.iterations}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Start training process
        start_time = time.time()

        # Get environment with PYTHONPATH set for the method
        env = self._get_method_env()

        # Set working directory to method directory for relative imports
        cwd = self._method_dir if self._method_dir else None

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                cwd=cwd
            )

            # Start output monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_output,
                args=(self._process, self._log_file, progress_callback)
            )
            monitor_thread.start()

            # Wait for completion
            return_code = self._process.wait()
            monitor_thread.join()

            elapsed = time.time() - start_time

            if return_code != 0:
                raise RuntimeError(
                    f"Training failed with return code {return_code}. "
                    f"Check log file: {self._log_file}"
                )

            logger.info(f"Training completed in {elapsed:.1f}s")
            logger.info(f"Final iteration: {self.progress.current_iteration}")
            if self.progress.psnr:
                logger.info(f"Final PSNR: {self.progress.psnr:.2f}")

            # Save training metadata
            self._save_training_metadata(model_path, elapsed)

            return model_path

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            if self._process:
                self._process.terminate()
            raise

        except Exception as e:
            logger.error(f"Training error: {e}")
            if self._process:
                self._process.terminate()
            raise

    def _save_training_metadata(self, model_path: Path, elapsed_time: float):
        """Save training metadata to JSON file."""
        metadata = {
            "method": self.method,
            "iterations": self.train_config.iterations,
            "elapsed_time_seconds": elapsed_time,
            "completed_at": datetime.now().isoformat(),
            "config": {
                "resolution": self.train_config.resolution,
                "white_background": self.train_config.white_background,
                "save_iterations": self.train_config.save_iterations,
                "test_iterations": self.train_config.test_iterations,
            },
            "final_metrics": {
                "iteration": self.progress.current_iteration,
                "loss": self.progress.loss,
                "psnr": self.progress.psnr,
                "num_gaussians": self.progress.num_gaussians,
            }
        }

        metadata_path = model_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved training metadata to {metadata_path}")

    def resume_training(
        self,
        model_path: Path,
        additional_iterations: int = 10000,
        **kwargs
    ) -> Path:
        """
        Resume training from a checkpoint.

        Args:
            model_path: Path to the model checkpoint
            additional_iterations: Number of additional iterations
            **kwargs: Additional training arguments

        Returns:
            Path to the trained model
        """
        # Find latest checkpoint
        checkpoints = list(model_path.glob("point_cloud/iteration_*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {model_path}")

        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('_')[1]))
        start_iteration = int(latest_checkpoint.name.split('_')[1])

        logger.info(f"Resuming from iteration {start_iteration}")

        # Update iterations
        total_iterations = start_iteration + additional_iterations
        self.train_config.iterations = total_iterations

        # Call train with start_checkpoint argument
        return self.train(
            source_path=model_path.parent / "source",  # Assumes standard structure
            model_path=model_path,
            start_checkpoint=str(latest_checkpoint),
            **kwargs
        )

    def stop(self):
        """Stop the training process gracefully."""
        if self._process and self._process.poll() is None:
            logger.warning("Stopping training process...")
            self._process.terminate()

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing training process...")
                self._process.kill()


class MockTrainWrapper(TrainWrapper):
    """
    Mock training wrapper for testing and development.

    Simulates training without actually running the training script.
    Useful for testing the pipeline without GPU or training code.
    """

    def train(
        self,
        source_path: Path,
        model_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        **kwargs
    ) -> Path:
        """Simulate training."""
        if model_path is None:
            model_path = self.config.get_model_dir()

        model_path.mkdir(parents=True, exist_ok=True)

        logger.info("=== MOCK TRAINING MODE ===")
        logger.info(f"Would train on: {source_path}")
        logger.info(f"Would save to: {model_path}")
        logger.info(f"Iterations: {self.train_config.iterations}")

        # Simulate progress
        self.progress = TrainingProgress(
            total_iterations=self.train_config.iterations
        )

        # Simulate a few iterations
        for i in range(0, self.train_config.iterations + 1, 1000):
            self.progress.current_iteration = i
            self.progress.loss = 0.1 * (1 - i / self.train_config.iterations)
            self.progress.psnr = 20 + 10 * (i / self.train_config.iterations)
            self.progress.elapsed_time = i * 0.01

            if progress_callback:
                progress_callback(self.progress)

            time.sleep(0.1)  # Small delay to simulate

        # Create mock output
        (model_path / "point_cloud").mkdir(exist_ok=True)
        (model_path / "point_cloud" / "iteration_30000").mkdir(exist_ok=True)

        # Save mock checkpoint info
        with open(model_path / "cfg_args", 'w') as f:
            f.write("Mock training completed\n")

        self._save_training_metadata(model_path, 1.0)

        logger.info("=== MOCK TRAINING COMPLETE ===")

        return model_path


def main():
    """CLI entry point for training wrapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Train 4DGS model")
    parser.add_argument("source_path", type=Path, help="Path to training data")
    parser.add_argument("model_path", type=Path, help="Path to save model")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations")
    parser.add_argument("--method", choices=["4dgs", "4dgaussians", "custom"],
                        default="4dgs", help="Training method")
    parser.add_argument("--mock", action="store_true", help="Use mock training (testing)")

    args = parser.parse_args()

    config = PipelineConfig()
    config.training.iterations = args.iterations

    if args.mock:
        wrapper = MockTrainWrapper(config, args.method)
    else:
        wrapper = TrainWrapper(config, args.method)

    def progress_callback(progress: TrainingProgress):
        print(f"\rIteration: {progress.current_iteration}/{progress.total_iterations} "
              f"({progress.progress_percent:.1f}%) "
              f"Loss: {progress.loss:.4f if progress.loss else 'N/A'} "
              f"ETA: {progress.eta}", end="")

    try:
        model_path = wrapper.train(
            args.source_path,
            args.model_path,
            progress_callback=progress_callback
        )
        print(f"\n\nTraining complete. Model saved to: {model_path}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")


if __name__ == "__main__":
    main()
