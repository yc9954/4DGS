"""
COLMAP Wrapper for 4DGS Pipeline.

Automates COLMAP Structure-from-Motion (SfM) pipeline for camera pose estimation.
Handles feature extraction, matching, and sparse reconstruction without GUI.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from enum import Enum
import time

from loguru import logger

from .config import PipelineConfig, ColmapConfig


class ColmapStep(Enum):
    """Enumeration of COLMAP pipeline steps."""

    FEATURE_EXTRACTOR = "feature_extractor"
    EXHAUSTIVE_MATCHER = "exhaustive_matcher"
    SEQUENTIAL_MATCHER = "sequential_matcher"
    SPATIAL_MATCHER = "spatial_matcher"
    MAPPER = "mapper"
    IMAGE_UNDISTORTER = "image_undistorter"
    MODEL_CONVERTER = "model_converter"


class ColmapWrapper:
    """
    Wrapper for automating COLMAP command-line operations.

    This class handles:
    - Feature extraction from images
    - Feature matching (exhaustive, sequential, or spatial)
    - Sparse 3D reconstruction (mapping)
    - Model format conversion
    - Camera pose extraction

    For static multi-view setup, we run COLMAP once on all cameras
    to get consistent camera poses across views.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the COLMAP wrapper.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.colmap_config = config.colmap
        self._check_colmap()

    def _check_colmap(self):
        """Verify COLMAP is installed and accessible."""
        if not shutil.which("colmap"):
            raise RuntimeError(
                "COLMAP not found. Please install COLMAP:\n"
                "  Ubuntu: sudo apt-get install colmap\n"
                "  Or build from source: https://colmap.github.io/install.html"
            )
        logger.debug("COLMAP found")

    def _run_colmap(
        self,
        command: str,
        args: Dict[str, str],
        step_name: str = "COLMAP"
    ) -> subprocess.CompletedProcess:
        """
        Execute a COLMAP command with the given arguments.

        Args:
            command: COLMAP command to run
            args: Dictionary of argument name -> value
            step_name: Name for logging purposes

        Returns:
            CompletedProcess object
        """
        cmd = ["colmap", command]

        for key, value in args.items():
            cmd.extend([f"--{key}", str(value)])

        logger.info(f"Running {step_name}...")
        logger.debug(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"{step_name} failed after {elapsed:.1f}s")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"COLMAP {command} failed:\n{result.stderr}")

        logger.info(f"{step_name} completed in {elapsed:.1f}s")

        return result

    def feature_extractor(
        self,
        image_path: Path,
        database_path: Path,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run COLMAP feature extraction.

        Extracts SIFT features from all images in the specified directory.

        Args:
            image_path: Directory containing input images
            database_path: Path to the COLMAP database file
            **kwargs: Additional arguments to pass to COLMAP

        Returns:
            CompletedProcess object
        """
        args = {
            "image_path": str(image_path),
            "database_path": str(database_path),
            "ImageReader.camera_model": self.colmap_config.camera_model,
            "ImageReader.single_camera": str(int(self.colmap_config.single_camera)),
            "ImageReader.single_camera_per_folder": str(int(self.colmap_config.single_camera_per_folder)),
            "SiftExtraction.max_image_size": str(self.colmap_config.max_image_size),
        }

        # GPU settings
        if self.colmap_config.use_gpu:
            args["SiftExtraction.use_gpu"] = "1"
            args["SiftExtraction.gpu_index"] = str(self.colmap_config.gpu_index)
        else:
            args["SiftExtraction.use_gpu"] = "0"

        # Thread settings
        if self.colmap_config.num_threads > 0:
            args["SiftExtraction.num_threads"] = str(self.colmap_config.num_threads)

        # Override with any additional arguments
        args.update(kwargs)

        return self._run_colmap(
            ColmapStep.FEATURE_EXTRACTOR.value,
            args,
            "Feature Extraction"
        )

    def exhaustive_matcher(
        self,
        database_path: Path,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run COLMAP exhaustive matching.

        Matches features between all pairs of images.
        Best for small datasets with few images (< 100).

        Args:
            database_path: Path to the COLMAP database file
            **kwargs: Additional arguments

        Returns:
            CompletedProcess object
        """
        args = {
            "database_path": str(database_path),
        }

        if self.colmap_config.use_gpu:
            args["SiftMatching.use_gpu"] = "1"
            args["SiftMatching.gpu_index"] = str(self.colmap_config.gpu_index)
        else:
            args["SiftMatching.use_gpu"] = "0"

        args.update(kwargs)

        return self._run_colmap(
            ColmapStep.EXHAUSTIVE_MATCHER.value,
            args,
            "Exhaustive Matching"
        )

    def sequential_matcher(
        self,
        database_path: Path,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run COLMAP sequential matching.

        Matches features between consecutive images.
        Best for video sequences.

        Args:
            database_path: Path to the COLMAP database file
            **kwargs: Additional arguments

        Returns:
            CompletedProcess object
        """
        args = {
            "database_path": str(database_path),
        }

        if self.colmap_config.use_gpu:
            args["SiftMatching.use_gpu"] = "1"
            args["SiftMatching.gpu_index"] = str(self.colmap_config.gpu_index)
        else:
            args["SiftMatching.use_gpu"] = "0"

        args.update(kwargs)

        return self._run_colmap(
            ColmapStep.SEQUENTIAL_MATCHER.value,
            args,
            "Sequential Matching"
        )

    def mapper(
        self,
        image_path: Path,
        database_path: Path,
        output_path: Path,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run COLMAP sparse mapper.

        Performs incremental Structure-from-Motion reconstruction
        to estimate camera poses and sparse 3D points.

        Args:
            image_path: Directory containing input images
            database_path: Path to the COLMAP database file
            output_path: Directory to save the sparse reconstruction
            **kwargs: Additional arguments

        Returns:
            CompletedProcess object
        """
        output_path.mkdir(parents=True, exist_ok=True)

        args = {
            "image_path": str(image_path),
            "database_path": str(database_path),
            "output_path": str(output_path),
        }

        args.update(kwargs)

        return self._run_colmap(
            ColmapStep.MAPPER.value,
            args,
            "Sparse Reconstruction"
        )

    def image_undistorter(
        self,
        image_path: Path,
        input_path: Path,
        output_path: Path,
        output_type: str = "COLMAP",
        **kwargs
    ) -> subprocess.CompletedProcess:
        """
        Run COLMAP image undistortion.

        Undistorts images based on estimated camera intrinsics.

        Args:
            image_path: Directory containing input images
            input_path: Path to the sparse reconstruction
            output_path: Directory to save undistorted images
            output_type: Output format (COLMAP, CMP-MVS, etc.)
            **kwargs: Additional arguments

        Returns:
            CompletedProcess object
        """
        output_path.mkdir(parents=True, exist_ok=True)

        args = {
            "image_path": str(image_path),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "output_type": output_type,
        }

        args.update(kwargs)

        return self._run_colmap(
            ColmapStep.IMAGE_UNDISTORTER.value,
            args,
            "Image Undistortion"
        )

    def model_converter(
        self,
        input_path: Path,
        output_path: Path,
        output_type: str = "TXT"
    ) -> subprocess.CompletedProcess:
        """
        Convert COLMAP model to different format.

        Args:
            input_path: Path to input model
            output_path: Path to output model
            output_type: Output format (TXT, BIN, etc.)

        Returns:
            CompletedProcess object
        """
        output_path.mkdir(parents=True, exist_ok=True)

        args = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "output_type": output_type,
        }

        return self._run_colmap(
            ColmapStep.MODEL_CONVERTER.value,
            args,
            "Model Conversion"
        )

    def run_pipeline(
        self,
        images_dir: Path,
        output_dir: Path,
        matcher_type: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run the complete COLMAP SfM pipeline.

        This executes the full pipeline:
        1. Feature extraction
        2. Feature matching
        3. Sparse reconstruction (mapping)

        Args:
            images_dir: Directory containing input images
            output_dir: Base output directory for COLMAP data
            matcher_type: Type of matcher to use (uses config if None)

        Returns:
            Dictionary with paths to output files/directories
        """
        matcher_type = matcher_type or self.colmap_config.matcher_type

        # Setup paths
        database_path = output_dir / "database.db"
        sparse_path = output_dir / "sparse"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        if self.config.skip_existing and (sparse_path / "0").exists():
            model_files = list((sparse_path / "0").glob("*.bin")) + \
                          list((sparse_path / "0").glob("*.txt"))
            if len(model_files) >= 3:  # cameras, images, points3D
                logger.info("COLMAP output already exists, skipping...")
                return {
                    "database": database_path,
                    "sparse": sparse_path / "0",
                    "images": images_dir,
                }

        # Count input images
        image_count = len(list(images_dir.glob("*.png"))) + \
                      len(list(images_dir.glob("*.jpg"))) + \
                      len(list(images_dir.glob("*.jpeg")))
        logger.info(f"Processing {image_count} images from {images_dir}")

        if image_count == 0:
            raise ValueError(f"No images found in {images_dir}")

        # Step 1: Feature extraction
        self.feature_extractor(images_dir, database_path)

        # Step 2: Feature matching
        if matcher_type == "exhaustive":
            self.exhaustive_matcher(database_path)
        elif matcher_type == "sequential":
            self.sequential_matcher(database_path)
        else:
            raise ValueError(f"Unknown matcher type: {matcher_type}")

        # Step 3: Sparse reconstruction
        self.mapper(images_dir, database_path, sparse_path)

        # Verify output
        reconstruction_path = sparse_path / "0"
        if not reconstruction_path.exists():
            # Check for alternative reconstruction directories
            sparse_dirs = list(sparse_path.glob("*"))
            if sparse_dirs:
                reconstruction_path = sparse_dirs[0]
                logger.warning(
                    f"Default reconstruction path not found, using: {reconstruction_path}"
                )
            else:
                raise RuntimeError(
                    "COLMAP reconstruction failed - no output produced. "
                    "This may indicate insufficient feature matches between images."
                )

        # Log reconstruction statistics
        self._log_reconstruction_stats(reconstruction_path)

        return {
            "database": database_path,
            "sparse": reconstruction_path,
            "images": images_dir,
        }

    def _log_reconstruction_stats(self, model_path: Path):
        """Log statistics about the reconstruction."""
        try:
            # Check for binary or text format
            cameras_file = model_path / "cameras.bin"
            if not cameras_file.exists():
                cameras_file = model_path / "cameras.txt"

            images_file = model_path / "images.bin"
            if not images_file.exists():
                images_file = model_path / "images.txt"

            points_file = model_path / "points3D.bin"
            if not points_file.exists():
                points_file = model_path / "points3D.txt"

            if all(f.exists() for f in [cameras_file, images_file, points_file]):
                logger.info(f"Reconstruction saved to: {model_path}")
                logger.info(f"  Cameras: {cameras_file.name}")
                logger.info(f"  Images: {images_file.name}")
                logger.info(f"  Points3D: {points_file.name}")
        except Exception as e:
            logger.debug(f"Could not log reconstruction stats: {e}")

    def run_for_static_cameras(
        self,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run COLMAP for static multi-view camera setup.

        For static cameras, we:
        1. Use one representative frame from each camera
        2. Run COLMAP to get camera poses
        3. Apply those poses to all frames

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with paths to output files
        """
        experiment_name = experiment_name or self.config.experiment_name
        experiment_dir = self.config.processed_dir / experiment_name
        colmap_dir = experiment_dir / "colmap"

        # Find representative frames (first frame from each camera)
        input_dir = experiment_dir / "input"
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        # Create directory with representative frames
        representative_dir = colmap_dir / "representative_images"
        representative_dir.mkdir(parents=True, exist_ok=True)

        cam_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(cam_dirs)} camera directories")

        for cam_dir in cam_dirs:
            images_dir = cam_dir / "images"
            if not images_dir.exists():
                logger.warning(f"No images directory found in {cam_dir}")
                continue

            # Get first frame
            frames = sorted(images_dir.glob("*"))
            if not frames:
                logger.warning(f"No frames found in {images_dir}")
                continue

            first_frame = frames[0]

            # Create symlink with camera name prefix
            link_name = f"{cam_dir.name}_{first_frame.name}"
            link_path = representative_dir / link_name

            if link_path.exists():
                link_path.unlink()
            link_path.symlink_to(first_frame.resolve())

            logger.debug(f"Added representative frame: {link_name}")

        # Run COLMAP on representative frames
        colmap_output = self.run_pipeline(representative_dir, colmap_dir)

        return colmap_output


def main():
    """CLI entry point for testing COLMAP wrapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Run COLMAP pipeline")
    parser.add_argument("images_dir", type=Path, help="Directory containing images")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"],
                        default="exhaustive", help="Matcher type")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")

    args = parser.parse_args()

    config = PipelineConfig()
    config.colmap.matcher_type = args.matcher
    config.colmap.use_gpu = args.gpu

    wrapper = ColmapWrapper(config)
    results = wrapper.run_pipeline(args.images_dir, args.output_dir)

    print(f"\nCOLMAP completed. Results:")
    for key, path in results.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
