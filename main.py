#!/usr/bin/env python3
"""
4D Gaussian Splatting Pipeline - Main Entry Point

This script orchestrates the complete pipeline from raw multi-view videos
to rendered 4D Gaussian Splatting output.

Pipeline Steps:
1. Frame Extraction: Extract synchronized frames from multi-view videos
2. COLMAP Processing: Estimate camera poses using Structure-from-Motion
3. Data Conversion: Convert COLMAP output to transforms.json format
4. Training: Train the 4DGS model
5. Rendering: Generate output video from trained model

Usage:
    # Full pipeline
    python main.py --input data/inputs --experiment my_scene

    # Individual steps
    python main.py --input data/inputs --experiment my_scene --step preprocess
    python main.py --input data/inputs --experiment my_scene --step colmap
    python main.py --input data/inputs --experiment my_scene --step train
    python main.py --input data/inputs --experiment my_scene --step render

    # From config file
    python main.py --config configs/my_config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import PipelineConfig
from src.preprocess import FrameExtractor
from src.colmap_wrapper import ColmapWrapper
from src.convert_colmap import ColmapConverter
from src.train_wrapper import TrainWrapper, MockTrainWrapper
from src.render_wrapper import RenderWrapper, CameraPathGenerator


class Pipeline:
    """
    Main 4DGS Pipeline orchestrator.

    Coordinates all pipeline steps from raw video input to final rendering.
    """

    # Pipeline step definitions
    STEPS = ["preprocess", "colmap", "convert", "train", "render"]

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration object
        """
        self.config = config

        # Initialize components (lazy loading)
        self._frame_extractor: Optional[FrameExtractor] = None
        self._colmap_wrapper: Optional[ColmapWrapper] = None
        self._train_wrapper: Optional[TrainWrapper] = None
        self._render_wrapper: Optional[RenderWrapper] = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        # Remove default handler
        logger.remove()

        # Console logging
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        if self.config.verbose:
            logger.add(sys.stderr, format=log_format, level="DEBUG")
        else:
            logger.add(sys.stderr, format=log_format, level="INFO")

        # File logging
        log_dir = self.config.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        logger.add(str(log_file), format=log_format, level="DEBUG")

    @property
    def frame_extractor(self) -> FrameExtractor:
        """Get or create frame extractor instance."""
        if self._frame_extractor is None:
            self._frame_extractor = FrameExtractor(self.config)
        return self._frame_extractor

    @property
    def colmap_wrapper(self) -> ColmapWrapper:
        """Get or create COLMAP wrapper instance."""
        if self._colmap_wrapper is None:
            self._colmap_wrapper = ColmapWrapper(self.config)
        return self._colmap_wrapper

    @property
    def train_wrapper(self) -> TrainWrapper:
        """Get or create training wrapper instance."""
        if self._train_wrapper is None:
            self._train_wrapper = TrainWrapper(self.config)
        return self._train_wrapper

    @property
    def render_wrapper(self) -> RenderWrapper:
        """Get or create render wrapper instance."""
        if self._render_wrapper is None:
            self._render_wrapper = RenderWrapper(self.config)
        return self._render_wrapper

    def step_preprocess(self) -> dict:
        """
        Step 1: Extract frames from videos.

        Returns:
            Dictionary with extraction results
        """
        logger.info("=" * 60)
        logger.info("STEP 1: FRAME EXTRACTION")
        logger.info("=" * 60)

        # Discover and extract videos
        results = self.frame_extractor.extract_all_videos()

        # Create symlinks for COLMAP
        self.frame_extractor.create_frame_symlinks(results)

        logger.info(f"Extracted frames from {len(results)} cameras")

        return results

    def step_colmap(self) -> dict:
        """
        Step 2: Run COLMAP for camera pose estimation.

        Returns:
            Dictionary with COLMAP output paths
        """
        logger.info("=" * 60)
        logger.info("STEP 2: CAMERA POSE ESTIMATION (COLMAP)")
        logger.info("=" * 60)

        # Run COLMAP for static camera setup
        results = self.colmap_wrapper.run_for_static_cameras()

        logger.info(f"COLMAP sparse model saved to: {results['sparse']}")

        return results

    def step_convert(self) -> Path:
        """
        Step 3: Convert COLMAP output to transforms.json.

        Returns:
            Path to the transforms.json file
        """
        logger.info("=" * 60)
        logger.info("STEP 3: CONVERT COLMAP TO TRANSFORMS.JSON")
        logger.info("=" * 60)

        # Find COLMAP output
        colmap_dir = self.config.get_colmap_dir()
        sparse_dir = colmap_dir / "sparse" / "0"

        if not sparse_dir.exists():
            # Try alternative paths
            sparse_dirs = list((colmap_dir / "sparse").glob("*"))
            if sparse_dirs:
                sparse_dir = sparse_dirs[0]
            else:
                raise FileNotFoundError(
                    f"COLMAP sparse reconstruction not found in {colmap_dir}/sparse/"
                )

        # Convert to transforms.json
        converter = ColmapConverter(sparse_dir)
        transforms_path = self.config.get_experiment_dir() / "transforms.json"

        # Get frame counts for time annotation
        input_dir = self.config.get_experiment_dir() / "input"
        cam_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if cam_dirs:
            images_dir = cam_dirs[0] / "images"
            total_frames = len(list(images_dir.glob("*")))
        else:
            total_frames = 1

        # Convert with 4DGS-specific format
        converter.convert_for_4dgs(
            output_dir=self.config.get_experiment_dir(),
            cameras_per_frame=len(cam_dirs),
            total_frames=total_frames
        )

        logger.info(f"Saved transforms.json to: {transforms_path}")

        return transforms_path

    def step_train(self, mock: bool = False) -> Path:
        """
        Step 4: Train the 4DGS model.

        Args:
            mock: If True, use mock training (for testing)

        Returns:
            Path to the trained model
        """
        logger.info("=" * 60)
        logger.info("STEP 4: 4DGS TRAINING")
        logger.info("=" * 60)

        # Use mock wrapper if requested
        if mock:
            logger.warning("Using MOCK training mode")
            self._train_wrapper = MockTrainWrapper(self.config)

        # Training source path
        source_path = self.config.get_experiment_dir()

        # Verify transforms.json exists
        transforms_path = source_path / "transforms.json"
        if not transforms_path.exists():
            logger.warning(
                f"transforms.json not found at {transforms_path}. "
                "Running conversion step first..."
            )
            self.step_convert()

        # Progress callback
        def on_progress(progress):
            if progress.current_iteration % 1000 == 0:
                logger.info(
                    f"Training: {progress.current_iteration}/{progress.total_iterations} "
                    f"({progress.progress_percent:.1f}%) "
                    f"Loss: {progress.loss:.4f if progress.loss else 'N/A'}"
                )

        # Run training
        model_path = self.train_wrapper.train(
            source_path=source_path,
            progress_callback=on_progress
        )

        logger.info(f"Model saved to: {model_path}")

        return model_path

    def step_render(self) -> Path:
        """
        Step 5: Render the trained model to video.

        Returns:
            Path to the output video
        """
        logger.info("=" * 60)
        logger.info("STEP 5: RENDERING")
        logger.info("=" * 60)

        # Find model path
        model_path = self.config.get_model_dir()

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Please run training first."
            )

        # Find transforms.json
        transforms_path = self.config.get_experiment_dir() / "transforms.json"

        # Render
        output_video = self.render_wrapper.render_and_encode(
            model_path=model_path,
            training_transforms=transforms_path
        )

        logger.info(f"Output video: {output_video}")

        return output_video

    def run(
        self,
        steps: Optional[List[str]] = None,
        mock_training: bool = False
    ) -> dict:
        """
        Run the pipeline.

        Args:
            steps: List of steps to run (None = all steps)
            mock_training: Use mock training mode

        Returns:
            Dictionary with results from each step
        """
        if steps is None:
            steps = self.STEPS

        # Validate steps
        for step in steps:
            if step not in self.STEPS:
                raise ValueError(
                    f"Unknown step: {step}. Available: {self.STEPS}"
                )

        # Create output directories
        self.config.create_directories()

        logger.info("=" * 60)
        logger.info("4D GAUSSIAN SPLATTING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Input: {self.config.input_dir}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"Steps: {steps}")
        logger.info("=" * 60)

        results = {}
        start_time = datetime.now()

        try:
            if "preprocess" in steps:
                results["preprocess"] = self.step_preprocess()

            if "colmap" in steps:
                results["colmap"] = self.step_colmap()

            if "convert" in steps:
                results["convert"] = self.step_convert()

            if "train" in steps:
                results["train"] = self.step_train(mock=mock_training)

            if "render" in steps:
                results["render"] = self.step_render()

            elapsed = datetime.now() - start_time

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Total time: {elapsed}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

        return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="4D Gaussian Splatting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from videos
  python main.py --input ./videos --experiment my_scene

  # Only preprocess and COLMAP
  python main.py --input ./videos --experiment my_scene --step preprocess --step colmap

  # Resume from training
  python main.py --experiment my_scene --step train --step render

  # Use config file
  python main.py --config configs/my_config.yaml

  # Mock training (for testing pipeline)
  python main.py --input ./videos --experiment test --mock
        """
    )

    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/outputs"),
        help="Output directory (default: data/outputs)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not specified)"
    )

    # Pipeline control
    parser.add_argument(
        "--step", "-s",
        action="append",
        dest="steps",
        choices=Pipeline.STEPS,
        help="Specific step(s) to run (can be specified multiple times)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Load configuration from YAML file"
    )

    # Frame extraction options
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame extraction FPS (default: 30)"
    )
    parser.add_argument(
        "--frame-format",
        choices=["png", "jpg"],
        default="png",
        help="Frame output format (default: png)"
    )

    # COLMAP options
    parser.add_argument(
        "--matcher",
        choices=["exhaustive", "sequential"],
        default="exhaustive",
        help="COLMAP matcher type (default: exhaustive)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU for COLMAP"
    )

    # Training options
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Training iterations (default: 30000)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=-1,
        help="Training resolution (-1 for original, default: -1)"
    )

    # Rendering options
    parser.add_argument(
        "--render-path",
        choices=["interpolate", "spiral", "orbit"],
        default="interpolate",
        help="Render camera path type (default: interpolate)"
    )
    parser.add_argument(
        "--render-frames",
        type=int,
        default=300,
        help="Number of frames to render (default: 300)"
    )

    # Other options
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock training (for testing pipeline without GPU)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing outputs, reprocess everything"
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Create configuration from command-line arguments."""
    # Load from file if specified
    if args.config and args.config.exists():
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # Override with command-line arguments
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output

    # Auto-generate experiment name if not specified
    if args.experiment:
        config.experiment_name = args.experiment
    elif config.experiment_name == "experiment_001":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.experiment_name = f"exp_{timestamp}"

    # Frame extraction
    config.frame_extraction.fps = args.fps
    config.frame_extraction.format = args.frame_format

    # COLMAP
    config.colmap.matcher_type = args.matcher
    config.colmap.use_gpu = not args.no_gpu

    # Training
    config.training.iterations = args.iterations
    config.training.resolution = args.resolution

    # Rendering
    config.render.camera_path_type = args.render_path
    config.render.num_frames = args.render_frames

    # Other
    config.verbose = args.verbose
    config.skip_existing = not args.no_skip

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    config = create_config_from_args(args)

    # Validate configuration
    if args.input is None and not args.config:
        # Check if running partial pipeline without input
        if args.steps and all(s in ["train", "render"] for s in args.steps):
            # These steps don't need input videos
            pass
        else:
            logger.error("Input directory is required. Use --input or --config.")
            sys.exit(1)

    # Create and run pipeline
    pipeline = Pipeline(config)

    try:
        results = pipeline.run(
            steps=args.steps,
            mock_training=args.mock
        )

        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)

        if "preprocess" in results:
            print(f"Cameras processed: {len(results['preprocess'])}")

        if "colmap" in results:
            print(f"COLMAP output: {results['colmap']['sparse']}")

        if "convert" in results:
            print(f"Transforms file: {results['convert']}")

        if "train" in results:
            print(f"Trained model: {results['train']}")

        if "render" in results:
            print(f"Output video: {results['render']}")

        print("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
