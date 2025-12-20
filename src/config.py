"""
Configuration module for the 4DGS Pipeline.

Provides centralized configuration management with sensible defaults
and validation for all pipeline parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import yaml
from loguru import logger


@dataclass
class FrameExtractionConfig:
    """Configuration for video frame extraction."""

    fps: float = 30.0  # Frames per second to extract
    quality: int = 95  # JPEG quality (1-100) or PNG compression level
    format: str = "png"  # Output format: 'png' or 'jpg'
    start_time: float = 0.0  # Start time in seconds
    duration: Optional[float] = None  # Duration to extract (None = full video)
    resize_width: Optional[int] = None  # Resize width (None = original)
    resize_height: Optional[int] = None  # Resize height (None = original)


@dataclass
class ColmapConfig:
    """Configuration for COLMAP camera pose estimation."""

    # Feature extraction settings
    camera_model: str = "OPENCV"  # Camera model: SIMPLE_PINHOLE, PINHOLE, OPENCV, etc.
    single_camera: bool = False  # Use same camera intrinsics for all images
    single_camera_per_folder: bool = True  # Same intrinsics per camera/folder

    # Matching settings
    matcher_type: str = "exhaustive"  # 'exhaustive', 'sequential', 'spatial'

    # Reconstruction settings
    use_gpu: bool = True
    gpu_index: int = 0

    # Quality settings
    max_image_size: int = 3200  # Maximum image dimension
    num_threads: int = -1  # -1 = use all available


@dataclass
class TrainingConfig:
    """Configuration for 4DGS training."""

    iterations: int = 30000
    save_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])
    test_iterations: List[int] = field(default_factory=lambda: [7000, 15000, 30000])

    # Learning rates
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # Densification
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002

    # Resolution
    resolution: int = -1  # -1 = use original resolution
    white_background: bool = False

    # Time-related (for 4D)
    time_duration: float = 1.0  # Normalized time duration


@dataclass
class RenderConfig:
    """Configuration for rendering."""

    output_format: str = "mp4"
    fps: int = 30
    codec: str = "libx264"
    crf: int = 18  # Quality factor (lower = better, 18 is visually lossless)

    # Camera path
    camera_path_type: str = "interpolate"  # 'interpolate', 'spiral', 'orbit'
    num_frames: int = 300  # Number of frames to render

    # Resolution
    width: Optional[int] = None  # None = use training resolution
    height: Optional[int] = None


@dataclass
class PipelineConfig:
    """Master configuration for the entire pipeline."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path.cwd())
    input_dir: Path = field(default_factory=lambda: Path("data/inputs"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    output_dir: Path = field(default_factory=lambda: Path("data/outputs"))

    # Sub-configurations
    frame_extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)
    colmap: ColmapConfig = field(default_factory=ColmapConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    # Pipeline options
    experiment_name: str = "experiment_001"
    skip_existing: bool = True  # Skip steps if output already exists
    verbose: bool = True

    def __post_init__(self):
        """Convert string paths to Path objects and resolve relative paths."""
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.processed_dir, str):
            self.processed_dir = Path(self.processed_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Make paths absolute if they're relative
        if not self.input_dir.is_absolute():
            self.input_dir = self.project_root / self.input_dir
        if not self.processed_dir.is_absolute():
            self.processed_dir = self.project_root / self.processed_dir
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir

    def get_experiment_dir(self) -> Path:
        """Get the directory for the current experiment."""
        return self.processed_dir / self.experiment_name

    def get_colmap_dir(self) -> Path:
        """Get the COLMAP output directory."""
        return self.get_experiment_dir() / "colmap"

    def get_model_dir(self) -> Path:
        """Get the trained model output directory."""
        return self.output_dir / self.experiment_name / "model"

    def get_render_dir(self) -> Path:
        """Get the render output directory."""
        return self.output_dir / self.experiment_name / "renders"

    def validate(self) -> bool:
        """Validate the configuration."""
        errors = []

        if not self.input_dir.exists():
            errors.append(f"Input directory does not exist: {self.input_dir}")

        if self.frame_extraction.format not in ["png", "jpg", "jpeg"]:
            errors.append(f"Invalid frame format: {self.frame_extraction.format}")

        if self.frame_extraction.quality < 1 or self.frame_extraction.quality > 100:
            errors.append(f"Quality must be between 1-100: {self.frame_extraction.quality}")

        if errors:
            for error in errors:
                logger.error(error)
            return False

        return True

    def create_directories(self):
        """Create all necessary output directories."""
        dirs_to_create = [
            self.processed_dir,
            self.output_dir,
            self.get_experiment_dir(),
            self.get_colmap_dir(),
            self.get_model_dir(),
            self.get_render_dir(),
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configurations
        frame_config = FrameExtractionConfig(**config_dict.pop('frame_extraction', {}))
        colmap_config = ColmapConfig(**config_dict.pop('colmap', {}))
        training_config = TrainingConfig(**config_dict.pop('training', {}))
        render_config = RenderConfig(**config_dict.pop('render', {}))

        return cls(
            frame_extraction=frame_config,
            colmap=colmap_config,
            training=training_config,
            render=render_config,
            **config_dict
        )

    def to_yaml(self, yaml_path: Path):
        """Save configuration to a YAML file."""
        config_dict = {
            'project_root': str(self.project_root),
            'input_dir': str(self.input_dir),
            'processed_dir': str(self.processed_dir),
            'output_dir': str(self.output_dir),
            'experiment_name': self.experiment_name,
            'skip_existing': self.skip_existing,
            'verbose': self.verbose,
            'frame_extraction': {
                'fps': self.frame_extraction.fps,
                'quality': self.frame_extraction.quality,
                'format': self.frame_extraction.format,
                'start_time': self.frame_extraction.start_time,
                'duration': self.frame_extraction.duration,
                'resize_width': self.frame_extraction.resize_width,
                'resize_height': self.frame_extraction.resize_height,
            },
            'colmap': {
                'camera_model': self.colmap.camera_model,
                'single_camera': self.colmap.single_camera,
                'single_camera_per_folder': self.colmap.single_camera_per_folder,
                'matcher_type': self.colmap.matcher_type,
                'use_gpu': self.colmap.use_gpu,
                'gpu_index': self.colmap.gpu_index,
                'max_image_size': self.colmap.max_image_size,
                'num_threads': self.colmap.num_threads,
            },
            'training': {
                'iterations': self.training.iterations,
                'save_iterations': self.training.save_iterations,
                'test_iterations': self.training.test_iterations,
                'resolution': self.training.resolution,
                'white_background': self.training.white_background,
            },
            'render': {
                'output_format': self.render.output_format,
                'fps': self.render.fps,
                'camera_path_type': self.render.camera_path_type,
                'num_frames': self.render.num_frames,
            },
        }

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to: {yaml_path}")
