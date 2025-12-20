"""
4D Gaussian Splatting MVP Pipeline

A complete automation wrapper for processing multi-view videos
into 4D Gaussian Splatting reconstructions.
"""

__version__ = "0.1.0"
__author__ = "4DGS Pipeline Team"

from .config import PipelineConfig
from .preprocess import FrameExtractor
from .colmap_wrapper import ColmapWrapper
from .convert_colmap import ColmapConverter
from .train_wrapper import TrainWrapper
from .render_wrapper import RenderWrapper

__all__ = [
    "PipelineConfig",
    "FrameExtractor",
    "ColmapWrapper",
    "ColmapConverter",
    "TrainWrapper",
    "RenderWrapper",
]
