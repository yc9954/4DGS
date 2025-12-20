"""Tests for the configuration module."""

import pytest
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PipelineConfig,
    FrameExtractionConfig,
    ColmapConfig,
    TrainingConfig,
    RenderConfig
)


class TestFrameExtractionConfig:
    """Tests for FrameExtractionConfig."""

    def test_default_values(self):
        config = FrameExtractionConfig()
        assert config.fps == 30.0
        assert config.quality == 95
        assert config.format == "png"

    def test_custom_values(self):
        config = FrameExtractionConfig(fps=60.0, format="jpg", quality=80)
        assert config.fps == 60.0
        assert config.format == "jpg"
        assert config.quality == 80


class TestColmapConfig:
    """Tests for ColmapConfig."""

    def test_default_values(self):
        config = ColmapConfig()
        assert config.camera_model == "OPENCV"
        assert config.matcher_type == "exhaustive"
        assert config.use_gpu is True

    def test_gpu_disabled(self):
        config = ColmapConfig(use_gpu=False)
        assert config.use_gpu is False


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.iterations == 30000
        assert 7000 in config.save_iterations
        assert 30000 in config.save_iterations

    def test_custom_iterations(self):
        config = TrainingConfig(iterations=10000)
        assert config.iterations == 10000


class TestRenderConfig:
    """Tests for RenderConfig."""

    def test_default_values(self):
        config = RenderConfig()
        assert config.output_format == "mp4"
        assert config.fps == 30
        assert config.camera_path_type == "interpolate"


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        config = PipelineConfig()
        assert config.experiment_name == "experiment_001"
        assert config.skip_existing is True

    def test_path_resolution(self):
        config = PipelineConfig(project_root=Path("/test/root"))
        # Paths should be resolved relative to project_root
        assert config.input_dir == Path("/test/root/data/inputs")
        assert config.processed_dir == Path("/test/root/data/processed")

    def test_experiment_directories(self):
        config = PipelineConfig(
            project_root=Path("/test"),
            experiment_name="my_exp"
        )
        assert "my_exp" in str(config.get_experiment_dir())
        assert "my_exp" in str(config.get_colmap_dir())
        assert "my_exp" in str(config.get_model_dir())

    def test_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"

            # Create and save config
            original = PipelineConfig(
                experiment_name="test_exp",
                project_root=Path(tmpdir)
            )
            original.frame_extraction.fps = 60.0
            original.training.iterations = 10000
            original.to_yaml(yaml_path)

            # Load config
            loaded = PipelineConfig.from_yaml(yaml_path)

            assert loaded.experiment_name == "test_exp"
            assert loaded.frame_extraction.fps == 60.0
            assert loaded.training.iterations == 10000

    def test_create_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                project_root=Path(tmpdir),
                experiment_name="test"
            )
            config.create_directories()

            assert config.processed_dir.exists()
            assert config.output_dir.exists()
            assert config.get_experiment_dir().exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
