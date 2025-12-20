"""Tests for the COLMAP conversion module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.convert_colmap import Camera, Image, ColmapConverter


class TestCamera:
    """Tests for Camera dataclass."""

    def test_simple_pinhole(self):
        camera = Camera(
            id=1,
            model="SIMPLE_PINHOLE",
            width=800,
            height=600,
            params=[500.0, 400.0, 300.0]  # f, cx, cy
        )
        assert camera.fx == 500.0
        assert camera.fy == 500.0  # Same as fx for SIMPLE_PINHOLE
        assert camera.cx == 400.0
        assert camera.cy == 300.0

    def test_pinhole(self):
        camera = Camera(
            id=1,
            model="PINHOLE",
            width=800,
            height=600,
            params=[500.0, 550.0, 400.0, 300.0]  # fx, fy, cx, cy
        )
        assert camera.fx == 500.0
        assert camera.fy == 550.0
        assert camera.cx == 400.0
        assert camera.cy == 300.0

    def test_opencv(self):
        camera = Camera(
            id=1,
            model="OPENCV",
            width=1920,
            height=1080,
            params=[1000.0, 1000.0, 960.0, 540.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert camera.fx == 1000.0
        assert camera.fy == 1000.0
        assert camera.cx == 960.0
        assert camera.cy == 540.0

    def test_camera_angle(self):
        camera = Camera(
            id=1,
            model="PINHOLE",
            width=800,
            height=600,
            params=[500.0, 500.0, 400.0, 300.0]
        )
        # FOV should be reasonable
        assert 0.5 < camera.camera_angle_x < 2.0
        assert 0.5 < camera.camera_angle_y < 2.0


class TestImage:
    """Tests for Image dataclass."""

    def test_identity_rotation(self):
        # Identity quaternion (1, 0, 0, 0) should give identity rotation
        image = Image(
            id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([0.0, 0.0, 0.0]),
            camera_id=1,
            name="test.png"
        )
        R = image.qvec2rotmat()
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90_degree_rotation(self):
        # 90 degree rotation around Z axis
        # q = (cos(45°), 0, 0, sin(45°))
        angle = np.pi / 2
        image = Image(
            id=1,
            qvec=np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]),
            tvec=np.array([0.0, 0.0, 0.0]),
            camera_id=1,
            name="test.png"
        )
        R = image.qvec2rotmat()

        # Should rotate X to Y, Y to -X
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_transform_matrix_shape(self):
        image = Image(
            id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([1.0, 2.0, 3.0]),
            camera_id=1,
            name="test.png"
        )
        T = image.get_transform_matrix()

        assert T.shape == (4, 4)
        assert T[3, 3] == 1.0
        np.testing.assert_array_equal(T[3, :3], [0, 0, 0])

    def test_transform_matrix_inversion(self):
        # Test that transform is properly inverted from COLMAP convention
        image = Image(
            id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity rotation
            tvec=np.array([1.0, 2.0, 3.0]),
            camera_id=1,
            name="test.png"
        )
        T = image.get_transform_matrix()

        # Camera position should be at -tvec for identity rotation
        np.testing.assert_array_almost_equal(T[:3, 3], [-1.0, -2.0, -3.0])


class TestColmapConverterIntegration:
    """Integration tests for ColmapConverter."""

    def test_transforms_json_format(self):
        """Test that output conforms to expected transforms.json format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal COLMAP text format files
            (tmpdir / "cameras.txt").write_text(
                "# Camera list\n"
                "1 PINHOLE 800 600 500 500 400 300\n"
            )
            (tmpdir / "images.txt").write_text(
                "# Image list\n"
                "1 1 0 0 0 0 0 0 1 test_image.png\n"
                "0.0 0.0 -1\n"  # 2D points line (required but we skip it)
            )
            (tmpdir / "points3D.txt").write_text(
                "# Points3D\n"
            )

            converter = ColmapConverter(tmpdir)
            transforms = converter.convert()

            # Check required fields
            assert "camera_angle_x" in transforms
            assert "camera_angle_y" in transforms
            assert "fl_x" in transforms
            assert "fl_y" in transforms
            assert "frames" in transforms
            assert len(transforms["frames"]) == 1

            frame = transforms["frames"][0]
            assert "file_path" in frame
            assert "transform_matrix" in frame
            assert len(frame["transform_matrix"]) == 4
            assert len(frame["transform_matrix"][0]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
