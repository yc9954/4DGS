"""
COLMAP to transforms.json Converter for 4DGS Pipeline.

Converts COLMAP's binary/text output format to the standard transforms.json
format used by NeRF and Gaussian Splatting implementations.

The transforms.json format contains:
- Camera intrinsics (focal length, principal point)
- Per-image camera extrinsics (rotation + translation as 4x4 matrix)
- Image file paths

This module handles both binary (.bin) and text (.txt) COLMAP formats.
"""

import struct
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

from loguru import logger


# ============================================================================
# COLMAP Data Structures
# ============================================================================

@dataclass
class Camera:
    """
    Represents a COLMAP camera (intrinsic parameters).

    COLMAP camera models:
    - SIMPLE_PINHOLE: f, cx, cy
    - PINHOLE: fx, fy, cx, cy
    - SIMPLE_RADIAL: f, cx, cy, k1
    - RADIAL: f, cx, cy, k1, k2
    - OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    - OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
    """

    id: int
    model: str
    width: int
    height: int
    params: List[float]

    @property
    def fx(self) -> float:
        """Focal length in x direction."""
        if self.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            return self.params[0]
        return self.params[0]  # PINHOLE, OPENCV, etc.

    @property
    def fy(self) -> float:
        """Focal length in y direction."""
        if self.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            return self.params[0]  # Same as fx for simple models
        return self.params[1]  # PINHOLE, OPENCV, etc.

    @property
    def cx(self) -> float:
        """Principal point x coordinate."""
        if self.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            return self.params[1]
        return self.params[2]  # PINHOLE, OPENCV, etc.

    @property
    def cy(self) -> float:
        """Principal point y coordinate."""
        if self.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            return self.params[2]
        return self.params[3]  # PINHOLE, OPENCV, etc.

    @property
    def camera_angle_x(self) -> float:
        """Horizontal field of view in radians."""
        return 2 * np.arctan(self.width / (2 * self.fx))

    @property
    def camera_angle_y(self) -> float:
        """Vertical field of view in radians."""
        return 2 * np.arctan(self.height / (2 * self.fy))


@dataclass
class Image:
    """
    Represents a COLMAP image (extrinsic parameters + metadata).

    The extrinsics are stored as:
    - Quaternion (qw, qx, qy, qz) for rotation
    - Translation vector (tx, ty, tz)

    Note: COLMAP stores camera-to-world transformation,
    which we convert to the standard transforms.json format.
    """

    id: int
    qvec: np.ndarray  # Quaternion (qw, qx, qy, qz)
    tvec: np.ndarray  # Translation (tx, ty, tz)
    camera_id: int
    name: str
    xys: Optional[np.ndarray] = None  # 2D keypoints
    point3D_ids: Optional[np.ndarray] = None  # Associated 3D point IDs

    def qvec2rotmat(self) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.

        COLMAP uses Hamilton quaternion convention (w, x, y, z).

        Returns:
            3x3 rotation matrix
        """
        qw, qx, qy, qz = self.qvec

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        return R

    def get_transform_matrix(self) -> np.ndarray:
        """
        Get the 4x4 camera-to-world transformation matrix.

        COLMAP stores world-to-camera, so we invert it.

        The resulting matrix transforms points from camera space to world space:
        p_world = T @ p_camera

        Returns:
            4x4 transformation matrix (camera-to-world)
        """
        # COLMAP rotation: world-to-camera
        R = self.qvec2rotmat()
        t = self.tvec

        # World-to-camera transformation
        # p_camera = R @ p_world + t
        # Therefore: p_world = R^T @ (p_camera - t) = R^T @ p_camera - R^T @ t

        # Camera-to-world transformation
        R_c2w = R.T
        t_c2w = -R.T @ t

        # Build 4x4 matrix
        transform = np.eye(4)
        transform[:3, :3] = R_c2w
        transform[:3, 3] = t_c2w

        return transform


@dataclass
class Point3D:
    """Represents a COLMAP 3D point."""

    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


# ============================================================================
# COLMAP File Readers
# ============================================================================

class ColmapReader:
    """
    Reads COLMAP model files (both binary and text formats).

    This class handles parsing of:
    - cameras.bin / cameras.txt
    - images.bin / images.txt
    - points3D.bin / points3D.txt
    """

    # Camera model name to ID mapping
    CAMERA_MODEL_IDS = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5,
        "FULL_OPENCV": 6,
        "FOV": 7,
        "SIMPLE_RADIAL_FISHEYE": 8,
        "RADIAL_FISHEYE": 9,
        "THIN_PRISM_FISHEYE": 10,
    }

    # Camera model ID to name mapping
    CAMERA_MODEL_NAMES = {v: k for k, v in CAMERA_MODEL_IDS.items()}

    # Number of parameters for each camera model
    CAMERA_MODEL_PARAMS = {
        "SIMPLE_PINHOLE": 3,
        "PINHOLE": 4,
        "SIMPLE_RADIAL": 4,
        "RADIAL": 5,
        "OPENCV": 8,
        "OPENCV_FISHEYE": 8,
        "FULL_OPENCV": 12,
        "FOV": 5,
        "SIMPLE_RADIAL_FISHEYE": 4,
        "RADIAL_FISHEYE": 5,
        "THIN_PRISM_FISHEYE": 12,
    }

    def __init__(self, model_path: Path):
        """
        Initialize the reader with a path to the COLMAP model.

        Args:
            model_path: Directory containing COLMAP output files
        """
        self.model_path = Path(model_path)
        self.is_binary = self._detect_format()

    def _detect_format(self) -> bool:
        """Detect if the model is in binary or text format."""
        if (self.model_path / "cameras.bin").exists():
            return True
        elif (self.model_path / "cameras.txt").exists():
            return False
        else:
            raise FileNotFoundError(
                f"No COLMAP model found at {self.model_path}. "
                "Expected cameras.bin/txt, images.bin/txt, points3D.bin/txt"
            )

    def read_cameras(self) -> Dict[int, Camera]:
        """
        Read camera intrinsics from COLMAP output.

        Returns:
            Dictionary mapping camera ID to Camera object
        """
        if self.is_binary:
            return self._read_cameras_binary()
        return self._read_cameras_text()

    def _read_cameras_binary(self) -> Dict[int, Camera]:
        """Read cameras from binary format."""
        cameras = {}
        cameras_file = self.model_path / "cameras.bin"

        with open(cameras_file, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<I", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]

                model_name = self.CAMERA_MODEL_NAMES[model_id]
                num_params = self.CAMERA_MODEL_PARAMS[model_name]
                params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))

                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model_name,
                    width=width,
                    height=height,
                    params=list(params)
                )

        logger.debug(f"Read {len(cameras)} cameras from binary format")
        return cameras

    def _read_cameras_text(self) -> Dict[int, Camera]:
        """Read cameras from text format."""
        cameras = {}
        cameras_file = self.model_path / "cameras.txt"

        with open(cameras_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]

                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params
                )

        logger.debug(f"Read {len(cameras)} cameras from text format")
        return cameras

    def read_images(self) -> Dict[int, Image]:
        """
        Read image extrinsics from COLMAP output.

        Returns:
            Dictionary mapping image ID to Image object
        """
        if self.is_binary:
            return self._read_images_binary()
        return self._read_images_text()

    def _read_images_binary(self) -> Dict[int, Image]:
        """Read images from binary format."""
        images = {}
        images_file = self.model_path / "images.bin"

        with open(images_file, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]
                qvec = np.array(struct.unpack("<4d", f.read(32)))
                tvec = np.array(struct.unpack("<3d", f.read(24)))
                camera_id = struct.unpack("<I", f.read(4))[0]

                # Read image name (null-terminated string)
                name_chars = []
                while True:
                    char = f.read(1)
                    if char == b"\x00":
                        break
                    name_chars.append(char.decode("utf-8"))
                name = "".join(name_chars)

                # Read 2D points
                num_points2D = struct.unpack("<Q", f.read(8))[0]
                xys = np.zeros((num_points2D, 2))
                point3D_ids = np.zeros(num_points2D, dtype=np.int64)

                for i in range(num_points2D):
                    xys[i] = struct.unpack("<2d", f.read(16))
                    point3D_ids[i] = struct.unpack("<q", f.read(8))[0]

                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=name,
                    xys=xys,
                    point3D_ids=point3D_ids
                )

        logger.debug(f"Read {len(images)} images from binary format")
        return images

    def _read_images_text(self) -> Dict[int, Image]:
        """Read images from text format."""
        images = {}
        images_file = self.model_path / "images.txt"

        with open(images_file, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            # Parse image line
            parts = line.split()
            image_id = int(parts[0])
            qvec = np.array([float(parts[j]) for j in range(1, 5)])
            tvec = np.array([float(parts[j]) for j in range(5, 8)])
            camera_id = int(parts[8])
            name = parts[9]

            # Next line contains 2D points (skip for now)
            i += 2

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name
            )

        logger.debug(f"Read {len(images)} images from text format")
        return images

    def read_points3D(self) -> Dict[int, Point3D]:
        """
        Read 3D points from COLMAP output.

        Returns:
            Dictionary mapping point ID to Point3D object
        """
        if self.is_binary:
            return self._read_points3D_binary()
        return self._read_points3D_text()

    def _read_points3D_binary(self) -> Dict[int, Point3D]:
        """Read 3D points from binary format."""
        points = {}
        points_file = self.model_path / "points3D.bin"

        with open(points_file, "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_points):
                point_id = struct.unpack("<Q", f.read(8))[0]
                xyz = np.array(struct.unpack("<3d", f.read(24)))
                rgb = np.array(struct.unpack("<3B", f.read(3)))
                error = struct.unpack("<d", f.read(8))[0]

                track_length = struct.unpack("<Q", f.read(8))[0]
                image_ids = np.zeros(track_length, dtype=np.int64)
                point2D_idxs = np.zeros(track_length, dtype=np.int64)

                for i in range(track_length):
                    image_ids[i] = struct.unpack("<I", f.read(4))[0]
                    point2D_idxs[i] = struct.unpack("<I", f.read(4))[0]

                points[point_id] = Point3D(
                    id=point_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs
                )

        logger.debug(f"Read {len(points)} 3D points from binary format")
        return points

    def _read_points3D_text(self) -> Dict[int, Point3D]:
        """Read 3D points from text format."""
        points = {}
        points_file = self.model_path / "points3D.txt"

        with open(points_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                point_id = int(parts[0])
                xyz = np.array([float(parts[i]) for i in range(1, 4)])
                rgb = np.array([int(parts[i]) for i in range(4, 7)])
                error = float(parts[7])

                # Track (image_id, point2D_idx pairs)
                track_parts = parts[8:]
                track_length = len(track_parts) // 2
                image_ids = np.array([int(track_parts[2*i]) for i in range(track_length)])
                point2D_idxs = np.array([int(track_parts[2*i+1]) for i in range(track_length)])

                points[point_id] = Point3D(
                    id=point_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs
                )

        logger.debug(f"Read {len(points)} 3D points from text format")
        return points


# ============================================================================
# Transforms.json Converter
# ============================================================================

class ColmapConverter:
    """
    Converts COLMAP output to transforms.json format.

    The transforms.json format is the standard format used by
    NeRF implementations and Gaussian Splatting methods.

    Format:
    {
        "camera_angle_x": float,  # Horizontal FOV in radians
        "camera_angle_y": float,  # Vertical FOV in radians
        "fl_x": float,  # Focal length x
        "fl_y": float,  # Focal length y
        "cx": float,  # Principal point x
        "cy": float,  # Principal point y
        "w": int,  # Image width
        "h": int,  # Image height
        "frames": [
            {
                "file_path": str,  # Relative path to image
                "transform_matrix": [[4x4 matrix]],  # Camera-to-world transform
                "time": float,  # (For 4DGS) Normalized timestamp
            },
            ...
        ]
    }
    """

    def __init__(self, model_path: Path):
        """
        Initialize the converter.

        Args:
            model_path: Path to COLMAP model directory
        """
        self.model_path = Path(model_path)
        self.reader = ColmapReader(model_path)

    def convert(
        self,
        output_path: Optional[Path] = None,
        image_path_prefix: str = "./images/",
        include_points: bool = False,
        time_mapping: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Convert COLMAP model to transforms.json format.

        Args:
            output_path: Where to save transforms.json (None = don't save)
            image_path_prefix: Prefix for image file paths in output
            include_points: Whether to include 3D points (for visualization)
            time_mapping: Optional mapping of image names to time values (for 4DGS)

        Returns:
            Dictionary in transforms.json format
        """
        # Read COLMAP data
        cameras = self.reader.read_cameras()
        images = self.reader.read_images()

        if not cameras:
            raise ValueError("No cameras found in COLMAP model")
        if not images:
            raise ValueError("No images found in COLMAP model")

        logger.info(f"Converting {len(images)} images with {len(cameras)} camera(s)")

        # Get the first camera for global intrinsics
        # (Gaussian Splatting typically assumes shared intrinsics)
        first_camera = next(iter(cameras.values()))

        # Build transforms dictionary
        transforms = {
            "camera_angle_x": float(first_camera.camera_angle_x),
            "camera_angle_y": float(first_camera.camera_angle_y),
            "fl_x": float(first_camera.fx),
            "fl_y": float(first_camera.fy),
            "cx": float(first_camera.cx),
            "cy": float(first_camera.cy),
            "w": first_camera.width,
            "h": first_camera.height,
            "aabb_scale": 16,  # Scene bounding box scale (NeRF convention)
            "frames": []
        }

        # Process each image
        for image_id, image in sorted(images.items(), key=lambda x: x[1].name):
            camera = cameras[image.camera_id]

            # Get camera-to-world transform
            transform_matrix = image.get_transform_matrix()

            # Build frame entry
            frame = {
                "file_path": f"{image_path_prefix}{image.name}",
                "transform_matrix": transform_matrix.tolist(),
            }

            # Add per-camera intrinsics if different from first camera
            if camera.id != first_camera.id:
                frame["fl_x"] = float(camera.fx)
                frame["fl_y"] = float(camera.fy)
                frame["cx"] = float(camera.cx)
                frame["cy"] = float(camera.cy)
                frame["w"] = camera.width
                frame["h"] = camera.height

            # Add time value for 4DGS if provided
            if time_mapping is not None:
                if image.name in time_mapping:
                    frame["time"] = time_mapping[image.name]
                else:
                    # Try to extract time from frame number in filename
                    frame["time"] = self._extract_time_from_name(image.name)

            transforms["frames"].append(frame)

        # Optionally include 3D points
        if include_points:
            points3D = self.reader.read_points3D()
            transforms["points"] = [
                {
                    "xyz": point.xyz.tolist(),
                    "rgb": point.rgb.tolist()
                }
                for point in points3D.values()
            ]

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(transforms, f, indent=2)

            logger.info(f"Saved transforms.json to {output_path}")

        return transforms

    def _extract_time_from_name(self, name: str) -> float:
        """
        Extract normalized time from image filename.

        Assumes filenames like 'frame_000001.png' where the number
        represents frame index.

        Args:
            name: Image filename

        Returns:
            Normalized time value [0, 1]
        """
        import re

        # Try to extract frame number
        match = re.search(r'(\d+)', name)
        if match:
            # Will need to normalize later based on total frames
            return float(match.group(1))
        return 0.0

    def convert_for_4dgs(
        self,
        output_dir: Path,
        cameras_per_frame: int,
        total_frames: int,
        image_path_prefix: str = "./images/"
    ) -> Dict[str, Any]:
        """
        Convert COLMAP model to 4DGS format with time annotations.

        For 4D Gaussian Splatting, we need:
        - Static camera poses (from COLMAP)
        - Time values for each frame

        This method handles the case where we have multiple cameras
        capturing the same time frame, and multiple time frames.

        Args:
            output_dir: Directory to save output files
            cameras_per_frame: Number of cameras (views) per time frame
            total_frames: Total number of time frames
            image_path_prefix: Prefix for image paths

        Returns:
            Transforms dictionary
        """
        cameras = self.reader.read_cameras()
        images = self.reader.read_images()

        logger.info(
            f"Converting for 4DGS: {cameras_per_frame} cameras x {total_frames} frames"
        )

        # Get camera intrinsics
        first_camera = next(iter(cameras.values()))

        transforms = {
            "camera_angle_x": float(first_camera.camera_angle_x),
            "camera_angle_y": float(first_camera.camera_angle_y),
            "fl_x": float(first_camera.fx),
            "fl_y": float(first_camera.fy),
            "cx": float(first_camera.cx),
            "cy": float(first_camera.cy),
            "w": first_camera.width,
            "h": first_camera.height,
            "aabb_scale": 16,
            "frames": []
        }

        # Group images by camera
        # Assumes naming convention: {cam_name}_frame_{number}.{ext}
        camera_images = defaultdict(list)

        for image in images.values():
            # Extract camera name and frame number
            parts = image.name.rsplit('_frame_', 1)
            if len(parts) == 2:
                cam_name = parts[0]
                camera_images[cam_name].append(image)
            else:
                # Fallback: use full name
                camera_images[image.name].append(image)

        # Sort images within each camera by name (frame order)
        for cam_name in camera_images:
            camera_images[cam_name].sort(key=lambda x: x.name)

        # Build frames with time annotations
        for cam_name, cam_images in camera_images.items():
            for frame_idx, image in enumerate(cam_images):
                camera = cameras[image.camera_id]
                transform_matrix = image.get_transform_matrix()

                # Normalize time to [0, 1]
                time_value = frame_idx / max(1, total_frames - 1) if total_frames > 1 else 0.0

                frame = {
                    "file_path": f"{image_path_prefix}{image.name}",
                    "transform_matrix": transform_matrix.tolist(),
                    "time": time_value,
                    "camera_id": cam_name
                }

                transforms["frames"].append(frame)

        # Sort frames by time, then by camera
        transforms["frames"].sort(key=lambda x: (x.get("time", 0), x.get("camera_id", "")))

        # Save transforms
        output_path = output_dir / "transforms.json"
        with open(output_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Saved 4DGS transforms to {output_path}")

        return transforms


def convert_colmap_to_transforms(
    model_path: Path,
    output_path: Path,
    image_path_prefix: str = "./images/"
) -> Path:
    """
    Convenience function to convert COLMAP model to transforms.json.

    Args:
        model_path: Path to COLMAP model directory
        output_path: Path to save transforms.json
        image_path_prefix: Prefix for image file paths

    Returns:
        Path to the saved transforms.json file
    """
    converter = ColmapConverter(model_path)
    converter.convert(output_path, image_path_prefix)
    return output_path


def main():
    """CLI entry point for COLMAP conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert COLMAP model to transforms.json"
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to COLMAP model directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for transforms.json"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="./images/",
        help="Image path prefix"
    )
    parser.add_argument(
        "--include-points",
        action="store_true",
        help="Include 3D points in output"
    )

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.model_path / "transforms.json"

    converter = ColmapConverter(args.model_path)
    transforms = converter.convert(
        output_path,
        args.prefix,
        args.include_points
    )

    print(f"\nConversion complete:")
    print(f"  Cameras: {len(transforms['frames'])} frames")
    print(f"  Resolution: {transforms['w']}x{transforms['h']}")
    print(f"  FOV: {np.degrees(transforms['camera_angle_x']):.1f}° x "
          f"{np.degrees(transforms['camera_angle_y']):.1f}°")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
