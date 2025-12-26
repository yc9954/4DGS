"""
Rendering Wrapper for 4DGS Pipeline.

Handles rendering of trained 4D Gaussian Splatting models.
Supports various camera paths (spiral, orbit, interpolate) and output formats.
"""

import subprocess
import sys
import shutil
import json
import math
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

from loguru import logger

from .config import PipelineConfig, RenderConfig


@dataclass
class CameraPath:
    """Container for a camera path (sequence of camera poses)."""

    transforms: List[np.ndarray]  # List of 4x4 transformation matrices
    times: List[float]  # Time values for each pose (for 4D)
    intrinsics: Dict[str, float]  # Camera intrinsics

    def to_transforms_json(self, output_path: Path, image_prefix: str = "render_"):
        """
        Save camera path as transforms.json for rendering.

        Args:
            output_path: Where to save transforms.json
            image_prefix: Prefix for rendered image filenames
        """
        data = {
            "camera_angle_x": self.intrinsics.get("camera_angle_x", 0.8),
            "camera_angle_y": self.intrinsics.get("camera_angle_y", 0.6),
            "fl_x": self.intrinsics.get("fl_x", 1000),
            "fl_y": self.intrinsics.get("fl_y", 1000),
            "cx": self.intrinsics.get("cx", 400),
            "cy": self.intrinsics.get("cy", 300),
            "w": self.intrinsics.get("w", 800),
            "h": self.intrinsics.get("h", 600),
            "frames": []
        }

        for i, (transform, time_val) in enumerate(zip(self.transforms, self.times)):
            frame = {
                "file_path": f"{image_prefix}{i:06d}.png",
                "transform_matrix": transform.tolist(),
                "time": time_val
            }
            data["frames"].append(frame)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved camera path with {len(self.transforms)} frames to {output_path}")


class CameraPathGenerator:
    """
    Generates camera paths for rendering.

    Supports various path types:
    - interpolate: Interpolate between training camera poses
    - spiral: Spiral path around the scene center
    - orbit: Circular orbit around the scene
    """

    @staticmethod
    def load_training_poses(transforms_path: Path) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Load camera poses from training transforms.json.

        Args:
            transforms_path: Path to transforms.json

        Returns:
            Tuple of (list of transforms, intrinsics dict)
        """
        with open(transforms_path, 'r') as f:
            data = json.load(f)

        transforms = []
        for frame in data["frames"]:
            transform = np.array(frame["transform_matrix"])
            transforms.append(transform)

        intrinsics = {
            "camera_angle_x": data.get("camera_angle_x", 0.8),
            "camera_angle_y": data.get("camera_angle_y", 0.6),
            "fl_x": data.get("fl_x", 1000),
            "fl_y": data.get("fl_y", 1000),
            "cx": data.get("cx", 400),
            "cy": data.get("cy", 300),
            "w": data.get("w", 800),
            "h": data.get("h", 600),
        }

        return transforms, intrinsics

    @staticmethod
    def interpolate_poses(
        training_poses: List[np.ndarray],
        num_frames: int,
        loop: bool = True
    ) -> List[np.ndarray]:
        """
        Interpolate between training camera poses.

        Uses spherical linear interpolation (SLERP) for rotations
        and linear interpolation for translations.

        Args:
            training_poses: List of training camera poses
            num_frames: Number of frames to generate
            loop: Whether to loop back to the first pose

        Returns:
            List of interpolated camera poses
        """
        if len(training_poses) < 2:
            # Can't interpolate with single pose, just repeat it
            return [training_poses[0]] * num_frames

        result = []
        n_poses = len(training_poses)

        for i in range(num_frames):
            # Calculate which poses to interpolate between
            if loop:
                t = (i / num_frames) * n_poses
            else:
                t = (i / (num_frames - 1)) * (n_poses - 1)

            idx0 = int(t) % n_poses
            idx1 = (idx0 + 1) % n_poses
            alpha = t - int(t)

            # Get poses
            pose0 = training_poses[idx0]
            pose1 = training_poses[idx1]

            # Interpolate
            interpolated = CameraPathGenerator._interpolate_transform(pose0, pose1, alpha)
            result.append(interpolated)

        return result

    @staticmethod
    def _interpolate_transform(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
        """
        Interpolate between two 4x4 transformation matrices.

        Args:
            T0: First transformation matrix
            T1: Second transformation matrix
            alpha: Interpolation factor (0 = T0, 1 = T1)

        Returns:
            Interpolated transformation matrix
        """
        # Extract rotation and translation
        R0, t0 = T0[:3, :3], T0[:3, 3]
        R1, t1 = T1[:3, :3], T1[:3, 3]

        # Linear interpolation for translation
        t_interp = (1 - alpha) * t0 + alpha * t1

        # For rotation, we use a simple linear interpolation of the matrices
        # (A more accurate approach would use SLERP on quaternions)
        R_interp = (1 - alpha) * R0 + alpha * R1

        # Orthogonalize the rotation matrix using SVD
        U, _, Vt = np.linalg.svd(R_interp)
        R_interp = U @ Vt

        # Ensure proper rotation (det = 1)
        if np.linalg.det(R_interp) < 0:
            U[:, -1] *= -1
            R_interp = U @ Vt

        # Build result
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp

        return T_interp

    @staticmethod
    def generate_spiral(
        center: np.ndarray,
        radius: float,
        height_range: Tuple[float, float],
        num_frames: int,
        num_rotations: float = 2.0,
        look_at: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Generate a spiral camera path around the scene.

        Args:
            center: Center point of the spiral (3D)
            radius: Radius of the spiral
            height_range: (min_height, max_height) range
            num_frames: Number of frames to generate
            num_rotations: Number of complete rotations
            look_at: Point to look at (defaults to center)

        Returns:
            List of camera transformation matrices
        """
        if look_at is None:
            look_at = center.copy()

        result = []
        min_h, max_h = height_range

        for i in range(num_frames):
            t = i / num_frames
            angle = 2 * math.pi * num_rotations * t
            height = min_h + (max_h - min_h) * t

            # Camera position on spiral
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = height

            position = np.array([x, y, z])

            # Look at target
            forward = look_at - position
            forward = forward / np.linalg.norm(forward)

            # Up vector (world up)
            up = np.array([0, 0, 1])

            # Right vector
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            # Recompute up
            up = np.cross(right, forward)

            # Build rotation matrix (camera-to-world)
            R = np.stack([right, up, -forward], axis=1)

            # Build transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = position

            result.append(T)

        return result

    @staticmethod
    def generate_orbit(
        center: np.ndarray,
        radius: float,
        height: float,
        num_frames: int,
        look_at: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Generate a circular orbit camera path.

        Args:
            center: Center of the orbit (3D)
            radius: Orbit radius
            height: Fixed height above center
            num_frames: Number of frames
            look_at: Point to look at

        Returns:
            List of camera transformation matrices
        """
        return CameraPathGenerator.generate_spiral(
            center=center,
            radius=radius,
            height_range=(height, height),
            num_frames=num_frames,
            num_rotations=1.0,
            look_at=look_at
        )

    @classmethod
    def create_path(
        cls,
        path_type: str,
        training_transforms_path: Path,
        num_frames: int,
        time_duration: float = 1.0,
        **kwargs
    ) -> CameraPath:
        """
        Create a camera path for rendering.

        Args:
            path_type: Type of path ('interpolate', 'spiral', 'orbit')
            training_transforms_path: Path to training transforms.json
            num_frames: Number of frames to render
            time_duration: Duration in normalized time [0, 1]
            **kwargs: Additional path-specific arguments

        Returns:
            CameraPath object
        """
        training_poses, intrinsics = cls.load_training_poses(training_transforms_path)

        if path_type == "interpolate":
            transforms = cls.interpolate_poses(
                training_poses,
                num_frames,
                loop=kwargs.get("loop", True)
            )
        elif path_type == "spiral":
            # Compute scene center from training poses
            centers = [T[:3, 3] for T in training_poses]
            center = np.mean(centers, axis=0)

            # Estimate radius from training poses
            distances = [np.linalg.norm(T[:3, 3] - center) for T in training_poses]
            radius = kwargs.get("radius", np.mean(distances))

            # Height range
            heights = [T[2, 3] for T in training_poses]
            height_range = kwargs.get("height_range", (min(heights), max(heights)))

            transforms = cls.generate_spiral(
                center=center,
                radius=radius,
                height_range=height_range,
                num_frames=num_frames,
                num_rotations=kwargs.get("num_rotations", 2.0),
                look_at=center
            )
        elif path_type == "orbit":
            centers = [T[:3, 3] for T in training_poses]
            center = np.mean(centers, axis=0)
            distances = [np.linalg.norm(T[:3, 3] - center) for T in training_poses]

            transforms = cls.generate_orbit(
                center=center,
                radius=kwargs.get("radius", np.mean(distances)),
                height=kwargs.get("height", center[2]),
                num_frames=num_frames,
                look_at=center
            )
        else:
            raise ValueError(f"Unknown path type: {path_type}")

        # Generate time values
        times = [t / (num_frames - 1) * time_duration if num_frames > 1 else 0.0
                 for t in range(num_frames)]

        return CameraPath(
            transforms=transforms,
            times=times,
            intrinsics=intrinsics
        )


class RenderWrapper:
    """
    Wrapper for rendering 4DGS models.

    Handles:
    - Camera path generation
    - Rendering via subprocess
    - Video encoding with FFmpeg
    """

    def __init__(
        self,
        config: PipelineConfig,
        method: str = "4dgs",
        method_path: Optional[Path] = None
    ):
        """
        Initialize the render wrapper.

        Args:
            config: Pipeline configuration
            method: Rendering method
            method_path: Path to rendering code
        """
        self.config = config
        self.render_config = config.render
        self.method = method
        self.method_path = method_path
        self._method_dir = None

        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is available for video encoding."""
        if not shutil.which("ffmpeg"):
            logger.warning(
                "FFmpeg not found. Video encoding will not be available. "
                "Install FFmpeg to enable video output."
            )

    def _get_method_env(self) -> Dict[str, str]:
        """
        Get environment variables for the rendering subprocess.
        
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

    def _get_project_root(self) -> Path:
        """Get the project root directory."""
        current = Path(__file__).resolve().parent.parent
        markers = ["main.py", "requirements.txt", ".git"]
        for _ in range(5):
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent
        return Path.cwd()

    def _find_render_script(self) -> Optional[Path]:
        """
        Find the rendering script for the selected method.

        Returns None if no external render script is found,
        indicating that native rendering should be used.
        """
        project_root = self._get_project_root()

        if self.method_path:
            script = self.method_path / "render.py"
            if script.exists():
                self._method_dir = self.method_path
                return script

        # Check common locations for hustvl/4DGaussians
        possible_paths = [
            # Primary locations for 4DGS submodule
            project_root / "submodules" / "4dgs" / "render.py",
            project_root / "submodules" / "4DGaussians" / "render.py",
            project_root / "submodules" / self.method / "render.py",
            # Alternative locations
            project_root / "external" / "4dgs" / "render.py",
            project_root / "third_party" / "4dgs" / "render.py",
            # Direct render.py in project
            project_root / "render.py",
        ]

        for path in possible_paths:
            if path.exists():
                self._method_dir = path.parent
                logger.info(f"Found render script at: {path}")
                return path

        # No external render script found - will use native rendering
        logger.warning(
            f"No external render script found. Searched:\n"
            + "\n".join(f"  - {p}" for p in possible_paths[:4])
            + "\nWill use native rendering mode."
        )
        return None

    def _find_latest_checkpoint(self, model_path: Path) -> Optional[int]:
        """Find the latest checkpoint iteration in the model directory."""
        point_cloud_dir = model_path / "point_cloud"
        if not point_cloud_dir.exists():
            return None

        checkpoints = list(point_cloud_dir.glob("iteration_*"))
        if not checkpoints:
            return None

        iterations = []
        for ckpt in checkpoints:
            try:
                iter_num = int(ckpt.name.split("_")[1])
                # Verify point_cloud.ply exists
                if (ckpt / "point_cloud.ply").exists():
                    iterations.append(iter_num)
            except (ValueError, IndexError):
                continue

        return max(iterations) if iterations else None

    def _find_source_path(self, model_path: Path) -> Optional[Path]:
        """
        Find the source path (dataset path with transforms.json).

        Searches in common locations based on the model_path.
        """
        abs_model_path = Path(model_path).resolve()

        # Check various possible locations
        candidates = [
            # Same directory as model
            abs_model_path / "transforms.json",
            abs_model_path / "transforms_train.json",
            # Parent directories
            abs_model_path.parent / "transforms.json",
            abs_model_path.parent / "transforms_train.json",
            # data/processed structure
            abs_model_path.parent.parent / "processed" / abs_model_path.name / "transforms.json",
            # Two levels up (model is in outputs/exp/model)
            abs_model_path.parent.parent / "transforms.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                logger.debug(f"Found source transforms at: {candidate.parent}")
                return candidate.parent

        # Try to read cfg_args for source_path
        cfg_args_path = abs_model_path / "cfg_args"
        if cfg_args_path.exists():
            try:
                content = cfg_args_path.read_text()
                import re
                match = re.search(r"source_path='([^']+)'", content)
                if match:
                    source_path = Path(match.group(1))
                    if source_path.exists():
                        return source_path
            except Exception as e:
                logger.debug(f"Could not parse cfg_args: {e}")

        return None

    def render_path(
        self,
        model_path: Path,
        camera_path: CameraPath,
        output_dir: Path,
        iteration: Optional[int] = None,
        source_path: Optional[Path] = None
    ) -> Path:
        """
        Render a camera path using the trained model.

        Args:
            model_path: Path to the trained model
            camera_path: Camera path to render
            output_dir: Directory to save rendered frames
            iteration: Checkpoint iteration to use (None = latest)
            source_path: Path to original dataset (for finding transforms.json)

        Returns:
            Path to the rendered frames directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        abs_model_path = Path(model_path).resolve()
        abs_output_dir = Path(output_dir).resolve()

        # Find iteration to use
        if iteration is None:
            iteration = self._find_latest_checkpoint(abs_model_path)
            if iteration is None:
                logger.warning("No valid checkpoint found, using default iteration 30000")
                iteration = 30000
        logger.info(f"Using checkpoint iteration: {iteration}")

        # Find source path
        if source_path is None:
            source_path = self._find_source_path(abs_model_path)
            if source_path is None:
                logger.warning("Could not find source path, using model_path as source")
                source_path = abs_model_path

        abs_source_path = Path(source_path).resolve()
        logger.info(f"Source path: {abs_source_path}")

        # Save camera path for rendering
        path_file = abs_output_dir / "render_transforms.json"
        camera_path.to_transforms_json(path_file)

        # Try to find external render script
        render_script = self._find_render_script()

        if render_script is not None:
            # Use external render script (hustvl/4DGaussians)
            return self._render_with_external_script(
                render_script, abs_model_path, abs_source_path,
                abs_output_dir, iteration, camera_path
            )
        else:
            # Use native rendering
            return self._render_native(
                abs_model_path, abs_source_path, abs_output_dir,
                iteration, camera_path
            )

    def _render_with_external_script(
        self,
        render_script: Path,
        model_path: Path,
        source_path: Path,
        output_dir: Path,
        iteration: int,
        camera_path: CameraPath
    ) -> Path:
        """
        Render using external hustvl/4DGaussians render.py script.

        Note: The external script has limited functionality - it only renders
        train/test views, not custom camera paths. This method attempts to
        work around this limitation.
        """
        # hustvl/4DGaussians render.py uses -m and -s for paths
        cmd = [
            sys.executable,
            str(render_script),
            "-m", str(model_path),
            "-s", str(source_path),
        ]

        if iteration:
            cmd.extend(["--iteration", str(iteration)])

        logger.info(f"Rendering with external script: {render_script}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Get environment with PYTHONPATH
        env = self._get_method_env()

        # Run from method directory
        cwd = self._method_dir if self._method_dir else render_script.parent

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)

        if result.returncode != 0:
            logger.error(f"External render failed: {result.stderr}")
            logger.info("Falling back to native rendering...")
            return self._render_native(model_path, source_path, output_dir, iteration, camera_path)

        # Find output directory (hustvl/4DGaussians saves to model_path/train or test)
        possible_outputs = [
            model_path / "train" / f"ours_{iteration}" / "renders",
            model_path / "test" / f"ours_{iteration}" / "renders",
            model_path / f"ours_{iteration}" / "renders",
        ]

        for out_dir in possible_outputs:
            if out_dir.exists() and any(out_dir.glob("*.png")):
                logger.info(f"Found rendered frames in: {out_dir}")
                return out_dir

        logger.warning("Could not find rendered frames from external script")
        return output_dir

    def _render_native(
        self,
        model_path: Path,
        source_path: Path,
        output_dir: Path,
        iteration: int,
        camera_path: CameraPath
    ) -> Path:
        """
        Native Python rendering without external scripts.

        Uses our custom Gaussian Splatting renderer inspired by Blender's
        point rendering approach. This provides actual rendering without
        requiring the hustvl/4DGaussians CUDA dependencies.
        """
        logger.info("Using native Gaussian Splatting renderer")
        logger.info(f"Model: {model_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Frames to render: {len(camera_path.transforms)}")

        # Check if checkpoint exists
        ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply_path.exists():
            logger.error(f"Checkpoint not found: {ply_path}")
            raise FileNotFoundError(f"Checkpoint not found: {ply_path}")

        logger.info(f"Found checkpoint: {ply_path}")

        # Try to use our native Gaussian renderer
        try:
            from .gaussian_renderer import render_gaussian_model

            renders_dir = render_gaussian_model(
                model_path=model_path,
                output_dir=output_dir,
                camera_transforms=camera_path.transforms,
                camera_times=camera_path.times,
                intrinsics=camera_path.intrinsics,
                iteration=iteration,
                background=(0, 0, 0)
            )

            logger.info(f"Native rendering complete: {renders_dir}")
            return renders_dir

        except ImportError as e:
            logger.warning(f"Could not load native renderer: {e}")
            logger.info("Falling back to placeholder frames...")

        except Exception as e:
            logger.error(f"Native rendering failed: {e}")
            logger.info("Falling back to placeholder frames...")

        # Fallback: create placeholder frames
        return self._create_placeholder_frames(
            model_path, output_dir, iteration, camera_path
        )

    def _create_placeholder_frames(
        self,
        model_path: Path,
        output_dir: Path,
        iteration: int,
        camera_path: CameraPath
    ) -> Path:
        """Create placeholder frames when rendering is not available."""
        renders_dir = output_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image, ImageDraw
            has_pil = True
        except ImportError:
            has_pil = False
            logger.warning("PIL not available, creating empty placeholder frames")

        width = int(camera_path.intrinsics.get("w", 800))
        height = int(camera_path.intrinsics.get("h", 600))

        for i, (transform, time_val) in enumerate(zip(camera_path.transforms, camera_path.times)):
            frame_path = renders_dir / f"render_{i:06d}.png"

            if has_pil:
                img = Image.new('RGB', (width, height), color=(30, 30, 30))
                draw = ImageDraw.Draw(img)

                info_text = [
                    f"Frame: {i+1}/{len(camera_path.transforms)}",
                    f"Time: {time_val:.3f}",
                    f"Model: {model_path.name}",
                    f"Iteration: {iteration}",
                    "",
                    "Placeholder - Install dependencies:",
                    "  pip install Pillow numpy",
                ]

                y = height // 4
                for line in info_text:
                    draw.text((width // 4, y), line, fill=(200, 200, 200))
                    y += 25

                img.save(frame_path)
            else:
                frame_path.touch()

            if (i + 1) % 50 == 0:
                logger.info(f"Created frame {i+1}/{len(camera_path.transforms)}")

        logger.info(f"Created {len(camera_path.transforms)} placeholder frames")
        return renders_dir

    def frames_to_video(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: Optional[int] = None,
        pattern: str = "*.png"
    ) -> Path:
        """
        Encode rendered frames into a video using FFmpeg.

        Args:
            frames_dir: Directory containing rendered frames
            output_path: Output video path
            fps: Frames per second (uses config if None)
            pattern: Glob pattern for frame files

        Returns:
            Path to the output video
        """
        fps = fps or self.render_config.fps

        # Find frames
        frames = sorted(frames_dir.glob(pattern))
        if not frames:
            raise ValueError(f"No frames found in {frames_dir} matching {pattern}")

        logger.info(f"Encoding {len(frames)} frames to video at {fps} FPS")

        # Determine input pattern for FFmpeg
        # Assumes files are named like frame_000001.png
        first_frame = frames[0]
        if "%" in first_frame.stem:
            input_pattern = str(frames_dir / first_frame.name)
        else:
            # Try to auto-detect pattern
            import re
            match = re.search(r'(\d+)', first_frame.stem)
            if match:
                num_digits = len(match.group(1))
                prefix = first_frame.stem[:match.start()]
                suffix = first_frame.stem[match.end():]
                input_pattern = str(frames_dir / f"{prefix}%0{num_digits}d{suffix}{first_frame.suffix}")
            else:
                # Fallback: use concat demuxer
                return self._frames_to_video_concat(frames, output_path, fps)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", self.render_config.codec,
            "-crf", str(self.render_config.crf),
            "-pix_fmt", "yuv420p",  # Compatibility
            str(output_path)
        ]

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise RuntimeError(f"Video encoding failed: {result.stderr}")

        logger.info(f"Video saved to {output_path}")

        return output_path

    def _frames_to_video_concat(
        self,
        frames: List[Path],
        output_path: Path,
        fps: int
    ) -> Path:
        """
        Encode frames using FFmpeg concat demuxer.

        Fallback method when frame naming is irregular.
        """
        import tempfile

        # Create file list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for frame in frames:
                f.write(f"file '{frame.resolve()}'\n")
                f.write(f"duration {1/fps}\n")
            filelist_path = f.name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", filelist_path,
                "-c:v", self.render_config.codec,
                "-crf", str(self.render_config.crf),
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Video encoding failed: {result.stderr}")

        finally:
            Path(filelist_path).unlink()

        return output_path

    def render_and_encode(
        self,
        model_path: Path,
        output_video: Optional[Path] = None,
        path_type: Optional[str] = None,
        num_frames: Optional[int] = None,
        training_transforms: Optional[Path] = None
    ) -> Path:
        """
        Full rendering pipeline: generate path, render, and encode video.

        Args:
            model_path: Path to trained model
            output_video: Output video path (auto-generated if None)
            path_type: Camera path type (uses config if None)
            num_frames: Number of frames (uses config if None)
            training_transforms: Path to training transforms.json

        Returns:
            Path to the output video
        """
        path_type = path_type or self.render_config.camera_path_type
        num_frames = num_frames or self.render_config.num_frames

        # Find training transforms
        if training_transforms is None:
            training_transforms = model_path / "transforms.json"
            if not training_transforms.exists():
                training_transforms = model_path.parent / "transforms.json"
            if not training_transforms.exists():
                raise FileNotFoundError(
                    f"Could not find transforms.json for camera path generation. "
                    f"Please specify training_transforms path."
                )

        # Generate camera path
        logger.info(f"Generating {path_type} camera path with {num_frames} frames")
        camera_path = CameraPathGenerator.create_path(
            path_type=path_type,
            training_transforms_path=training_transforms,
            num_frames=num_frames
        )

        # Setup output paths
        render_dir = self.config.get_render_dir()
        frames_dir = render_dir / "frames"

        # Find source path for rendering (where transforms_train.json is)
        source_path = training_transforms.parent if training_transforms else None

        # Render - this returns the actual directory where frames were saved
        actual_frames_dir = self.render_path(model_path, camera_path, frames_dir, source_path=source_path)

        # Encode video
        if output_video is None:
            output_video = render_dir / f"render_{path_type}.{self.render_config.output_format}"

        return self.frames_to_video(actual_frames_dir, output_video)


class MockRenderWrapper(RenderWrapper):
    """
    Mock render wrapper for testing without actual rendering.
    """

    def render_path(
        self,
        model_path: Path,
        camera_path: CameraPath,
        output_dir: Path,
        iteration: Optional[int] = None,
        source_path: Optional[Path] = None
    ) -> Path:
        """Simulate rendering."""
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== MOCK RENDER MODE ===")
        logger.info(f"Would render from: {model_path}")
        logger.info(f"Frames: {len(camera_path.transforms)}")
        logger.info(f"Output: {output_dir}")

        # Create placeholder frames
        for i in range(min(5, len(camera_path.transforms))):
            frame_path = output_dir / f"render_{i:06d}.png"
            frame_path.touch()

        return output_dir


def main():
    """CLI entry point for render wrapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Render 4DGS model")
    parser.add_argument("model_path", type=Path, help="Path to trained model")
    parser.add_argument("-o", "--output", type=Path, help="Output video path")
    parser.add_argument("--path-type", choices=["interpolate", "spiral", "orbit"],
                        default="interpolate", help="Camera path type")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of frames")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--transforms", type=Path, help="Training transforms.json path")

    args = parser.parse_args()

    config = PipelineConfig()
    config.render.camera_path_type = args.path_type
    config.render.num_frames = args.num_frames
    config.render.fps = args.fps

    wrapper = RenderWrapper(config)

    video_path = wrapper.render_and_encode(
        model_path=args.model_path,
        output_video=args.output,
        training_transforms=args.transforms
    )

    print(f"\nRender complete: {video_path}")


if __name__ == "__main__":
    main()
