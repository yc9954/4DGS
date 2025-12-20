"""
Video Preprocessing Module for 4DGS Pipeline.

Handles extraction of frames from multi-view video files using FFmpeg.
Supports synchronized frame extraction, quality control, and various output formats.
"""

import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import json
import re

from loguru import logger
from tqdm import tqdm

from .config import PipelineConfig, FrameExtractionConfig


@dataclass
class VideoInfo:
    """Container for video metadata."""

    path: Path
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    codec: str


class FrameExtractor:
    """
    Extracts frames from multi-view video files.

    This class provides functionality to:
    - Probe video files for metadata
    - Extract frames at specified FPS
    - Ensure synchronization across multiple cameras
    - Handle various video formats and codecs
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the FrameExtractor.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.frame_config = config.frame_extraction
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is installed and accessible."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg:\n"
                "  Ubuntu: sudo apt-get install ffmpeg\n"
                "  macOS: brew install ffmpeg"
            )
        if not shutil.which("ffprobe"):
            raise RuntimeError(
                "FFprobe not found. It should be installed with FFmpeg."
            )
        logger.debug("FFmpeg and FFprobe found")

    def probe_video(self, video_path: Path) -> VideoInfo:
        """
        Extract metadata from a video file using FFprobe.

        Args:
            video_path: Path to the video file

        Returns:
            VideoInfo object containing video metadata
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed for {video_path}: {result.stderr}")

        probe_data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}")

        # Parse frame rate (can be "30/1" or "29.97" format)
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)

        # Get duration
        duration = float(video_stream.get("duration", 0))
        if duration == 0:
            duration = float(probe_data.get("format", {}).get("duration", 0))

        # Calculate total frames
        nb_frames = video_stream.get("nb_frames")
        if nb_frames:
            total_frames = int(nb_frames)
        else:
            total_frames = int(duration * fps)

        return VideoInfo(
            path=video_path,
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            duration=duration,
            total_frames=total_frames,
            codec=video_stream.get("codec_name", "unknown")
        )

    def discover_videos(self, input_dir: Optional[Path] = None) -> List[Path]:
        """
        Find all video files in the input directory.

        Args:
            input_dir: Directory to search (uses config.input_dir if None)

        Returns:
            List of paths to video files, sorted alphabetically
        """
        if input_dir is None:
            input_dir = self.config.input_dir

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        videos = []

        for ext in video_extensions:
            videos.extend(input_dir.glob(f"*{ext}"))
            videos.extend(input_dir.glob(f"*{ext.upper()}"))

        videos = sorted(set(videos))
        logger.info(f"Found {len(videos)} video files in {input_dir}")

        return videos

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        fps: Optional[float] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[int, Path]:
        """
        Extract frames from a single video file.

        Args:
            video_path: Path to the input video
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract (uses config if None)
            start_time: Start time in seconds (uses config if None)
            duration: Duration to extract (uses config if None)

        Returns:
            Tuple of (number of frames extracted, output directory path)
        """
        # Use config values if not specified
        fps = fps or self.frame_config.fps
        start_time = start_time if start_time is not None else self.frame_config.start_time
        duration = duration or self.frame_config.duration

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]  # -y to overwrite existing files

        # Input options
        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        if duration:
            cmd.extend(["-t", str(duration)])

        cmd.extend(["-i", str(video_path)])

        # Video filters
        filters = []

        # Frame rate filter
        filters.append(f"fps={fps}")

        # Resize if specified
        if self.frame_config.resize_width or self.frame_config.resize_height:
            w = self.frame_config.resize_width or -1
            h = self.frame_config.resize_height or -1
            filters.append(f"scale={w}:{h}")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Output format and quality
        if self.frame_config.format.lower() == "png":
            cmd.extend(["-compression_level", "3"])  # Fast PNG compression
            output_pattern = output_dir / "frame_%06d.png"
        else:
            cmd.extend(["-q:v", str(max(1, (100 - self.frame_config.quality) // 4))])
            output_pattern = output_dir / "frame_%06d.jpg"

        cmd.append(str(output_pattern))

        logger.info(f"Extracting frames from {video_path.name} at {fps} FPS")
        logger.debug(f"Command: {' '.join(cmd)}")

        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for {video_path}:\n{result.stderr}"
            )

        # Count extracted frames
        pattern = f"*.{self.frame_config.format}"
        frame_count = len(list(output_dir.glob(pattern)))

        logger.info(f"Extracted {frame_count} frames to {output_dir}")

        return frame_count, output_dir

    def extract_all_videos(
        self,
        videos: Optional[List[Path]] = None,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Extract frames from all video files.

        For 4DGS, we need frames organized by camera (view).
        Structure: processed/{experiment}/input/{cam_name}/images/

        Args:
            videos: List of video paths (discovers if None)
            experiment_name: Name of the experiment (uses config if None)

        Returns:
            Dictionary mapping camera names to extraction info
        """
        if videos is None:
            videos = self.discover_videos()

        if not videos:
            raise ValueError(f"No videos found in {self.config.input_dir}")

        experiment_name = experiment_name or self.config.experiment_name
        base_output_dir = self.config.processed_dir / experiment_name / "input"

        results = {}

        # First, probe all videos to find the minimum duration
        # (for synchronization, we extract the same number of frames from each)
        logger.info("Probing videos for synchronization...")
        video_infos = {}
        min_duration = float("inf")

        for video in videos:
            info = self.probe_video(video)
            video_infos[video] = info
            min_duration = min(min_duration, info.duration)
            logger.debug(
                f"  {video.name}: {info.width}x{info.height}, "
                f"{info.fps:.2f} FPS, {info.duration:.2f}s"
            )

        # Apply duration limit if specified in config
        if self.frame_config.duration:
            min_duration = min(min_duration, self.frame_config.duration)

        logger.info(f"Synchronized duration: {min_duration:.2f}s")

        # Extract frames from each video
        for video in tqdm(videos, desc="Extracting frames"):
            # Camera name is derived from video filename (without extension)
            cam_name = video.stem
            output_dir = base_output_dir / cam_name / "images"

            # Skip if already processed
            if self.config.skip_existing and output_dir.exists():
                existing_frames = len(list(output_dir.glob(f"*.{self.frame_config.format}")))
                if existing_frames > 0:
                    logger.info(f"Skipping {cam_name} - already has {existing_frames} frames")
                    results[cam_name] = {
                        "video_path": video,
                        "output_dir": output_dir,
                        "frame_count": existing_frames,
                        "video_info": video_infos[video],
                        "skipped": True
                    }
                    continue

            # Extract with synchronized duration
            frame_count, out_dir = self.extract_frames(
                video,
                output_dir,
                duration=min_duration
            )

            results[cam_name] = {
                "video_path": video,
                "output_dir": out_dir,
                "frame_count": frame_count,
                "video_info": video_infos[video],
                "skipped": False
            }

        # Verify synchronization
        frame_counts = [r["frame_count"] for r in results.values()]
        if len(set(frame_counts)) > 1:
            logger.warning(
                f"Frame count mismatch across cameras: {dict(zip(results.keys(), frame_counts))}"
            )
        else:
            logger.info(f"All cameras synchronized with {frame_counts[0]} frames each")

        # Save extraction metadata
        self._save_extraction_metadata(results, base_output_dir.parent)

        return results

    def _save_extraction_metadata(self, results: Dict, output_dir: Path):
        """Save metadata about the extraction process."""
        metadata = {
            "config": {
                "fps": self.frame_config.fps,
                "format": self.frame_config.format,
                "quality": self.frame_config.quality,
            },
            "cameras": {}
        }

        for cam_name, info in results.items():
            metadata["cameras"][cam_name] = {
                "source_video": str(info["video_path"]),
                "frame_count": info["frame_count"],
                "original_fps": info["video_info"].fps,
                "original_resolution": f"{info['video_info'].width}x{info['video_info'].height}",
                "original_duration": info["video_info"].duration,
            }

        metadata_path = output_dir / "extraction_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved extraction metadata to {metadata_path}")

    def create_frame_symlinks(
        self,
        results: Dict[str, Dict],
        experiment_name: Optional[str] = None
    ) -> Path:
        """
        Create a unified frame structure for COLMAP processing.

        For static camera pose estimation, we need all frames from all cameras
        in a single directory. This creates symlinks to avoid duplication.

        Structure:
            colmap/images/
                cam1_frame_000001.png
                cam1_frame_000002.png
                ...
                cam2_frame_000001.png

        Args:
            results: Output from extract_all_videos()
            experiment_name: Experiment name (uses config if None)

        Returns:
            Path to the unified images directory
        """
        experiment_name = experiment_name or self.config.experiment_name
        colmap_images_dir = self.config.processed_dir / experiment_name / "colmap" / "images"
        colmap_images_dir.mkdir(parents=True, exist_ok=True)

        for cam_name, info in results.items():
            frame_dir = info["output_dir"]
            for frame_path in sorted(frame_dir.glob("*")):
                if frame_path.is_file():
                    # Prefix with camera name for uniqueness
                    link_name = f"{cam_name}_{frame_path.name}"
                    link_path = colmap_images_dir / link_name

                    # Create symlink (remove if exists)
                    if link_path.exists():
                        link_path.unlink()
                    link_path.symlink_to(frame_path.resolve())

        total_links = len(list(colmap_images_dir.glob("*")))
        logger.info(f"Created {total_links} symlinks in {colmap_images_dir}")

        return colmap_images_dir


class DatasetAdapter:
    """
    Adapter for converting public dataset formats to pipeline format.

    Supports:
    - N3DV (Neural 3D Video): Multi-camera video sequences
    - D-NeRF: Synthetic dynamic scenes
    - LLFF: Forward-facing scenes with poses_bounds.npy
    - Custom: Direct transforms.json format
    """

    # Known dataset signatures
    DATASET_SIGNATURES = {
        "n3dv": ["poses_bounds.npy", "cam00", "cam01"],
        "dnerf": ["transforms_train.json", "transforms_test.json"],
        "llff": ["poses_bounds.npy", "images"],
        "nerf": ["transforms.json"],
    }

    def __init__(self):
        """Initialize the dataset adapter."""
        pass

    def detect_dataset_type(self, input_dir: Path) -> Optional[str]:
        """
        Detect the type of dataset based on directory structure.

        Args:
            input_dir: Path to the input dataset directory

        Returns:
            Dataset type string, or None if unknown
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            return None

        files_and_dirs = [f.name for f in input_dir.iterdir()]

        # Check for D-NeRF format first (most specific)
        if "transforms_train.json" in files_and_dirs:
            return "dnerf"

        # Check for existing transforms.json (already in our format)
        if "transforms.json" in files_and_dirs:
            return None  # Already in correct format

        # Check for N3DV multi-camera format
        cam_dirs = [d for d in files_and_dirs if d.startswith("cam")]
        if len(cam_dirs) >= 2:
            return "n3dv"

        # Check for LLFF format
        if "poses_bounds.npy" in files_and_dirs:
            return "llff"

        return None

    def adapt_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        dataset_type: str
    ) -> Path:
        """
        Adapt a dataset to pipeline format.

        Args:
            input_dir: Input dataset directory
            output_dir: Output directory for adapted data
            dataset_type: Type of dataset ('n3dv', 'dnerf', 'llff')

        Returns:
            Path to the adapted dataset
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if dataset_type == "n3dv":
            return self.adapt_n3dv_dataset(input_dir, output_dir)
        elif dataset_type == "dnerf":
            return self.adapt_dnerf_dataset(input_dir, output_dir)
        elif dataset_type == "llff":
            return self.convert_llff_to_transforms(input_dir, output_dir)
        else:
            logger.warning(f"Unknown dataset type: {dataset_type}, copying as-is")
            return self._copy_dataset(input_dir, output_dir)

    def adapt_n3dv_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Path:
        """
        Adapt Neural 3D Video dataset format to pipeline format.

        N3DV structure:
            dataset/
                cam00/
                    images/
                        000000.png
                        000001.png
                cam01/
                    images/
                poses_bounds.npy (optional, LLFF format)

        Args:
            input_dir: Input N3DV dataset directory
            output_dir: Output directory

        Returns:
            Path to adapted dataset
        """
        import shutil
        import numpy as np

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all camera directories
        cam_dirs = sorted([
            d for d in input_dir.iterdir()
            if d.is_dir() and d.name.startswith("cam")
        ])

        if not cam_dirs:
            logger.error("No camera directories found in N3DV dataset")
            return input_dir

        logger.info(f"Found {len(cam_dirs)} cameras in N3DV dataset")

        # Check for poses_bounds.npy (LLFF format poses)
        poses_file = input_dir / "poses_bounds.npy"
        has_poses = poses_file.exists()

        # Create images directory and gather all frames
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        frames = []
        cam_frame_counts = {}

        for cam_dir in cam_dirs:
            cam_name = cam_dir.name
            cam_images_dir = cam_dir / "images"

            if not cam_images_dir.exists():
                cam_images_dir = cam_dir  # Images might be directly in cam dir

            image_files = sorted(list(cam_images_dir.glob("*.png")) +
                                list(cam_images_dir.glob("*.jpg")))

            cam_frame_counts[cam_name] = len(image_files)

            for i, img_path in enumerate(image_files):
                # Create unique filename
                new_name = f"{cam_name}_{img_path.name}"
                new_path = images_dir / new_name

                # Create symlink or copy
                if new_path.exists():
                    new_path.unlink()
                new_path.symlink_to(img_path.resolve())

                # Calculate time value (normalized 0-1)
                total_frames = len(image_files)
                time_val = i / max(1, total_frames - 1) if total_frames > 1 else 0.0

                frames.append({
                    "file_path": f"images/{new_name}",
                    "time": time_val,
                    "camera_id": int(cam_name.replace("cam", "")),
                    "frame_id": i
                })

        # Load or generate camera poses
        if has_poses:
            transforms = self._load_llff_poses(poses_file, frames)
        else:
            logger.warning("No poses_bounds.npy found, generating placeholder poses")
            transforms = self._generate_placeholder_poses(frames, len(cam_dirs))

        # Save transforms.json
        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Adapted N3DV dataset: {len(frames)} frames from {len(cam_dirs)} cameras")
        logger.info(f"Saved transforms to: {transforms_path}")

        return output_dir

    def adapt_dnerf_dataset(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Path:
        """
        Adapt D-NeRF dataset format to pipeline format.

        D-NeRF structure:
            dataset/
                transforms_train.json
                transforms_test.json
                train/
                    r_0.png
                test/
                    r_0.png

        Args:
            input_dir: Input D-NeRF dataset directory
            output_dir: Output directory

        Returns:
            Path to adapted dataset
        """
        import shutil

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load D-NeRF transforms
        train_transforms_path = input_dir / "transforms_train.json"
        test_transforms_path = input_dir / "transforms_test.json"

        if not train_transforms_path.exists():
            logger.error("transforms_train.json not found")
            return input_dir

        with open(train_transforms_path, 'r') as f:
            train_transforms = json.load(f)

        # Copy images directory structure
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Process frames
        frames = []
        for frame in train_transforms.get("frames", []):
            file_path = frame.get("file_path", "")

            # Handle relative paths
            if file_path.startswith("./"):
                file_path = file_path[2:]

            src_path = input_dir / file_path
            if not src_path.suffix:
                # Try common extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (input_dir / f"{file_path}{ext}").exists():
                        src_path = input_dir / f"{file_path}{ext}"
                        file_path = f"{file_path}{ext}"
                        break

            if src_path.exists():
                # Copy or link image
                dst_name = src_path.name
                dst_path = images_dir / dst_name

                if not dst_path.exists():
                    shutil.copy(src_path, dst_path)

                # Update frame data
                new_frame = {
                    "file_path": f"images/{dst_name}",
                    "transform_matrix": frame.get("transform_matrix"),
                    "time": frame.get("time", 0.0),
                }
                frames.append(new_frame)

        # Build output transforms
        transforms = {
            "camera_angle_x": train_transforms.get("camera_angle_x", 0.8),
            "camera_angle_y": train_transforms.get("camera_angle_y"),
            "fl_x": train_transforms.get("fl_x"),
            "fl_y": train_transforms.get("fl_y"),
            "cx": train_transforms.get("cx"),
            "cy": train_transforms.get("cy"),
            "w": train_transforms.get("w", 800),
            "h": train_transforms.get("h", 800),
            "frames": frames
        }

        # Remove None values
        transforms = {k: v for k, v in transforms.items() if v is not None}

        # Save transforms.json
        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Adapted D-NeRF dataset: {len(frames)} frames")
        logger.info(f"Saved transforms to: {transforms_path}")

        return output_dir

    def convert_llff_to_transforms(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> Path:
        """
        Convert LLFF format (poses_bounds.npy) to transforms.json.

        LLFF structure:
            dataset/
                images/
                    image001.png
                poses_bounds.npy (N x 17 array)

        Args:
            input_dir: Input LLFF dataset directory
            output_dir: Output directory

        Returns:
            Path to converted dataset
        """
        import shutil
        import numpy as np

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        poses_file = input_dir / "poses_bounds.npy"
        if not poses_file.exists():
            logger.error("poses_bounds.npy not found")
            return input_dir

        # Load poses
        poses_arr = np.load(poses_file)
        logger.info(f"Loaded poses_bounds.npy: shape {poses_arr.shape}")

        # LLFF format: N x 17 (3x4 pose + 2 bounds + height/width/focal)
        # Reshape to (N, 17)
        if len(poses_arr.shape) == 2:
            num_images = poses_arr.shape[0]
        else:
            poses_arr = poses_arr.reshape(-1, 17)
            num_images = poses_arr.shape[0]

        # Find images
        images_dir = input_dir / "images"
        if not images_dir.exists():
            images_dir = input_dir

        image_files = sorted(
            list(images_dir.glob("*.png")) +
            list(images_dir.glob("*.jpg")) +
            list(images_dir.glob("*.jpeg"))
        )

        if len(image_files) != num_images:
            logger.warning(
                f"Image count ({len(image_files)}) != pose count ({num_images})"
            )

        # Create output images directory
        out_images_dir = output_dir / "images"
        out_images_dir.mkdir(exist_ok=True)

        frames = []

        for i in range(min(num_images, len(image_files))):
            pose_data = poses_arr[i]

            # Extract pose (3x4 -> 4x4)
            pose = pose_data[:12].reshape(3, 4)
            transform = np.eye(4)
            transform[:3, :] = pose

            # LLFF uses a different coordinate convention
            # Convert from LLFF to standard NeRF coordinate system
            transform = self._llff_to_nerf_pose(transform)

            # Get bounds
            near, far = pose_data[12], pose_data[13]

            # Get intrinsics (height, width, focal)
            h, w, focal = pose_data[14], pose_data[15], pose_data[16]

            # Copy image
            src_img = image_files[i]
            dst_img = out_images_dir / src_img.name
            if not dst_img.exists():
                shutil.copy(src_img, dst_img)

            frames.append({
                "file_path": f"images/{src_img.name}",
                "transform_matrix": transform.tolist(),
                "near": float(near),
                "far": float(far),
                "time": i / max(1, num_images - 1)  # Normalized time
            })

        # Compute camera parameters from first image
        h, w, focal = poses_arr[0, 14], poses_arr[0, 15], poses_arr[0, 16]
        fov_x = 2 * np.arctan(w / (2 * focal))
        fov_y = 2 * np.arctan(h / (2 * focal))

        transforms = {
            "camera_angle_x": float(fov_x),
            "camera_angle_y": float(fov_y),
            "fl_x": float(focal),
            "fl_y": float(focal),
            "cx": float(w / 2),
            "cy": float(h / 2),
            "w": int(w),
            "h": int(h),
            "frames": frames
        }

        # Save transforms
        transforms_path = output_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)

        logger.info(f"Converted LLFF dataset: {len(frames)} frames")
        return output_dir

    def _llff_to_nerf_pose(self, pose: 'np.ndarray') -> 'np.ndarray':
        """Convert LLFF pose to NeRF coordinate convention."""
        import numpy as np

        # LLFF uses (x: right, y: up, z: backward)
        # NeRF uses (x: right, y: up, z: forward)
        # So we need to negate the z axis

        # Also, LLFF stores camera-to-world, which is what we want
        convert = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        return pose @ convert

    def _load_llff_poses(
        self,
        poses_file: Path,
        frames: list
    ) -> dict:
        """Load LLFF poses and apply to frames."""
        import numpy as np

        poses_arr = np.load(poses_file)
        num_poses = poses_arr.shape[0]

        # Get camera intrinsics from first pose
        h, w, focal = poses_arr[0, 14], poses_arr[0, 15], poses_arr[0, 16]
        fov_x = 2 * np.arctan(w / (2 * focal))

        # Apply poses to frames (if possible)
        for i, frame in enumerate(frames):
            if i < num_poses:
                pose = poses_arr[i, :12].reshape(3, 4)
                transform = np.eye(4)
                transform[:3, :] = pose
                transform = self._llff_to_nerf_pose(transform)
                frame["transform_matrix"] = transform.tolist()

        transforms = {
            "camera_angle_x": float(fov_x),
            "fl_x": float(focal),
            "fl_y": float(focal),
            "cx": float(w / 2),
            "cy": float(h / 2),
            "w": int(w),
            "h": int(h),
            "frames": frames
        }

        return transforms

    def _generate_placeholder_poses(
        self,
        frames: list,
        num_cameras: int
    ) -> dict:
        """Generate placeholder poses when no pose data is available."""
        import numpy as np

        # Create camera ring around origin
        for i, frame in enumerate(frames):
            cam_id = frame.get("camera_id", i % num_cameras)
            angle = (cam_id / num_cameras) * 2 * np.pi
            radius = 4.0

            # Camera position
            cam_x = radius * np.cos(angle)
            cam_y = 0.5
            cam_z = radius * np.sin(angle)

            # Look at origin
            forward = np.array([0, 0, 0]) - np.array([cam_x, cam_y, cam_z])
            forward = forward / np.linalg.norm(forward)

            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            transform = np.eye(4)
            transform[:3, 0] = right
            transform[:3, 1] = up
            transform[:3, 2] = -forward
            transform[:3, 3] = [cam_x, cam_y, cam_z]

            frame["transform_matrix"] = transform.tolist()

        transforms = {
            "camera_angle_x": 0.8,  # ~45 degrees
            "frames": frames
        }

        return transforms

    def _copy_dataset(self, input_dir: Path, output_dir: Path) -> Path:
        """Copy dataset as-is when no adaptation is needed."""
        import shutil

        if input_dir == output_dir:
            return output_dir

        if output_dir.exists():
            shutil.rmtree(output_dir)

        shutil.copytree(input_dir, output_dir)
        return output_dir


def main():
    """CLI entry point for testing frame extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from multi-view videos")
    parser.add_argument("input_dir", type=Path, help="Directory containing video files")
    parser.add_argument("output_dir", type=Path, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=30.0, help="Extraction FPS")
    parser.add_argument("--format", choices=["png", "jpg"], default="png", help="Output format")
    parser.add_argument("--quality", type=int, default=95, help="Output quality (1-100)")

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        input_dir=args.input_dir,
        processed_dir=args.output_dir,
    )
    config.frame_extraction.fps = args.fps
    config.frame_extraction.format = args.format
    config.frame_extraction.quality = args.quality

    # Run extraction
    extractor = FrameExtractor(config)
    results = extractor.extract_all_videos()

    print(f"\nExtraction complete. Results:")
    for cam, info in results.items():
        print(f"  {cam}: {info['frame_count']} frames")


if __name__ == "__main__":
    main()
