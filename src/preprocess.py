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
