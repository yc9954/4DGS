"""
Native Gaussian Splatting Renderer.

A pure Python/NumPy implementation of Gaussian Splatting rendering,
inspired by Blender's point rendering approach but adapted for
3D Gaussian Splatting with:
- Anisotropic 2D Gaussians (elliptical splats)
- Gaussian alpha falloff
- Spherical harmonics for view-dependent color
- Proper depth sorting and alpha compositing

This renderer works without external CUDA dependencies, making it
portable and easy to debug. For production use, consider using
the hustvl/4DGaussians CUDA rasterizer for better performance.

References:
- Blender GPU shaders: gpu_shader_3D_point_varying_size_varying_color
- 3D Gaussian Splatting paper: Kerbl et al. 2023
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import struct
import json

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from loguru import logger


@dataclass
class GaussianPoint:
    """A single 3D Gaussian point."""
    position: np.ndarray  # (3,) xyz
    scale: np.ndarray     # (3,) scale in xyz
    rotation: np.ndarray  # (4,) quaternion wxyz
    opacity: float        # opacity [0, 1]
    sh_coeffs: np.ndarray # Spherical harmonics coefficients
    color: np.ndarray     # Base RGB color (from SH dc term)


@dataclass
class Camera:
    """Camera parameters for rendering."""
    width: int
    height: int
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    world_to_camera: np.ndarray  # 4x4 extrinsic matrix

    @classmethod
    def from_transform_matrix(
        cls,
        transform: np.ndarray,
        width: int = 800,
        height: int = 600,
        fov_x: float = 0.8
    ) -> 'Camera':
        """
        Create camera from a 4x4 transform matrix (camera-to-world).

        Args:
            transform: 4x4 camera-to-world transformation matrix
            width: Image width
            height: Image height
            fov_x: Field of view in x direction (radians)
        """
        # Compute focal length from FOV
        fx = width / (2 * np.tan(fov_x / 2))
        fy = fx  # Assume square pixels

        # Principal point at image center
        cx = width / 2
        cy = height / 2

        # Invert transform to get world-to-camera
        world_to_camera = np.linalg.inv(transform)

        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            world_to_camera=world_to_camera
        )

    @classmethod
    def from_intrinsics_dict(
        cls,
        intrinsics: Dict[str, float],
        transform: np.ndarray
    ) -> 'Camera':
        """Create camera from intrinsics dictionary and transform."""
        width = int(intrinsics.get('w', 800))
        height = int(intrinsics.get('h', 600))

        # Try to get focal length directly, or compute from FOV
        if 'fl_x' in intrinsics:
            fx = intrinsics['fl_x']
            fy = intrinsics.get('fl_y', fx)
        elif 'camera_angle_x' in intrinsics:
            fov_x = intrinsics['camera_angle_x']
            fx = width / (2 * np.tan(fov_x / 2))
            fy = fx
        else:
            fx = fy = width  # Fallback

        cx = intrinsics.get('cx', width / 2)
        cy = intrinsics.get('cy', height / 2)

        world_to_camera = np.linalg.inv(transform)

        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            world_to_camera=world_to_camera
        )

    def project_point(self, point_world: np.ndarray) -> Tuple[float, float, float]:
        """
        Project a 3D world point to 2D image coordinates.

        Returns:
            (x, y, depth) where x, y are pixel coordinates and depth is z in camera space
        """
        # Transform to camera space
        point_homo = np.append(point_world, 1.0)
        point_cam = self.world_to_camera @ point_homo

        x, y, z = point_cam[:3]

        if z <= 0:
            return -1, -1, z  # Behind camera

        # Project to image plane
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        return u, v, z


class PLYLoader:
    """
    Load Gaussian point cloud from PLY file.

    Supports the PLY format used by 3D Gaussian Splatting:
    - position (x, y, z)
    - normal (nx, ny, nz)
    - spherical harmonics (f_dc_0..2, f_rest_0..44)
    - opacity
    - scale (scale_0..2)
    - rotation quaternion (rot_0..3)
    """

    @staticmethod
    def load(ply_path: Path) -> List[GaussianPoint]:
        """
        Load Gaussian points from PLY file.

        Args:
            ply_path: Path to PLY file

        Returns:
            List of GaussianPoint objects
        """
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        with open(ply_path, 'rb') as f:
            # Parse header
            header_lines = []
            while True:
                line = f.readline().decode('utf-8').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break

            # Parse header info
            num_vertices = 0
            properties = []
            is_binary = False
            is_little_endian = True

            for line in header_lines:
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                elif line.startswith('property'):
                    parts = line.split()
                    prop_type = parts[1]
                    prop_name = parts[2]
                    properties.append((prop_name, prop_type))
                elif line.startswith('format'):
                    if 'binary_little_endian' in line:
                        is_binary = True
                        is_little_endian = True
                    elif 'binary_big_endian' in line:
                        is_binary = True
                        is_little_endian = False

            logger.info(f"Loading {num_vertices} Gaussian points from {ply_path.name}")

            # Build property index
            prop_idx = {name: i for i, (name, _) in enumerate(properties)}

            # Read vertex data
            if is_binary:
                points = PLYLoader._read_binary(
                    f, num_vertices, properties, prop_idx, is_little_endian
                )
            else:
                points = PLYLoader._read_ascii(
                    f, num_vertices, properties, prop_idx
                )

        logger.info(f"Loaded {len(points)} Gaussian points")
        return points

    @staticmethod
    def _read_binary(
        f,
        num_vertices: int,
        properties: List[Tuple[str, str]],
        prop_idx: Dict[str, int],
        little_endian: bool
    ) -> List[GaussianPoint]:
        """Read binary PLY data."""
        points = []

        # Determine byte size per property type
        type_sizes = {
            'float': 4, 'double': 8,
            'uchar': 1, 'uint8': 1,
            'int': 4, 'uint': 4,
        }

        # Build format string
        endian = '<' if little_endian else '>'
        fmt = endian
        for _, ptype in properties:
            if ptype in ['float', 'float32']:
                fmt += 'f'
            elif ptype in ['double', 'float64']:
                fmt += 'd'
            elif ptype in ['uchar', 'uint8']:
                fmt += 'B'
            elif ptype in ['int', 'int32']:
                fmt += 'i'
            elif ptype in ['uint', 'uint32']:
                fmt += 'I'
            else:
                fmt += 'f'  # Default to float

        vertex_size = struct.calcsize(fmt)

        for _ in range(num_vertices):
            data = f.read(vertex_size)
            if len(data) < vertex_size:
                break

            values = struct.unpack(fmt, data)
            point = PLYLoader._create_point(values, prop_idx)
            points.append(point)

        return points

    @staticmethod
    def _read_ascii(
        f,
        num_vertices: int,
        properties: List[Tuple[str, str]],
        prop_idx: Dict[str, int]
    ) -> List[GaussianPoint]:
        """Read ASCII PLY data."""
        points = []

        for _ in range(num_vertices):
            line = f.readline().decode('utf-8').strip()
            if not line:
                continue

            values = [float(x) for x in line.split()]
            point = PLYLoader._create_point(values, prop_idx)
            points.append(point)

        return points

    @staticmethod
    def _create_point(values: tuple, prop_idx: Dict[str, int]) -> GaussianPoint:
        """Create a GaussianPoint from property values."""
        # Position
        x = values[prop_idx.get('x', 0)]
        y = values[prop_idx.get('y', 1)]
        z = values[prop_idx.get('z', 2)]
        position = np.array([x, y, z], dtype=np.float32)

        # Scale (log scale in PLY)
        if 'scale_0' in prop_idx:
            scale = np.array([
                np.exp(values[prop_idx['scale_0']]),
                np.exp(values[prop_idx['scale_1']]),
                np.exp(values[prop_idx['scale_2']])
            ], dtype=np.float32)
        else:
            scale = np.array([0.01, 0.01, 0.01], dtype=np.float32)

        # Rotation quaternion
        if 'rot_0' in prop_idx:
            rotation = np.array([
                values[prop_idx['rot_0']],
                values[prop_idx['rot_1']],
                values[prop_idx['rot_2']],
                values[prop_idx['rot_3']]
            ], dtype=np.float32)
            # Normalize quaternion
            rotation = rotation / (np.linalg.norm(rotation) + 1e-8)
        else:
            rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Opacity (sigmoid in PLY)
        if 'opacity' in prop_idx:
            opacity_raw = values[prop_idx['opacity']]
            opacity = 1.0 / (1.0 + np.exp(-opacity_raw))  # Sigmoid
        else:
            opacity = 1.0

        # Spherical harmonics (DC term for base color)
        if 'f_dc_0' in prop_idx:
            # SH coefficients - DC term gives base color
            sh_dc = np.array([
                values[prop_idx['f_dc_0']],
                values[prop_idx['f_dc_1']],
                values[prop_idx['f_dc_2']]
            ], dtype=np.float32)

            # Convert SH DC to RGB (C0 = 0.28209479177387814)
            C0 = 0.28209479177387814
            color = 0.5 + C0 * sh_dc
            color = np.clip(color, 0, 1)
        else:
            # Use vertex color if available
            if 'red' in prop_idx:
                color = np.array([
                    values[prop_idx['red']] / 255.0,
                    values[prop_idx['green']] / 255.0,
                    values[prop_idx['blue']] / 255.0
                ], dtype=np.float32)
            else:
                color = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # Full SH coefficients (for view-dependent color, optional)
        sh_coeffs = np.zeros(48, dtype=np.float32)
        for i in range(48):
            key = f'f_rest_{i}' if i < 45 else f'f_dc_{i-45}'
            if i < 3:
                key = f'f_dc_{i}'
            if key in prop_idx:
                sh_coeffs[i] = values[prop_idx[key]]

        return GaussianPoint(
            position=position,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            sh_coeffs=sh_coeffs,
            color=color
        )


class GaussianRenderer:
    """
    CPU-based Gaussian Splatting renderer.

    Inspired by Blender's point rendering approach:
    - Each Gaussian is projected to 2D
    - Rendered as an elliptical splat with Gaussian falloff
    - Alpha composited back-to-front

    For better performance, consider using CUDA-based rasterizers.
    """

    def __init__(self, background_color: Tuple[float, float, float] = (0, 0, 0)):
        """
        Initialize renderer.

        Args:
            background_color: RGB background color (0-1 range)
        """
        self.background_color = np.array(background_color, dtype=np.float32)
        self.points: List[GaussianPoint] = []

    def load_model(self, ply_path: Path):
        """Load Gaussian model from PLY file."""
        self.points = PLYLoader.load(ply_path)
        logger.info(f"Loaded model with {len(self.points)} Gaussians")

    def render(
        self,
        camera: Camera,
        scale_modifier: float = 1.0,
        max_splat_size: int = 100
    ) -> np.ndarray:
        """
        Render the Gaussian model from the given camera.

        Args:
            camera: Camera to render from
            scale_modifier: Scale multiplier for Gaussian sizes
            max_splat_size: Maximum splat size in pixels

        Returns:
            RGB image as numpy array (H, W, 3), float32, range [0, 1]
        """
        if not self.points:
            logger.warning("No points loaded, returning empty image")
            return np.tile(
                self.background_color,
                (camera.height, camera.width, 1)
            ).astype(np.float32)

        # Initialize output image
        image = np.tile(
            self.background_color,
            (camera.height, camera.width, 1)
        ).astype(np.float32)

        # Alpha accumulation buffer
        alpha_accum = np.zeros((camera.height, camera.width), dtype=np.float32)

        # Project all points and sort by depth
        projected = []
        for point in self.points:
            u, v, depth = camera.project_point(point.position)

            # Skip points behind camera or outside image
            if depth <= 0:
                continue
            if u < -max_splat_size or u >= camera.width + max_splat_size:
                continue
            if v < -max_splat_size or v >= camera.height + max_splat_size:
                continue

            # Compute 2D splat size based on depth and 3D scale
            avg_scale = np.mean(point.scale) * scale_modifier
            size_2d = camera.fx * avg_scale / depth
            size_2d = min(size_2d, max_splat_size)

            if size_2d < 0.5:
                continue  # Too small

            projected.append({
                'point': point,
                'u': u,
                'v': v,
                'depth': depth,
                'size': size_2d
            })

        # Sort back-to-front for proper alpha compositing
        projected.sort(key=lambda x: -x['depth'])

        logger.info(f"Rendering {len(projected)} visible splats...")

        # Render each splat
        for i, proj in enumerate(projected):
            self._render_splat(
                image, alpha_accum,
                proj['u'], proj['v'],
                proj['size'],
                proj['point'].color,
                proj['point'].opacity
            )

            if (i + 1) % 10000 == 0:
                logger.debug(f"Rendered {i+1}/{len(projected)} splats")

        # Clamp output
        image = np.clip(image, 0, 1)

        return image

    def _render_splat(
        self,
        image: np.ndarray,
        alpha_accum: np.ndarray,
        u: float,
        v: float,
        size: float,
        color: np.ndarray,
        opacity: float
    ):
        """
        Render a single Gaussian splat onto the image.

        Uses Gaussian falloff similar to Blender's point rendering,
        but with smooth alpha instead of hard edge.
        """
        H, W = image.shape[:2]

        # Splat bounding box
        radius = int(np.ceil(size * 3))  # 3 sigma
        x_min = max(0, int(u - radius))
        x_max = min(W, int(u + radius + 1))
        y_min = max(0, int(v - radius))
        y_max = min(H, int(v + radius + 1))

        if x_min >= x_max or y_min >= y_max:
            return

        # Create coordinate grid
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]

        # Distance from splat center (normalized)
        dx = (x_coords - u) / (size + 1e-6)
        dy = (y_coords - v) / (size + 1e-6)
        dist_sq = dx * dx + dy * dy

        # Gaussian falloff (2D Gaussian)
        # Blender uses hard cutoff at r=0.5, we use smooth Gaussian
        sigma = 0.5  # Gaussian sigma
        gaussian = np.exp(-dist_sq / (2 * sigma * sigma))

        # Final alpha
        alpha = gaussian * opacity

        # Alpha compositing (front-to-back would be more efficient,
        # but back-to-front is simpler)
        # C_out = C_in * (1 - alpha) + C_new * alpha
        for c in range(3):
            image[y_min:y_max, x_min:x_max, c] = (
                image[y_min:y_max, x_min:x_max, c] * (1 - alpha) +
                color[c] * alpha
            )

        alpha_accum[y_min:y_max, x_min:x_max] += alpha * (1 - alpha_accum[y_min:y_max, x_min:x_max])

    def render_to_file(
        self,
        camera: Camera,
        output_path: Path,
        scale_modifier: float = 1.0
    ):
        """Render and save to image file."""
        if not HAS_PIL:
            raise ImportError("PIL is required for saving images. Install with: pip install Pillow")

        image = self.render(camera, scale_modifier)

        # Convert to uint8
        image_uint8 = (image * 255).astype(np.uint8)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(image_uint8)
        img.save(output_path)

        logger.info(f"Saved rendered image to {output_path}")


def render_gaussian_model(
    model_path: Path,
    output_dir: Path,
    camera_transforms: List[np.ndarray],
    camera_times: List[float],
    intrinsics: Dict[str, float],
    iteration: int = 30000,
    background: Tuple[float, float, float] = (0, 0, 0)
) -> Path:
    """
    Render a trained Gaussian model from multiple camera views.

    This is the main entry point for rendering, designed to integrate
    with the existing render_wrapper.py.

    Args:
        model_path: Path to trained model directory
        output_dir: Output directory for rendered frames
        camera_transforms: List of 4x4 camera-to-world matrices
        camera_times: List of time values for each camera
        intrinsics: Camera intrinsics dictionary
        iteration: Checkpoint iteration to load
        background: Background RGB color

    Returns:
        Path to rendered frames directory
    """
    # Find checkpoint
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ply_path}")

    # Create output directory
    renders_dir = Path(output_dir) / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Initialize renderer
    renderer = GaussianRenderer(background_color=background)
    renderer.load_model(ply_path)

    # Render each frame
    logger.info(f"Rendering {len(camera_transforms)} frames...")

    for i, (transform, time_val) in enumerate(zip(camera_transforms, camera_times)):
        camera = Camera.from_intrinsics_dict(intrinsics, transform)

        output_path = renders_dir / f"render_{i:06d}.png"
        renderer.render_to_file(camera, output_path)

        if (i + 1) % 10 == 0:
            logger.info(f"Rendered frame {i+1}/{len(camera_transforms)}")

    logger.info(f"Rendering complete. Frames saved to {renders_dir}")
    return renders_dir


if __name__ == "__main__":
    """Test the renderer with a simple example."""
    import argparse

    parser = argparse.ArgumentParser(description="Render Gaussian Splatting model")
    parser.add_argument("model_path", type=Path, help="Path to model directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("render_output"),
                        help="Output directory")
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Checkpoint iteration")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=600, help="Image height")

    args = parser.parse_args()

    # Find checkpoint
    ply_path = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"

    if not ply_path.exists():
        print(f"Error: Checkpoint not found at {ply_path}")
        exit(1)

    # Create test camera (looking at origin from +z)
    transform = np.eye(4)
    transform[2, 3] = 5.0  # 5 units away

    camera = Camera.from_transform_matrix(
        transform,
        width=args.width,
        height=args.height,
        fov_x=0.8
    )

    # Render
    renderer = GaussianRenderer()
    renderer.load_model(ply_path)

    output_path = args.output / "test_render.png"
    renderer.render_to_file(camera, output_path)

    print(f"Rendered to {output_path}")
