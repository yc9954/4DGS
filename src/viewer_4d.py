#!/usr/bin/env python3
"""
4D Gaussian Splatting Interactive Viewer

Real-time interactive viewer for 4D Gaussian Splatting models.
Allows navigation through 3D space AND time simultaneously.

Controls:
    Camera Movement:
        W/S         - Move forward/backward
        A/D         - Move left/right
        Q/E         - Move up/down
        Mouse Drag  - Look around (rotate camera)
        Scroll      - Zoom in/out

    Time Control:
        LEFT/RIGHT  - Step through time
        SPACE       - Play/Pause animation
        HOME/END    - Jump to start/end
        1-9         - Jump to 10%-90% of timeline

    Other:
        R           - Reset camera to initial position
        T           - Toggle time display
        F           - Toggle FPS display
        P           - Screenshot
        ESC         - Exit

Usage:
    python src/viewer_4d.py --model data/outputs/my_experiment/model
    python src/viewer_4d.py --model data/outputs/my_experiment/model --time 0.5
"""

import sys
import argparse
import time
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable
import numpy as np

# Check for required packages
try:
    import torch
    import cv2
    HAS_OPENCV = True
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install torch opencv-python")
    sys.exit(1)

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


@dataclass
class Camera:
    """Interactive camera with position and orientation."""

    # Position in world space
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 4.0]))

    # Euler angles (in radians)
    yaw: float = 0.0      # Rotation around Y axis (left/right)
    pitch: float = 0.0    # Rotation around X axis (up/down)
    roll: float = 0.0     # Rotation around Z axis (tilt)

    # Camera intrinsics
    fov: float = 60.0     # Field of view in degrees
    aspect: float = 16/9  # Aspect ratio
    near: float = 0.1
    far: float = 100.0

    # Movement settings
    move_speed: float = 0.1
    look_speed: float = 0.003
    zoom_speed: float = 0.5

    def get_forward(self) -> np.ndarray:
        """Get forward direction vector."""
        return np.array([
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            -math.cos(self.yaw) * math.cos(self.pitch)
        ])

    def get_right(self) -> np.ndarray:
        """Get right direction vector."""
        return np.array([
            math.cos(self.yaw),
            0,
            math.sin(self.yaw)
        ])

    def get_up(self) -> np.ndarray:
        """Get up direction vector."""
        forward = self.get_forward()
        right = self.get_right()
        return np.cross(right, forward)

    def move_forward(self, amount: float):
        """Move camera forward/backward."""
        self.position += self.get_forward() * amount * self.move_speed

    def move_right(self, amount: float):
        """Move camera left/right."""
        self.position += self.get_right() * amount * self.move_speed

    def move_up(self, amount: float):
        """Move camera up/down."""
        self.position += np.array([0, 1, 0]) * amount * self.move_speed

    def rotate(self, delta_yaw: float, delta_pitch: float):
        """Rotate camera (look around)."""
        self.yaw += delta_yaw * self.look_speed
        self.pitch = np.clip(
            self.pitch + delta_pitch * self.look_speed,
            -math.pi / 2 + 0.01,
            math.pi / 2 - 0.01
        )

    def zoom(self, amount: float):
        """Zoom in/out (move forward along view direction)."""
        self.position += self.get_forward() * amount * self.zoom_speed

    def get_view_matrix(self) -> np.ndarray:
        """Get 4x4 view matrix (world to camera)."""
        forward = self.get_forward()
        right = self.get_right()
        up = self.get_up()

        # Rotation matrix
        rotation = np.eye(4)
        rotation[0, :3] = right
        rotation[1, :3] = up
        rotation[2, :3] = -forward

        # Translation
        translation = np.eye(4)
        translation[:3, 3] = -self.position

        return rotation @ translation

    def get_transform_matrix(self) -> np.ndarray:
        """Get 4x4 camera-to-world transform matrix (for rendering)."""
        forward = self.get_forward()
        right = self.get_right()
        up = self.get_up()

        transform = np.eye(4)
        transform[:3, 0] = right
        transform[:3, 1] = up
        transform[:3, 2] = -forward
        transform[:3, 3] = self.position

        return transform

    def get_intrinsics(self, width: int, height: int) -> dict:
        """Get camera intrinsics as dictionary."""
        fov_rad = math.radians(self.fov)
        focal = width / (2 * math.tan(fov_rad / 2))

        return {
            "camera_angle_x": fov_rad,
            "camera_angle_y": fov_rad * height / width,
            "fl_x": focal,
            "fl_y": focal,
            "cx": width / 2,
            "cy": height / 2,
            "w": width,
            "h": height,
        }

    def reset(self):
        """Reset camera to initial position."""
        self.position = np.array([0.0, 0.0, 4.0])
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0


@dataclass
class TimeController:
    """Controls time for 4D playback."""

    current_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 1.0
    time_step: float = 0.02  # Step size for manual control
    playback_speed: float = 0.5  # Playback speed multiplier
    is_playing: bool = False
    loop: bool = True

    def update(self, delta_time: float):
        """Update time if playing."""
        if self.is_playing:
            self.current_time += delta_time * self.playback_speed
            if self.current_time > self.max_time:
                if self.loop:
                    self.current_time = self.min_time
                else:
                    self.current_time = self.max_time
                    self.is_playing = False

    def step_forward(self):
        """Step forward in time."""
        self.current_time = min(self.current_time + self.time_step, self.max_time)

    def step_backward(self):
        """Step backward in time."""
        self.current_time = max(self.current_time - self.time_step, self.min_time)

    def toggle_play(self):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing

    def jump_to(self, normalized_time: float):
        """Jump to a specific time (0.0 to 1.0)."""
        self.current_time = self.min_time + normalized_time * (self.max_time - self.min_time)

    def get_normalized_time(self) -> float:
        """Get current time as normalized value (0.0 to 1.0)."""
        if self.max_time == self.min_time:
            return 0.0
        return (self.current_time - self.min_time) / (self.max_time - self.min_time)


class Gaussian4DRenderer:
    """
    Renderer for 4D Gaussian Splatting models.

    Loads a trained 4DGS model and renders it at specified camera pose and time.
    """

    def __init__(self, model_path: Path, device: str = "cuda"):
        """
        Initialize the renderer.

        Args:
            model_path: Path to the trained model directory
            device: Device to use for rendering ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.gaussian_model = None
        self.pipeline = None
        self.background = None

        self._load_model()

    def _find_4dgs_module(self) -> Optional[Path]:
        """Find the 4DGaussians module path."""
        possible_paths = [
            Path(__file__).parent.parent / "submodules" / "4dgs",
            Path(__file__).parent.parent / "submodules" / "4DGaussians",
            Path.cwd() / "submodules" / "4dgs",
            Path.cwd() / "submodules" / "4DGaussians",
        ]

        for path in possible_paths:
            if path.exists() and (path / "train.py").exists():
                return path

        return None

    def _load_model(self):
        """Load the 4DGS model."""
        # Find and add 4DGS to path
        module_path = self._find_4dgs_module()

        if module_path is None:
            print("Warning: 4DGaussians module not found. Using mock renderer.")
            print("To enable real rendering, run setup_runpod.sh first.")
            self.use_mock = True
            return

        self.use_mock = False

        # Add to Python path
        if str(module_path) not in sys.path:
            sys.path.insert(0, str(module_path))

        try:
            # Import 4DGS modules
            from scene import GaussianModel
            from scene.cameras import Camera as GSCamera
            from gaussian_renderer import render
            from arguments import ModelParams, PipelineParams
            from argparse import Namespace

            # Load model configuration
            cfg_path = self.model_path / "cfg_args"
            if cfg_path.exists():
                # Parse saved arguments
                pass  # TODO: Load from cfg_args

            # Initialize Gaussian model
            self.gaussian_model = GaussianModel(3)  # sh_degree=3

            # Find latest checkpoint
            ckpt_dirs = sorted(
                (self.model_path / "point_cloud").glob("iteration_*"),
                key=lambda x: int(x.name.split("_")[1])
            )

            if ckpt_dirs:
                latest_ckpt = ckpt_dirs[-1]
                print(f"Loading checkpoint: {latest_ckpt}")
                self.gaussian_model.load_ply(str(latest_ckpt / "point_cloud.ply"))
            else:
                raise FileNotFoundError(f"No checkpoints found in {self.model_path}")

            # Setup pipeline
            self.pipeline = PipelineParams(Namespace())
            self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)

            print(f"Model loaded: {self.gaussian_model.get_xyz.shape[0]} Gaussians")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock renderer.")
            self.use_mock = True

    def render(
        self,
        camera: Camera,
        time_value: float,
        width: int = 800,
        height: int = 600
    ) -> np.ndarray:
        """
        Render the scene at the given camera pose and time.

        Args:
            camera: Camera object with position and orientation
            time_value: Time value (0.0 to 1.0) for 4D animation
            width: Output image width
            height: Output image height

        Returns:
            RGB image as numpy array (H, W, 3), values 0-255
        """
        if self.use_mock:
            return self._render_mock(camera, time_value, width, height)

        return self._render_4dgs(camera, time_value, width, height)

    def _render_4dgs(
        self,
        camera: Camera,
        time_value: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """Render using actual 4DGS model."""
        try:
            from gaussian_renderer import render
            from scene.cameras import Camera as GSCamera

            # Create camera for 4DGS
            intrinsics = camera.get_intrinsics(width, height)
            transform = camera.get_transform_matrix()

            # Convert to 4DGS camera format
            R = transform[:3, :3].T  # World to camera rotation
            T = -R @ transform[:3, 3]  # World to camera translation

            gs_camera = GSCamera(
                colmap_id=0,
                R=R,
                T=T,
                FoVx=intrinsics["camera_angle_x"],
                FoVy=intrinsics["camera_angle_y"],
                image=torch.zeros((3, height, width)),
                gt_alpha_mask=None,
                image_name="viewer",
                uid=0,
                time=time_value,
            )

            # Render
            with torch.no_grad():
                result = render(
                    gs_camera,
                    self.gaussian_model,
                    self.pipeline,
                    self.background,
                    time_value,
                )

            # Convert to numpy
            image = result["render"].cpu().numpy()
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

            return image

        except Exception as e:
            print(f"Render error: {e}")
            return self._render_mock(camera, time_value, width, height)

    def _render_mock(
        self,
        camera: Camera,
        time_value: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Mock renderer for testing without actual model.
        Generates a procedural visualization of 3D Gaussians.
        """
        # Create image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Background gradient based on camera orientation
        for y in range(height):
            t = y / height
            r = int(20 + 30 * t + 20 * math.sin(camera.yaw))
            g = int(20 + 20 * t + 20 * math.cos(camera.pitch))
            b = int(40 + 40 * t)
            image[y, :] = [r, g, b]

        # Generate mock Gaussian positions (on a grid that deforms with time)
        np.random.seed(42)  # Consistent random
        n_gaussians = 200

        # Base positions
        base_pos = np.random.randn(n_gaussians, 3) * 2

        # Time-based deformation (wave effect)
        deform = np.zeros_like(base_pos)
        deform[:, 0] = 0.3 * np.sin(time_value * 2 * np.pi + base_pos[:, 1])
        deform[:, 1] = 0.3 * np.cos(time_value * 2 * np.pi + base_pos[:, 0])
        deform[:, 2] = 0.2 * np.sin(time_value * 4 * np.pi + base_pos[:, 0] * base_pos[:, 1])

        positions = base_pos + deform

        # Colors based on position
        colors = np.zeros((n_gaussians, 3))
        colors[:, 0] = np.clip((positions[:, 0] + 2) / 4 * 255, 50, 255)  # R
        colors[:, 1] = np.clip((positions[:, 1] + 2) / 4 * 255, 50, 255)  # G
        colors[:, 2] = np.clip((positions[:, 2] + 2) / 4 * 255, 50, 255)  # B

        # Project to screen
        view_matrix = camera.get_view_matrix()
        intrinsics = camera.get_intrinsics(width, height)

        for i in range(n_gaussians):
            # Transform to camera space
            pos_world = np.append(positions[i], 1.0)
            pos_cam = view_matrix @ pos_world

            # Skip if behind camera
            if pos_cam[2] >= -0.1:
                continue

            # Project to screen
            x = -pos_cam[0] / pos_cam[2] * intrinsics["fl_x"] + intrinsics["cx"]
            y = -pos_cam[1] / pos_cam[2] * intrinsics["fl_y"] + intrinsics["cy"]

            # Skip if outside screen
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # Size based on distance
            size = max(2, int(30 / (-pos_cam[2] + 1)))

            # Draw Gaussian (as circle with gradient)
            color = colors[i].astype(np.uint8)
            cv2.circle(image, (int(x), int(y)), size, color.tolist(), -1)
            cv2.circle(image, (int(x), int(y)), size + 2,
                      (int(color[0]*0.7), int(color[1]*0.7), int(color[2]*0.7)), 1)

        return image


class Viewer4D:
    """
    Main 4D Gaussian Splatting interactive viewer.

    Provides real-time interactive viewing of 4DGS models with
    camera control and time navigation.
    """

    def __init__(
        self,
        model_path: Path,
        width: int = 1280,
        height: int = 720,
        title: str = "4D Gaussian Splatting Viewer"
    ):
        """
        Initialize the viewer.

        Args:
            model_path: Path to the trained model
            width: Window width
            height: Window height
            title: Window title
        """
        self.width = width
        self.height = height
        self.title = title
        self.running = False

        # Initialize components
        self.camera = Camera()
        self.camera.aspect = width / height

        self.time_controller = TimeController()
        self.renderer = Gaussian4DRenderer(model_path)

        # Display settings
        self.show_ui = True
        self.show_fps = True
        self.show_time = True
        self.show_controls = True

        # Mouse state
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)

        # FPS tracking
        self.fps = 0.0
        self.frame_times: List[float] = []

        # Screenshot counter
        self.screenshot_counter = 0

    def run_opencv(self):
        """Run viewer using OpenCV (fallback mode)."""
        print("\n" + "=" * 60)
        print("4D Gaussian Splatting Interactive Viewer")
        print("=" * 60)
        print("\nControls:")
        print("  WASD      - Move camera")
        print("  QE        - Move up/down")
        print("  Arrow L/R - Change time")
        print("  SPACE     - Play/Pause")
        print("  R         - Reset camera")
        print("  P         - Screenshot")
        print("  H         - Toggle help")
        print("  ESC       - Exit")
        print("=" * 60 + "\n")

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.width, self.height)

        # Mouse callback
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mouse_pressed = True
                self.last_mouse_pos = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.mouse_pressed = False
            elif event == cv2.EVENT_MOUSEMOVE and self.mouse_pressed:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.camera.rotate(dx, dy)
                self.last_mouse_pos = (x, y)
            elif event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    self.camera.zoom(1)
                else:
                    self.camera.zoom(-1)

        cv2.setMouseCallback(self.title, mouse_callback)

        self.running = True
        last_time = time.time()

        while self.running:
            frame_start = time.time()

            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            # Update time controller
            self.time_controller.update(delta_time)

            # Render frame
            frame = self.renderer.render(
                self.camera,
                self.time_controller.current_time,
                self.width,
                self.height
            )

            # Draw UI overlay
            if self.show_ui:
                self._draw_ui_opencv(frame)

            # Display
            cv2.imshow(self.title, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self._handle_key_opencv(key)

            # Update FPS
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            self.fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0

        cv2.destroyAllWindows()

    def _handle_key_opencv(self, key: int):
        """Handle keyboard input for OpenCV mode."""
        if key == 27:  # ESC
            self.running = False
        elif key == ord('w'):
            self.camera.move_forward(1)
        elif key == ord('s'):
            self.camera.move_forward(-1)
        elif key == ord('a'):
            self.camera.move_right(-1)
        elif key == ord('d'):
            self.camera.move_right(1)
        elif key == ord('q'):
            self.camera.move_up(1)
        elif key == ord('e'):
            self.camera.move_up(-1)
        elif key == ord('r'):
            self.camera.reset()
            self.time_controller.current_time = 0.0
        elif key == ord(' '):
            self.time_controller.toggle_play()
        elif key == 81 or key == 2:  # Left arrow
            self.time_controller.step_backward()
        elif key == 83 or key == 3:  # Right arrow
            self.time_controller.step_forward()
        elif key == ord('h'):
            self.show_controls = not self.show_controls
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('t'):
            self.show_time = not self.show_time
        elif key == ord('p'):
            self._take_screenshot()
        elif key >= ord('1') and key <= ord('9'):
            # Jump to percentage
            pct = (key - ord('0')) / 10.0
            self.time_controller.jump_to(pct)

    def _draw_ui_opencv(self, frame: np.ndarray):
        """Draw UI overlay on frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Time bar at bottom
        if self.show_time:
            bar_height = 30
            bar_y = self.height - bar_height - 10
            bar_width = self.width - 40
            bar_x = 20

            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (40, 40, 40), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), 1)

            # Progress
            progress = self.time_controller.get_normalized_time()
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                         (80, 180, 80), -1)

            # Time text
            time_text = f"Time: {self.time_controller.current_time:.3f}"
            play_text = "▶ Playing" if self.time_controller.is_playing else "⏸ Paused"
            cv2.putText(frame, time_text, (bar_x + 10, bar_y + 20),
                       font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, play_text, (bar_x + bar_width - 100, bar_y + 20),
                       font, 0.5, (255, 255, 255), 1)

        # FPS counter
        if self.show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (self.width - 100, 30),
                       font, 0.6, (0, 255, 0), 1)

        # Camera info
        pos = self.camera.position
        cam_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        cv2.putText(frame, cam_text, (10, 30), font, 0.5, (200, 200, 200), 1)

        # Controls help
        if self.show_controls:
            controls = [
                "WASD: Move | QE: Up/Down",
                "Mouse: Look | Scroll: Zoom",
                "L/R Arrow: Time | Space: Play",
                "R: Reset | P: Screenshot | H: Hide",
            ]
            for i, text in enumerate(controls):
                y = 60 + i * 20
                cv2.putText(frame, text, (10, y), font, 0.4, (150, 150, 150), 1)

    def _take_screenshot(self):
        """Save current frame as screenshot."""
        frame = self.renderer.render(
            self.camera,
            self.time_controller.current_time,
            self.width,
            self.height
        )

        filename = f"screenshot_{self.screenshot_counter:04d}.png"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Screenshot saved: {filename}")
        self.screenshot_counter += 1

    def run(self):
        """Run the viewer."""
        if HAS_PYGAME:
            self._run_pygame()
        else:
            self.run_opencv()

    def _run_pygame(self):
        """Run viewer using Pygame (preferred mode)."""
        pygame.init()
        pygame.display.set_caption(self.title)
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()

        print("\n" + "=" * 60)
        print("4D Gaussian Splatting Interactive Viewer (Pygame)")
        print("=" * 60)
        print("\nControls:")
        print("  WASD/Arrows - Move camera")
        print("  QE          - Move up/down")
        print("  Mouse Drag  - Look around")
        print("  Scroll      - Zoom")
        print("  LEFT/RIGHT  - Change time")
        print("  SPACE       - Play/Pause")
        print("  R           - Reset camera")
        print("  P           - Screenshot")
        print("  ESC         - Exit")
        print("=" * 60 + "\n")

        self.running = True
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

        while self.running:
            delta_time = clock.tick(60) / 1000.0  # 60 FPS target

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key_pygame(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.mouse_pressed = True
                        self.last_mouse_pos = pygame.mouse.get_pos()
                    elif event.button == 4:  # Scroll up
                        self.camera.zoom(1)
                    elif event.button == 5:  # Scroll down
                        self.camera.zoom(-1)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_pressed = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_pressed:
                        x, y = pygame.mouse.get_pos()
                        dx = x - self.last_mouse_pos[0]
                        dy = y - self.last_mouse_pos[1]
                        self.camera.rotate(dx, dy)
                        self.last_mouse_pos = (x, y)

            # Continuous key input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.camera.move_forward(1)
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.camera.move_forward(-1)
            if keys[pygame.K_a]:
                self.camera.move_right(-1)
            if keys[pygame.K_d]:
                self.camera.move_right(1)
            if keys[pygame.K_q]:
                self.camera.move_up(1)
            if keys[pygame.K_e]:
                self.camera.move_up(-1)

            # Update time
            self.time_controller.update(delta_time)

            # Render
            frame = self.renderer.render(
                self.camera,
                self.time_controller.current_time,
                self.width,
                self.height
            )

            # Draw UI
            if self.show_ui:
                self._draw_ui_opencv(frame)  # Reuse OpenCV UI drawing

            # Display
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            # Update FPS
            self.fps = clock.get_fps()

        pygame.quit()

    def _handle_key_pygame(self, key):
        """Handle key press for Pygame mode."""
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.time_controller.toggle_play()
        elif key == pygame.K_LEFT:
            self.time_controller.step_backward()
        elif key == pygame.K_RIGHT:
            self.time_controller.step_forward()
        elif key == pygame.K_r:
            self.camera.reset()
            self.time_controller.current_time = 0.0
        elif key == pygame.K_h:
            self.show_controls = not self.show_controls
        elif key == pygame.K_f:
            self.show_fps = not self.show_fps
        elif key == pygame.K_t:
            self.show_time = not self.show_time
        elif key == pygame.K_p:
            self._take_screenshot()
        elif key == pygame.K_HOME:
            self.time_controller.jump_to(0.0)
        elif key == pygame.K_END:
            self.time_controller.jump_to(1.0)
        elif key in range(pygame.K_1, pygame.K_9 + 1):
            pct = (key - pygame.K_1 + 1) / 10.0
            self.time_controller.jump_to(pct)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="4D Gaussian Splatting Interactive Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View a trained model
  python src/viewer_4d.py --model data/outputs/my_experiment/model

  # Start at specific time
  python src/viewer_4d.py --model data/outputs/my_experiment/model --time 0.5

  # Custom window size
  python src/viewer_4d.py --model data/outputs/my_experiment/model --width 1920 --height 1080

Controls:
  WASD          Move camera
  QE            Move up/down
  Mouse Drag    Look around
  Scroll        Zoom in/out
  LEFT/RIGHT    Step through time
  SPACE         Play/Pause animation
  1-9           Jump to 10%-90% time
  R             Reset camera
  P             Screenshot
  H             Toggle help
  ESC           Exit
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height (default: 720)"
    )
    parser.add_argument(
        "--time", "-t",
        type=float,
        default=0.0,
        help="Initial time value 0.0-1.0 (default: 0.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for rendering (default: cuda)"
    )

    args = parser.parse_args()

    # Validate model path
    if not args.model.exists():
        print(f"Error: Model path does not exist: {args.model}")
        print("\nHint: Train a model first with:")
        print("  ./run_pipeline.sh data/inputs/my_video my_experiment")
        sys.exit(1)

    # Create and run viewer
    viewer = Viewer4D(
        model_path=args.model,
        width=args.width,
        height=args.height
    )

    viewer.time_controller.current_time = args.time
    viewer.run()


if __name__ == "__main__":
    main()
