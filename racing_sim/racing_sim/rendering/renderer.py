"""PyGame renderer for the racing simulation."""

import pygame
import numpy as np
import math
from typing import Optional, List, Tuple
from pymunk import Vec2d
from racing_sim.config.config import RenderConfig
from racing_sim.physics.car import Car
from racing_sim.physics.track import Track
from racing_sim.sensors.lidar import Lidar


class Renderer:
    """PyGame-based renderer with debug visualization."""

    def __init__(self, config: RenderConfig, render_mode: str = "human"):
        """
        Initialize the renderer.

        Args:
            config: Render configuration
            render_mode: "human" for display window, "rgb_array" for numpy array
        """
        self.config = config
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self._initialized = False
        self._reset_requested = False

    def _init_pygame(self):
        """Initialize PyGame (deferred until first render)."""
        if self._initialized:
            return

        pygame.init()

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.config.screen_width, self.config.screen_height)
            )
            pygame.display.set_caption("Racing Simulation")
            self.clock = pygame.time.Clock()
        else:
            # For rgb_array mode, create a surface without display
            self.screen = pygame.Surface(
                (self.config.screen_width, self.config.screen_height)
            )

        self._initialized = True

    def render(
        self,
        car: Car,
        track: Track,
        lidar: Optional[Lidar] = None,
        info: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Render the current state.

        Args:
            car: Car to render
            track: Track to render
            lidar: Optional lidar for debug visualization
            info: Optional info dict for HUD display

        Returns:
            numpy array of pixels if render_mode is "rgb_array", None otherwise
        """
        self._init_pygame()

        # Clear screen
        self.screen.fill(self.config.background_color)

        # Render track
        self._render_track(track)

        # Render checkpoints (faint lines)
        self._render_checkpoints(track)

        # Render lidar rays if enabled
        if self.config.show_lidar and lidar is not None:
            self._render_lidar(lidar)

        # Render car
        self._render_car(car)

        # Render HUD
        if info is not None:
            self._render_hud(info, car)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.config.fps)
            return None
        else:
            # Return numpy array for rgb_array mode
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )

    def _render_track(self, track: Track):
        """Render track walls."""
        for segment in track.get_all_wall_segments():
            start = self._to_screen(segment[0])
            end = self._to_screen(segment[1])
            pygame.draw.line(self.screen, self.config.wall_color, start, end, 3)

    def _render_checkpoints(self, track: Track):
        """Render checkpoint lines (faint)."""
        checkpoint_color = (50, 50, 50)  # Faint gray
        for inner, outer in track.checkpoints:
            start = self._to_screen(inner)
            end = self._to_screen(outer)
            pygame.draw.line(self.screen, checkpoint_color, start, end, 1)

    def _render_car(self, car: Car, color: Optional[Tuple[int, int, int]] = None):
        """Render the car as a rotated rectangle."""
        car_color = color if color is not None else self.config.car_color
        corners = car.get_corners()
        screen_corners = [self._to_screen(c) for c in corners]
        pygame.draw.polygon(self.screen, car_color, screen_corners)

        # Draw a line indicating forward direction
        center = self._to_screen(car.position)
        forward = Vec2d(math.cos(car.angle), math.sin(car.angle))
        front = car.position + forward * (car.config.width / 2 + 5)
        front_screen = self._to_screen(front)
        pygame.draw.line(self.screen, (255, 255, 255), center, front_screen, 2)

    def _render_lidar(self, lidar: Lidar):
        """Render lidar rays with color indicating distance."""
        rays = lidar.get_debug_rays()

        for start, end, distance in rays:
            start_screen = self._to_screen(start)
            end_screen = self._to_screen(end)

            # Interpolate color based on distance (green=far, red=close)
            t = distance  # 0=hit close, 1=no hit/far
            r = int(
                self.config.lidar_hit_color[0] * (1 - t)
                + self.config.lidar_clear_color[0] * t
            )
            g = int(
                self.config.lidar_hit_color[1] * (1 - t)
                + self.config.lidar_clear_color[1] * t
            )
            b = int(
                self.config.lidar_hit_color[2] * (1 - t)
                + self.config.lidar_clear_color[2] * t
            )
            color = (r, g, b)

            pygame.draw.line(self.screen, color, start_screen, end_screen, 2)

            # Draw hit point
            if distance < 1.0:
                pygame.draw.circle(self.screen, (255, 255, 0), end_screen, 4)

    def _render_hud(self, info: dict, car: Car):
        """Render heads-up display with stats."""
        font = pygame.font.Font(None, 24)

        y_offset = 10
        texts = [
            f"Speed: {car.speed:.1f}",
            f"Checkpoint: {info.get('checkpoint', 0)}",
            f"Reward: {info.get('episode_reward', 0):.2f}",
            f"Step: {info.get('step', 0)}",
        ]

        for text in texts:
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20

    def _to_screen(self, point: Vec2d) -> Tuple[int, int]:
        """Convert Pymunk coordinates to screen coordinates."""
        # Pymunk uses bottom-left origin, PyGame uses top-left
        # Flip y-axis
        return (int(point.x), int(self.config.screen_height - point.y))

    def close(self):
        """Clean up PyGame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self.screen = None
            self.clock = None

    def handle_events(self) -> bool:
        """
        Handle PyGame events.

        Returns:
            False if quit event received, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self._reset_requested = True
        return True

    def get_keyboard_input(self) -> Tuple[float, float]:
        """
        Get steering and throttle from keyboard input.

        Returns:
            Tuple of (steering, throttle) from keyboard
        """
        keys = pygame.key.get_pressed()

        steering = 0.0
        throttle = 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 1.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0

        return steering, throttle

    def was_reset_requested(self) -> bool:
        """
        Check if reset was requested and clear the flag.

        Returns:
            True if 'r' key was pressed since last check
        """
        was_requested = self._reset_requested
        self._reset_requested = False
        return was_requested
