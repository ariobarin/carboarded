"""PyGame renderer for the racing simulation."""

import pygame
import numpy as np
import math
from typing import Optional, List, Tuple
from pymunk import Vec2d
from racing_sim.config.config import RenderConfig, GridConfig
from racing_sim.physics.car import Car
from racing_sim.physics.track import Track
from racing_sim.editor.node_track import NodeTrack
from racing_sim.sensors.lidar import Lidar
from racing_sim.sensors.grid import compute_grid, compute_grid_positions
from racing_sim.utils.checkpoints import smoothed_checkpoint_lines


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
        self._random_start_toggle_requested = False
        self._show_lidar = config.show_lidar
        self._show_grid = config.show_grid
        self._grid_debug_mode = (
            False  # False = fast overlay, True = world-space circles
        )
        self._pov_mode = False  # Car POV mode: stationary car view with sensors
        self.last_actions: Optional[dict] = None  # {"human": (steer, throttle), "ai": ...}
        self._view_offset = Vec2d(0.0, 0.0)
        self._needs_view_update = True
        self._last_track_id: Optional[int] = None
        # Cached fonts (initialized in _init_pygame)
        self._font_small: Optional[pygame.font.Font] = None
        self._font_hud: Optional[pygame.font.Font] = None

    def _init_pygame(self):
        """Initialize PyGame (deferred until first render)."""
        if self._initialized:
            return

        pygame.init()

        if self.render_mode == "human":
            # Use DOUBLEBUF for smoother rendering
            self.screen = pygame.display.set_mode(
                (self.config.screen_width, self.config.screen_height),
                pygame.DOUBLEBUF | pygame.RESIZABLE,
            )
            pygame.display.set_caption("Racing Simulation")
            self.clock = pygame.time.Clock()
        else:
            # For rgb_array mode, create a surface without display
            self.screen = pygame.Surface(
                (self.config.screen_width, self.config.screen_height)
            )

        # Cache fonts for faster rendering
        self._font_small = pygame.font.Font(None, 20)
        self._font_hud = pygame.font.Font(None, 24)

        self._initialized = True

    def render(
        self,
        car: Car,
        track: Track,
        lidar: Optional[Lidar] = None,
        info: Optional[dict] = None,
        ai_car: Optional[Car] = None,
        ai_info: Optional[dict] = None,
        sensor_car: Optional[Car] = None,
        obs_type: str = "lidar",
    ) -> Optional[np.ndarray]:
        """
        Render the current state.

        Args:
            car: Car to render (human car in race mode)
            track: Track to render
            lidar: Optional lidar for debug visualization
            info: Optional info dict for HUD display (human)
            ai_car: Optional second car to render (AI opponent in race mode)
            ai_info: Optional info dict for AI car in race mode
            sensor_car: Optional car to use for sensor visualization (defaults to car)
            obs_type: Observation type ("lidar" or "grid") for POV mode visualization

        Returns:
            numpy array of pixels if render_mode is "rgb_array", None otherwise
        """
        self._init_pygame()

        # POV mode: render stationary car view with sensor visualization
        if self._pov_mode:
            return self._render_pov_mode(car, track, lidar, info, obs_type)

        self._refresh_view(track)

        # Use sensor_car if provided, otherwise use car
        car_for_sensors = sensor_car if sensor_car is not None else car

        # Clear screen
        self.screen.fill(self.config.background_color)

        # Render track
        self._render_track(track)

        # Render checkpoints (faint lines)
        self._render_checkpoints(track)

        # Render lidar rays if enabled
        if self._show_lidar and lidar is not None:
            self._render_lidar(lidar)

        # Render CNN grid if enabled (from sensor car's perspective)
        if self._show_grid:
            if self._grid_debug_mode:
                self._render_grid_worldspace(car_for_sensors, track)
            else:
                self._render_grid_overlay(car_for_sensors, track)

        # Render AI car first (so human car appears on top)
        if ai_car is not None:
            self._render_car(ai_car, color=(255, 100, 100))  # Red for AI

        # Render human car
        self._render_car(car, color=(100, 150, 255))  # Blue for human

        # Render HUD
        if info is not None:
            self._render_hud(info, car, ai_info, ai_car)

        # Render action bars (steering/throttle visualization)
        if self.last_actions:
            self._render_action_bars(self.last_actions)

        self._render_help()

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
        checkpoints = track.checkpoints
        if isinstance(track, NodeTrack):
            checkpoints = smoothed_checkpoint_lines(checkpoints)
        for inner, outer in checkpoints:
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

    def _render_grid_overlay(
        self, car: Car, track: Track, grid_config: Optional[GridConfig] = None
    ):
        """Render grid as efficient overlay in corner of screen.

        Uses pygame.surfarray for fast numpy->surface conversion.
        ~2,592 draw calls -> 1 blit call.

        Args:
            car: Car whose grid to visualize.
            track: Track for occupancy checks.
            grid_config: Grid sensor config. Uses default GridConfig if None.
        """
        if grid_config is None:
            grid_config = GridConfig()

        grid = compute_grid(car.position, car.angle, track, grid_config)

        # Create RGB array from binary grid (e.g., 36x36 -> 36x36x3)
        rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        on_mask = grid > 0
        rgb[on_mask] = self.config.grid_on_color
        rgb[~on_mask] = self.config.grid_off_color

        # Flip columns so col 0 (right of car) appears on right of screen
        rgb = np.fliplr(rgb)

        # Create surface (swapaxes for pygame's column-major format)
        # Use convert() for faster blitting (matches display format)
        grid_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1)).convert()
        scaled = pygame.transform.scale(grid_surface, (144, 144))  # 4x scale

        # Blit to top-right corner with border
        pos = (self.config.screen_width - 154, 10)
        pygame.draw.rect(self.screen, (255, 255, 255), (*pos, 148, 148), 2)
        self.screen.blit(scaled, (pos[0] + 2, pos[1] + 2))

    def _render_grid_worldspace(
        self, car: Car, track: Track, grid_config: Optional[GridConfig] = None
    ):
        """Render grid as circles in world space (slow, for debugging).

        Uses compute_grid/compute_grid_positions from sensors.grid so the
        visualization matches exactly what the CNN agent observes.

        Args:
            car: Car whose grid to visualize.
            track: Track for occupancy checks.
            grid_config: Grid sensor config. Uses default GridConfig if None.
        """
        if grid_config is None:
            grid_config = GridConfig()

        grid = compute_grid(car.position, car.angle, track, grid_config)
        positions = compute_grid_positions(car.position, car.angle, grid_config)

        for row, col, world_pos in positions:
            on_track = grid[row, col] > 0
            color = (
                self.config.grid_on_color if on_track else self.config.grid_off_color
            )

            screen_pos = self._to_screen(world_pos)
            # Single circle per cell (removed white outline for minor speedup)
            pygame.draw.circle(self.screen, color, screen_pos, 4)

    def _render_pov_mode(
        self,
        car: Car,
        track: Track,
        lidar: Optional[Lidar],
        info: Optional[dict],
        obs_type: str = "lidar",
    ) -> Optional[np.ndarray]:
        """Render in POV mode - stationary car view with sensor visualization.
        
        Args:
            car: Car to get sensor data from.
            track: Track for grid computation.
            lidar: Lidar sensor (if using lidar obs).
            info: Info dict for HUD.
            obs_type: "lidar" or "grid" observation type.
        """
        # Dark background (car can't see track)
        self.screen.fill((30, 30, 30))
        
        # Fixed car position at bottom-center of screen
        car_screen_x = self.config.screen_width // 2
        car_screen_y = self.config.screen_height - self.config.pov_car_offset_y
        
        # Render sensor visualization based on obs_type (respects L/G/V toggles)
        if obs_type == "grid" and self._show_grid:
            if self._grid_debug_mode:
                self._render_pov_grid_worldspace(car, track, car_screen_x, car_screen_y)
            else:
                self._render_pov_grid(car, track, car_screen_x, car_screen_y)
        elif self._show_lidar and lidar is not None:
            self._render_pov_lidar(lidar, car_screen_x, car_screen_y)
        
        # Render stationary car (always on top)
        self._render_pov_car(car_screen_x, car_screen_y)
        
        # Render minimal HUD
        if info is not None:
            self._render_pov_hud(info, car)
        
        # POV mode indicator
        pov_text = self._font_hud.render("POV MODE (C to exit)", True, (200, 200, 200))
        self.screen.blit(pov_text, (self.config.screen_width // 2 - 80, 10))

        # Render action bars (steering/throttle visualization)
        if self.last_actions:
            self._render_action_bars(self.last_actions)

        self._render_help()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.config.fps)
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )

    def _render_pov_car(self, screen_x: int, screen_y: int):
        """Render a stationary car at fixed screen position (pointing up)."""
        # Car dimensions (scaled for visibility)
        car_width = 20
        car_height = 40
        
        # Draw car body (rectangle pointing up)
        car_rect = pygame.Rect(
            screen_x - car_width // 2,
            screen_y - car_height // 2,
            car_width,
            car_height
        )
        pygame.draw.rect(self.screen, (100, 150, 255), car_rect)
        
        # Draw forward indicator (line pointing up)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (screen_x, screen_y),
            (screen_x, screen_y - car_height // 2 - 10),
            3
        )

    def _render_pov_lidar(self, lidar: Lidar, car_x: int, car_y: int):
        """Render lidar rays from stationary car position.
        
        Rays emanate upward (forward direction) from the car.
        """
        rays = lidar.get_debug_rays()
        if not rays:
            return
        
        # Get ray angles from lidar config
        ray_angles_rad = lidar.ray_angles_rad
        max_ray_length = self.config.pov_lidar_max_length
        
        for i, (start, end, distance) in enumerate(rays):
            # Get the ray's relative angle (from lidar config)
            if i < len(ray_angles_rad):
                ray_angle = ray_angles_rad[i]
            else:
                ray_angle = 0
            
            # In POV mode, forward is UP on screen
            # Pygame coords: Y increases downward, so "up" = negative Y = angle -π/2
            # ray_angle convention: positive = left of forward (counterclockwise)
            #                       negative = right of forward (clockwise)
            # 
            # For forward ray (ray_angle=0), we want screen angle = -π/2 (up)
            # For left ray (+60°), we want upper-left on screen = -150° = -π/2 - 60°
            # For right ray (-60°), we want upper-right on screen = -30° = -π/2 + 60°
            #
            # Formula: screen_angle = -π/2 - ray_angle
            screen_angle = -math.pi / 2 - ray_angle
            
            # Calculate ray end point on screen
            ray_length = distance * max_ray_length
            end_x = car_x + math.cos(screen_angle) * ray_length
            end_y = car_y + math.sin(screen_angle) * ray_length
            
            # Color based on distance (red=close, green=far)
            t = distance
            r = int(self.config.lidar_hit_color[0] * (1 - t) + self.config.lidar_clear_color[0] * t)
            g = int(self.config.lidar_hit_color[1] * (1 - t) + self.config.lidar_clear_color[1] * t)
            b = int(self.config.lidar_hit_color[2] * (1 - t) + self.config.lidar_clear_color[2] * t)
            color = (r, g, b)
            
            pygame.draw.line(self.screen, color, (car_x, car_y), (int(end_x), int(end_y)), 3)
            
            # Draw hit point indicator
            if distance < 1.0:
                pygame.draw.circle(self.screen, (255, 255, 0), (int(end_x), int(end_y)), 5)

    def _render_pov_grid(self, car: Car, track: Track, car_x: int, car_y: int):
        """Render occupancy grid above the stationary car.
        
        Grid coordinate system (from compute_grid):
        - Row 0 = farthest forward (far_distance) → should be TOP of display
        - Row N-1 = closest to car (near_distance) → should be BOTTOM of display
        - Col 0 = RIGHT of car → should be RIGHT of screen
        - Col N-1 = LEFT of car → should be LEFT of screen
        
        After swapaxes(0,1): grid[row,col] → surface[col,row] → pygame x=col, y=row
        - pygame y=0 (top) gets row 0 (far) → CORRECT
        - pygame x=0 (left) gets col 0 (right of car) → WRONG, need to flip
        """
        from racing_sim.config.config import GridConfig
        grid_config = GridConfig()
        
        grid = compute_grid(car.position, car.angle, track, grid_config)
        
        # Create RGB array from binary grid
        rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        on_mask = grid > 0
        rgb[on_mask] = self.config.grid_on_color
        rgb[~on_mask] = self.config.grid_off_color
        
        # Flip columns so col 0 (right of car) appears on right of screen
        # After fliplr: col 0 → col N-1 position, which after swapaxes → x=N-1 (right)
        rgb = np.fliplr(rgb)
        
        # Create and scale surface
        grid_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1)).convert()
        # Scale to larger size for visibility
        grid_size = min(self.config.pov_grid_max_size, car_y - 60)
        scaled = pygame.transform.scale(grid_surface, (grid_size, grid_size))
        
        # Position grid above the car
        grid_x = car_x - grid_size // 2
        grid_y = car_y - grid_size - 25
        
        # Draw border
        pygame.draw.rect(self.screen, (100, 100, 100), (grid_x - 2, grid_y - 2, grid_size + 4, grid_size + 4), 2)
        self.screen.blit(scaled, (grid_x, grid_y))
        
    def _render_pov_grid_worldspace(self, car: Car, track: Track, car_x: int, car_y: int):
        """Render occupancy grid as circles in POV mode (worldspace style).
        
        Grid cells are rendered as circles radiating out from the car.
        """
        from racing_sim.config.config import GridConfig
        grid_config = GridConfig()
        
        grid = compute_grid(car.position, car.angle, track, grid_config)
        positions = compute_grid_positions(car.position, car.angle, grid_config)
        
        # Scale factor for converting world distances to screen pixels
        scale = self.config.pov_worldspace_scale
        
        for row, col, world_pos in positions:
            on_track = grid[row, col] > 0
            color = self.config.grid_on_color if on_track else self.config.grid_off_color
            
            # Calculate relative position from car
            rel_x = world_pos.x - car.position.x
            rel_y = world_pos.y - car.position.y
            
            # Rotate to car's frame (forward = up on screen)
            # Rotation by (-car.angle + π/2) transforms car-forward to screen-up
            cos_a = math.cos(-car.angle + math.pi / 2)
            sin_a = math.sin(-car.angle + math.pi / 2)
            
            # Transform to screen coordinates
            # Standard 2D rotation, then negate Y since pygame Y increases downward
            screen_rel_x = (rel_x * cos_a - rel_y * sin_a) * scale
            screen_rel_y = -(rel_x * sin_a + rel_y * cos_a) * scale
            
            screen_x = int(car_x + screen_rel_x)
            screen_y = int(car_y + screen_rel_y)
            
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), 4)

    def _render_pov_hud(self, info: dict, car: Car):
        """Render minimal HUD for POV mode."""
        y_offset = 40
        texts = [
            f"Speed: {car.speed:.1f}",
            f"Checkpoints: {info.get('checkpoints_passed', 0)}",
            f"Step: {info.get('step', 0)}",
        ]
        
        for text in texts:
            surface = self._font_hud.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 22

    def _render_hud(
        self,
        info: dict,
        car: Car,
        ai_info: Optional[dict] = None,
        ai_car: Optional[Car] = None,
    ):
        """Render heads-up display with stats."""
        y_offset = 10

        # Human stats (left side)
        texts = [
            f"You (Blue): {info.get('episode_reward', 0):.1f}",
            f"  Speed: {car.speed:.1f}",
            f"  Checkpoint: {info.get('checkpoint', 0)}",
            f"  Step: {info.get('step', 0)}",
        ]

        for text in texts:
            surface = self._font_hud.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20

        # AI stats (if provided)
        if ai_info is not None and ai_car is not None:
            y_offset = 10
            ai_texts = [
                f"AI (Red): {ai_info.get('episode_reward', 0):.1f}",
                f"  Speed: {ai_car.speed:.1f}",
                f"  Checkpoint: {ai_info.get('checkpoint', 0)}",
            ]

            for text in ai_texts:
                surface = self._font_hud.render(text, True, (255, 255, 255))
                self.screen.blit(surface, (self.config.screen_width - 200, y_offset))
                y_offset += 20

    def _render_action_bars(self, actions: dict):
        """Render steering/throttle visualization bars.

        Args:
            actions: Dict with "human" and/or "ai" keys, each mapping to
                     (steering_float, throttle_float).
        """
        has_human = "human" in actions
        has_ai = "ai" in actions
        bottom_y = self.config.screen_height - 20

        if has_human and has_ai:
            # Race mode: two clusters, left=YOU, right=AI
            left_x = 60
            right_x = self.config.screen_width - 310
            self._draw_action_bar_cluster(
                *actions["human"], "YOU", left_x, bottom_y, (100, 150, 255)
            )
            self._draw_action_bar_cluster(
                *actions["ai"], "AI", right_x, bottom_y, (255, 100, 100)
            )
        elif has_human:
            center_x = self.config.screen_width // 2 - 155
            self._draw_action_bar_cluster(
                *actions["human"], None, center_x, bottom_y, None
            )
        elif has_ai:
            center_x = self.config.screen_width // 2 - 155
            self._draw_action_bar_cluster(
                *actions["ai"], None, center_x, bottom_y, None
            )

    def _draw_action_bar_cluster(
        self,
        steering: float,
        throttle: float,
        label: Optional[str],
        anchor_x: int,
        anchor_y: int,
        label_color: Optional[Tuple[int, int, int]],
    ):
        """Draw one set of steering + throttle bars.

        Layout (left to right from anchor_x):
          STR [====|====>    ]  THR [######     ]

        Args:
            steering: -1.0 (left) to 1.0 (right).
            throttle: 0.0 to 1.0.
            label: Optional header text (e.g. "YOU", "AI").
            anchor_x: Left edge x of the cluster.
            anchor_y: Bottom edge y of the cluster.
            label_color: Color for label text, or None.
        """
        bar_h = 14
        str_w = 150
        thr_w = 80
        gap = 16
        label_gap = 6
        bg_color = (50, 50, 50)
        border_color = (100, 100, 100)
        amber = (230, 180, 50)
        green = (80, 200, 80)
        text_color = (180, 180, 180)

        # Vertical baseline: bars sit above anchor_y
        bar_y = anchor_y - bar_h

        # Optional label header above the bars
        if label and label_color:
            label_surf = self._font_small.render(label, True, label_color)
            # Center label over the full cluster width
            cluster_w = 32 + str_w + gap + 32 + thr_w
            self.screen.blit(
                label_surf,
                (anchor_x + cluster_w // 2 - label_surf.get_width() // 2,
                 bar_y - label_surf.get_height() - label_gap),
            )

        x = anchor_x

        # -- Steering label --
        str_label = self._font_small.render("STR", True, text_color)
        self.screen.blit(str_label, (x, bar_y))
        x += 32

        # -- Steering bar --
        pygame.draw.rect(self.screen, bg_color, (x, bar_y, str_w, bar_h))
        pygame.draw.rect(self.screen, border_color, (x, bar_y, str_w, bar_h), 1)
        # Center tick
        center_x = x + str_w // 2
        pygame.draw.line(
            self.screen, border_color,
            (center_x, bar_y), (center_x, bar_y + bar_h), 1
        )
        # Directional fill from center
        clamped_steer = max(-1.0, min(1.0, steering))
        if clamped_steer > 0:
            fill_w = int(clamped_steer * (str_w // 2))
            pygame.draw.rect(
                self.screen, amber,
                (center_x, bar_y + 1, fill_w, bar_h - 2),
            )
        elif clamped_steer < 0:
            fill_w = int(-clamped_steer * (str_w // 2))
            pygame.draw.rect(
                self.screen, amber,
                (center_x - fill_w, bar_y + 1, fill_w, bar_h - 2),
            )
        x += str_w + gap

        # -- Throttle label --
        thr_label = self._font_small.render("THR", True, text_color)
        self.screen.blit(thr_label, (x, bar_y))
        x += 32

        # -- Throttle bar --
        pygame.draw.rect(self.screen, bg_color, (x, bar_y, thr_w, bar_h))
        pygame.draw.rect(self.screen, border_color, (x, bar_y, thr_w, bar_h), 1)
        clamped_thr = max(0.0, min(1.0, throttle))
        fill_w = int(clamped_thr * thr_w)
        if fill_w > 0:
            pygame.draw.rect(
                self.screen, green,
                (x, bar_y + 1, fill_w, bar_h - 2),
            )

    def _render_help(self):
        """Render help text at bottom of screen."""
        help_text = (
            "WASD/Arrows: Drive | R: Reset | T: Random Start | L: Lidar | "
            "G: Grid | V: Grid Debug | C: POV | ESC: Quit"
        )
        text = self._font_small.render(help_text, True, (150, 150, 150))
        extra_clearance = 45 if self.last_actions else 0
        help_y = self.config.screen_height - 25 - extra_clearance
        help_y = max(10, help_y)
        self.screen.blit(text, (10, help_y))

    def _to_screen(self, point: Vec2d) -> Tuple[int, int]:
        """Convert Pymunk coordinates to screen coordinates."""
        # Pymunk uses bottom-left origin, PyGame uses top-left
        # Flip y-axis
        return (
            int(point.x + self._view_offset.x),
            int(self.config.screen_height - (point.y + self._view_offset.y)),
        )

    def _is_custom_track(self, track: Track) -> bool:
        """Return True when rendering a node-based custom track."""
        return track.__class__.__name__ == "NodeTrack"

    def _compute_track_bounds(
        self, track: Track
    ) -> Optional[Tuple[float, float, float, float]]:
        """Compute axis-aligned bounds for the track walls."""
        segments = track.get_all_wall_segments()
        if not segments:
            return None

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for p1, p2 in segments:
            min_x = min(min_x, p1.x, p2.x)
            min_y = min(min_y, p1.y, p2.y)
            max_x = max(max_x, p1.x, p2.x)
            max_y = max(max_y, p1.y, p2.y)

        return min_x, min_y, max_x, max_y

    def _refresh_view(self, track: Track) -> None:
        """Recenter custom tracks when needed."""
        track_id = id(track)
        if (
            self._last_track_id == track_id
            and not self._needs_view_update
        ):
            return

        if not self._is_custom_track(track):
            self._view_offset = Vec2d(0.0, 0.0)
            self._needs_view_update = False
            self._last_track_id = track_id
            return

        bounds = self._compute_track_bounds(track)
        if bounds is None:
            self._view_offset = Vec2d(0.0, 0.0)
        else:
            min_x, min_y, max_x, max_y = bounds
            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            self._view_offset = Vec2d(
                self.config.screen_width / 2.0 - center_x,
                self.config.screen_height / 2.0 - center_y,
            )

        self._needs_view_update = False
        self._last_track_id = track_id

    def close(self):
        """Clean up PyGame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self.screen = None
            self.clock = None
            self._font_small = None
            self._font_hud = None
            self.last_actions = None

    def handle_events(self) -> bool:
        """
        Handle PyGame events.

        Returns:
            False if quit event received, True otherwise
        """
        self._init_pygame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE and self.render_mode == "human":
                new_w = max(1, event.w)
                new_h = max(1, event.h)
                self.config.screen_width = new_w
                self.config.screen_height = new_h
                self.screen = pygame.display.set_mode(
                    (new_w, new_h), pygame.DOUBLEBUF | pygame.RESIZABLE
                )
                self._needs_view_update = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self._reset_requested = True
                if event.key == pygame.K_t:
                    self._random_start_toggle_requested = True
                if event.key == pygame.K_l:
                    self._show_lidar = not self._show_lidar
                if event.key == pygame.K_g:
                    self._show_grid = not self._show_grid
                if event.key == pygame.K_v:
                    self._grid_debug_mode = not self._grid_debug_mode
                if event.key == pygame.K_c:
                    self._pov_mode = not self._pov_mode
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

    def was_random_start_toggle_requested(self) -> bool:
        """
        Check if random start toggle was requested and clear the flag.

        Returns:
            True if 't' key was pressed since last check
        """
        was_requested = self._random_start_toggle_requested
        self._random_start_toggle_requested = False
        return was_requested
