"""Track geometry with walls and checkpoints."""

import pymunk
from pymunk import Vec2d
import math
import numpy as np
from typing import List, Tuple, Optional
from racing_sim.config.config import TrackConfig
from racing_sim.physics.car import CATEGORY_WALL


class Track:
    """Oval track with inner and outer walls."""

    def __init__(self, space: pymunk.Space, config: TrackConfig):
        """
        Initialize the track.

        Args:
            space: Pymunk physics space
            config: Track configuration
        """
        self.config = config
        self.space = space

        # Track parameters
        self.center = Vec2d(config.center_x, config.center_y)
        self.base_outer_radius_x = config.outer_radius_x
        self.base_outer_radius_y = config.outer_radius_y
        self.base_inner_radius_x = config.outer_radius_x - config.width
        self.base_inner_radius_y = config.outer_radius_y - config.width
        self.waviness = config.waviness
        self.waves = config.waves
        self.wave_phase = config.wave_phase

        # Wall segments
        self.outer_walls: List[pymunk.Segment] = []
        self.inner_walls: List[pymunk.Segment] = []

        # Checkpoints for progress tracking
        self.checkpoints: List[Tuple[Vec2d, Vec2d]] = []

        # Start position
        outer_rx, _, inner_rx, _ = self._radii_at_angle(0.0)
        self.start_position = Vec2d(
            self.center.x + (outer_rx + inner_rx) / 2,
            self.center.y,
        )
        self.start_angle = math.pi / 2  # Facing up (counter-clockwise)

        self._create_walls()
        self._create_checkpoints()
        self._create_bitmap()

    def _radius_scale(self, angle: float) -> float:
        if self.waviness <= 0.0 or self.waves <= 0:
            return 1.0
        scale = 1.0 + self.waviness * math.sin(self.waves * angle + self.wave_phase)
        return max(0.2, scale)

    def _radii_at_angle(self, angle: float) -> Tuple[float, float, float, float]:
        scale = self._radius_scale(angle)
        outer_rx = self.base_outer_radius_x * scale
        outer_ry = self.base_outer_radius_y * scale
        inner_rx = self.base_inner_radius_x * scale
        inner_ry = self.base_inner_radius_y * scale
        return outer_rx, outer_ry, inner_rx, inner_ry

    def _create_walls(self, num_segments: int = 64):
        """Create oval track walls using line segments."""
        # Static body for walls
        static_body = self.space.static_body

        # Create outer wall segments
        for i in range(num_segments):
            angle1 = (i / num_segments) * 2 * math.pi
            angle2 = ((i + 1) / num_segments) * 2 * math.pi

            outer_rx1, outer_ry1, inner_rx1, inner_ry1 = self._radii_at_angle(angle1)
            outer_rx2, outer_ry2, inner_rx2, inner_ry2 = self._radii_at_angle(angle2)

            # Outer wall points
            p1 = Vec2d(
                self.center.x + outer_rx1 * math.cos(angle1),
                self.center.y + outer_ry1 * math.sin(angle1),
            )
            p2 = Vec2d(
                self.center.x + outer_rx2 * math.cos(angle2),
                self.center.y + outer_ry2 * math.sin(angle2),
            )

            segment = pymunk.Segment(static_body, p1, p2, 2.0)
            segment.friction = 0.8
            segment.elasticity = 0.5
            segment.collision_type = 2  # Wall collision type
            segment.filter = pymunk.ShapeFilter(categories=CATEGORY_WALL)
            self.space.add(segment)
            self.outer_walls.append(segment)

            # Inner wall points
            p1_inner = Vec2d(
                self.center.x + inner_rx1 * math.cos(angle1),
                self.center.y + inner_ry1 * math.sin(angle1),
            )
            p2_inner = Vec2d(
                self.center.x + inner_rx2 * math.cos(angle2),
                self.center.y + inner_ry2 * math.sin(angle2),
            )

            segment_inner = pymunk.Segment(static_body, p1_inner, p2_inner, 2.0)
            segment_inner.friction = 0.8
            segment_inner.elasticity = 0.5
            segment_inner.collision_type = 2  # Wall collision type
            segment_inner.filter = pymunk.ShapeFilter(categories=CATEGORY_WALL)
            self.space.add(segment_inner)
            self.inner_walls.append(segment_inner)

    def _create_checkpoints(self, num_checkpoints: int = 64):
        """Create checkpoints around the track for progress tracking."""
        for i in range(num_checkpoints):
            angle = (i / num_checkpoints) * 2 * math.pi

            outer_rx, outer_ry, inner_rx, inner_ry = self._radii_at_angle(angle)

            # Checkpoint line from inner to outer wall
            inner_point = Vec2d(
                self.center.x + inner_rx * math.cos(angle),
                self.center.y + inner_ry * math.sin(angle),
            )
            outer_point = Vec2d(
                self.center.x + outer_rx * math.cos(angle),
                self.center.y + outer_ry * math.sin(angle),
            )

            self.checkpoints.append((inner_point, outer_point))

    def get_checkpoint_index(self, position: Vec2d) -> int:
        """
        Get the index of the checkpoint closest to the given position.

        Args:
            position: Position to check

        Returns:
            Index of the closest checkpoint
        """
        # Calculate angle from center to position
        delta = position - self.center
        angle = math.atan2(delta.y, delta.x)
        if angle < 0:
            angle += 2 * math.pi

        # Convert to checkpoint index
        num_checkpoints = len(self.checkpoints)
        index = int((angle / (2 * math.pi)) * num_checkpoints) % num_checkpoints
        return index

    def get_progress(self, position: Vec2d, last_checkpoint: int) -> Tuple[int, bool]:
        """
        Calculate progress and check if a new checkpoint was reached.

        Args:
            position: Current position
            last_checkpoint: Previously reached checkpoint index

        Returns:
            Tuple of (current_checkpoint_index, checkpoint_crossed)
        """
        current_checkpoint = self.get_checkpoint_index(position)

        # Check if we've crossed into a new checkpoint
        # Handle wraparound (from last checkpoint back to first)
        num_checkpoints = len(self.checkpoints)
        expected_next = (last_checkpoint + 1) % num_checkpoints

        crossed = (current_checkpoint == expected_next)

        return current_checkpoint, crossed

    def is_on_track(self, position: Vec2d) -> bool:
        """
        Check if a position is on the track (between inner and outer walls).

        Args:
            position: Position to check

        Returns:
            True if position is on track
        """
        delta = position - self.center

        angle = math.atan2(delta.y, delta.x)
        if angle < 0:
            angle += 2 * math.pi
        outer_rx, outer_ry, inner_rx, inner_ry = self._radii_at_angle(angle)

        # Normalized distance based on angle-adjusted ellipse equation
        normalized_outer = (delta.x / outer_rx) ** 2 + (delta.y / outer_ry) ** 2
        normalized_inner = (delta.x / inner_rx) ** 2 + (delta.y / inner_ry) ** 2

        return normalized_inner >= 1.0 and normalized_outer <= 1.0

    def _create_bitmap(self, resolution: float = 1.0):
        """Precompute track occupancy bitmap for fast lookups.

        Args:
            resolution: Pixels per world unit. Higher = more accurate but more memory.
        """
        # Compute bounding box with padding
        max_radius = max(self.base_outer_radius_x, self.base_outer_radius_y)
        if self.waviness > 0:
            max_radius *= (1.0 + self.waviness)  # Account for wavy expansion

        padding = 50  # Extra padding around track
        self._bitmap_min_x = self.center.x - max_radius - padding
        self._bitmap_max_x = self.center.x + max_radius + padding
        self._bitmap_min_y = self.center.y - max_radius - padding
        self._bitmap_max_y = self.center.y + max_radius + padding
        self._bitmap_resolution = resolution

        # Compute bitmap dimensions
        width = int((self._bitmap_max_x - self._bitmap_min_x) * resolution)
        height = int((self._bitmap_max_y - self._bitmap_min_y) * resolution)

        # Create bitmap using vectorized is_on_track logic
        # Generate grid of world coordinates
        x_coords = np.linspace(self._bitmap_min_x, self._bitmap_max_x, width)
        y_coords = np.linspace(self._bitmap_min_y, self._bitmap_max_y, height)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Vectorized is_on_track calculation
        dx = xx - self.center.x
        dy = yy - self.center.y

        angles = np.arctan2(dy, dx)
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)

        # Compute radii at each angle (vectorized)
        if self.waviness > 0 and self.waves > 0:
            scale = 1.0 + self.waviness * np.sin(self.waves * angles + self.wave_phase)
            scale = np.maximum(0.2, scale)
        else:
            scale = 1.0

        outer_rx = self.base_outer_radius_x * scale
        outer_ry = self.base_outer_radius_y * scale
        inner_rx = self.base_inner_radius_x * scale
        inner_ry = self.base_inner_radius_y * scale

        # Normalized distances
        normalized_outer = (dx / outer_rx) ** 2 + (dy / outer_ry) ** 2
        normalized_inner = (dx / inner_rx) ** 2 + (dy / inner_ry) ** 2

        # On track if between inner and outer ellipse
        self._track_bitmap = ((normalized_inner >= 1.0) & (normalized_outer <= 1.0)).astype(np.uint8)

    def is_on_track_fast(self, position: Vec2d) -> bool:
        """Fast track check using precomputed bitmap.

        Args:
            position: Position to check

        Returns:
            True if position is on track
        """
        # Convert world position to bitmap coordinates
        bx = int((position.x - self._bitmap_min_x) * self._bitmap_resolution)
        by = int((position.y - self._bitmap_min_y) * self._bitmap_resolution)

        # Bounds check
        if bx < 0 or bx >= self._track_bitmap.shape[1]:
            return False
        if by < 0 or by >= self._track_bitmap.shape[0]:
            return False

        return self._track_bitmap[by, bx] > 0

    def is_on_track_batch(self, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
        """Batch track check using precomputed bitmap.

        Args:
            world_x: Array of x coordinates
            world_y: Array of y coordinates

        Returns:
            Boolean array, True where positions are on track
        """
        # Convert world positions to bitmap coordinates
        bx = ((world_x - self._bitmap_min_x) * self._bitmap_resolution).astype(np.int32)
        by = ((world_y - self._bitmap_min_y) * self._bitmap_resolution).astype(np.int32)

        # Bounds check
        h, w = self._track_bitmap.shape
        valid = (bx >= 0) & (bx < w) & (by >= 0) & (by < h)

        # Clip to valid range for lookup (invalid positions will be masked out)
        bx_safe = np.clip(bx, 0, w - 1)
        by_safe = np.clip(by, 0, h - 1)

        # Lookup and mask invalid positions
        result = self._track_bitmap[by_safe, bx_safe] > 0
        result = result & valid

        return result

    def get_all_wall_segments(self) -> List[Tuple[Vec2d, Vec2d]]:
        """Get all wall segments as point pairs for rendering."""
        segments = []
        for wall in self.outer_walls + self.inner_walls:
            segments.append((wall.a, wall.b))
        return segments

    def get_spawn_position(self, checkpoint_index: int) -> Tuple[Vec2d, float]:
        """
        Get a valid spawn position and angle at a given checkpoint.

        The car is placed at the center of the track width (between inner
        and outer walls) and faces tangent to the track in the direction
        of progress (counter-clockwise).

        Args:
            checkpoint_index: Index of the checkpoint (0 to num_checkpoints-1)

        Returns:
            Tuple of (position, angle) where angle is in radians
        """
        num_checkpoints = len(self.checkpoints)
        checkpoint_index = checkpoint_index % num_checkpoints

        # Get the angle for this checkpoint
        angle = (checkpoint_index / num_checkpoints) * 2 * math.pi

        # Get radii at this angle
        outer_rx, outer_ry, inner_rx, inner_ry = self._radii_at_angle(angle)

        # Calculate center of track at this angle (midpoint between inner and outer)
        mid_rx = (outer_rx + inner_rx) / 2
        mid_ry = (outer_ry + inner_ry) / 2

        position = Vec2d(
            self.center.x + mid_rx * math.cos(angle),
            self.center.y + mid_ry * math.sin(angle),
        )

        # Calculate tangent angle (perpendicular to radial, counter-clockwise)
        # The radial points outward at 'angle', so tangent is angle + pi/2
        tangent_angle = angle + math.pi / 2

        return position, tangent_angle

    @property
    def num_checkpoints(self) -> int:
        """Return the number of checkpoints."""
        return len(self.checkpoints)
