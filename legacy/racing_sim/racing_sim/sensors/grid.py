"""Occupancy grid sensor for CNN-based observations.

Uses perspective (homographic) projection to simulate a camera mounted on
the car looking down at the ground. This creates a trapezoidal sampling
pattern: narrow at far distances, wide at near distances.
"""

import math
import numpy as np
from pymunk import Vec2d
from racing_sim.config.config import GridConfig
from racing_sim.physics.track import Track


def compute_grid(car_pos: Vec2d, car_angle: float, track: Track,
                 config: GridConfig) -> np.ndarray:
    """Compute an occupancy grid with perspective projection.

    The grid simulates a downward-looking camera mounted on the car.
    Row 0 is farthest forward, row N-1 is closest to the car.
    Near rows sample from a wider lateral span than far rows (trapezoid shape).

    Uses vectorized numpy operations for world position computation,
    with only the is_on_track() calls requiring iteration.

    Args:
        car_pos: Car position in world coordinates.
        car_angle: Car heading in radians.
        track: Track object with is_on_track method.
        config: Grid configuration with camera parameters.

    Returns:
        uint8 array of shape (grid_size, grid_size), values 0 or 255.
    """
    size = config.grid_size
    near = config.near_distance
    far = config.far_distance
    fov_h_rad = math.radians(config.fov_horizontal)
    tan_half_fov = math.tan(fov_h_rad / 2.0)

    # Vectorized direction computation
    cos_a = math.cos(car_angle)
    sin_a = math.sin(car_angle)
    forward = np.array([cos_a, sin_a])
    left = np.array([-sin_a, cos_a])

    # Create meshgrid of row/col indices
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')

    # Vectorized t and distance computation
    # t goes from 1 (far) at row 0 to 0 (near) at row (size-1)
    t = (size - 1 - rows) / (size - 1) if size > 1 else np.full_like(rows, 0.5, dtype=float)
    distances = near + t * (far - near)

    # Vectorized half-width computation
    half_widths = distances * tan_half_fov

    # Vectorized lateral_t and lateral_offset computation
    if size > 1:
        lateral_t = (cols - (size - 1) / 2.0) / ((size - 1) / 2.0)
    else:
        lateral_t = np.zeros_like(cols, dtype=float)
    lateral_offsets = lateral_t * half_widths

    # Vectorized world position computation
    world_x = car_pos.x + forward[0] * distances + left[0] * lateral_offsets
    world_y = car_pos.y + forward[1] * distances + left[1] * lateral_offsets

    # Use batch lookup if track has bitmap (much faster)
    if hasattr(track, 'is_on_track_batch'):
        on_track = track.is_on_track_batch(world_x, world_y)
        grid = (on_track * 255).astype(np.uint8)
    else:
        # Fallback to loop for tracks without bitmap
        grid = np.zeros((size, size), dtype=np.uint8)
        for r in range(size):
            for c in range(size):
                if track.is_on_track(Vec2d(world_x[r, c], world_y[r, c])):
                    grid[r, c] = 255

    return grid


def compute_grid_positions(car_pos: Vec2d, car_angle: float,
                           config: GridConfig):
    """Compute world positions for each grid cell (for renderer visualization).

    Uses the same perspective projection as compute_grid.

    Args:
        car_pos: Car position in world coordinates.
        car_angle: Car heading in radians.
        config: Grid configuration with camera parameters.

    Returns:
        List of (row, col, world_pos) tuples for each cell.
    """
    size = config.grid_size
    near = config.near_distance
    far = config.far_distance
    fov_h_rad = math.radians(config.fov_horizontal)
    tan_half_fov = math.tan(fov_h_rad / 2.0)

    forward = Vec2d(math.cos(car_angle), math.sin(car_angle))
    left = Vec2d(-forward.y, forward.x)

    positions = []

    for row in range(size):
        # t goes from 1 (far) to 0 (near)
        t = (size - 1 - row) / (size - 1) if size > 1 else 0.5
        distance = near + t * (far - near)

        # Lateral half-width at this distance
        half_width = distance * tan_half_fov

        for col in range(size):
            lateral_t = (col - (size - 1) / 2.0) / ((size - 1) / 2.0) if size > 1 else 0.0
            lateral_offset = lateral_t * half_width

            world_pos = car_pos + forward * distance + left * lateral_offset
            positions.append((row, col, world_pos))

    return positions
