"""Lidar sensor using Pymunk raycasting."""

import pymunk
from pymunk import Vec2d
import math
import numpy as np
from typing import List, Tuple, Optional
from racing_sim.config.config import LidarConfig
from racing_sim.physics.car import Car, CATEGORY_CAR


class Lidar:
    """Raycast-based lidar sensor for obstacle detection."""

    def __init__(self, space: pymunk.Space, config: LidarConfig):
        """
        Initialize the lidar sensor.

        Args:
            space: Pymunk physics space for raycasting
            config: Lidar configuration
        """
        self.space = space
        self.config = config

        # Convert angles to radians
        self.ray_angles_rad = [math.radians(a) for a in config.ray_angles]

        # Shape filter to ignore car's own shape
        self.shape_filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ CATEGORY_CAR)

        # Store last scan results for visualization
        self._last_rays: List[Tuple[Vec2d, Vec2d, float]] = []

    def scan(self, car: Car) -> np.ndarray:
        """
        Perform a lidar scan from the car's position.

        Args:
            car: Car to scan from

        Returns:
            Numpy array of normalized distances (0=hit at start, 1=no hit or max distance)
        """
        position = car.position
        car_angle = car.angle

        distances = np.ones(len(self.ray_angles_rad), dtype=np.float32)
        self._last_rays = []

        for i, ray_angle in enumerate(self.ray_angles_rad):
            # Calculate ray direction in world coordinates
            world_angle = car_angle + ray_angle
            direction = Vec2d(math.cos(world_angle), math.sin(world_angle))

            # Calculate ray end point
            end = position + direction * self.config.max_distance

            # Perform raycast
            query = self.space.segment_query_first(
                position,
                end,
                radius=1.0,
                shape_filter=self.shape_filter
            )

            if query is not None:
                # alpha is normalized distance (0-1) along the ray
                distances[i] = query.alpha
                hit_point = position + direction * (self.config.max_distance * query.alpha)
            else:
                # No hit, full distance
                distances[i] = 1.0
                hit_point = end

            # Store for visualization
            self._last_rays.append((position, hit_point, distances[i]))

        return distances

    def get_debug_rays(self) -> List[Tuple[Vec2d, Vec2d, float]]:
        """
        Get the last scan results for visualization.

        Returns:
            List of (start_point, end_point, normalized_distance) tuples
        """
        return self._last_rays.copy()

    def get_observation_size(self) -> int:
        """Get the size of the observation vector."""
        return len(self.ray_angles_rad)
