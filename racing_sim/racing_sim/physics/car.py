"""Car physics with lateral friction using Pymunk."""

import pymunk
from pymunk import Vec2d
import math
from racing_sim.config.config import CarConfig


# Collision categories
CATEGORY_CAR = 0b0001
CATEGORY_WALL = 0b0010


class Car:
    """Physics-based car with lateral friction for realistic tire grip."""

    def __init__(
        self,
        space: pymunk.Space,
        config: CarConfig,
        position: tuple,
        angle: float = 0.0,
        setup_collision_handler: bool = True,
    ):
        """
        Initialize the car.

        Args:
            space: Pymunk physics space
            config: Car configuration
            position: Initial (x, y) position
            angle: Initial angle in radians
            setup_collision_handler: If True, set up collision detection. Set False for ghost cars.
        """
        self.config = config
        self.space = space

        # Create car body
        self.body = pymunk.Body(
            config.mass,
            pymunk.moment_for_box(config.mass, (config.width, config.height)),
        )
        self.body.position = Vec2d(*position)
        self.body.angle = angle

        # Create car shape
        self.shape = pymunk.Poly.create_box(self.body, (config.width, config.height))
        self.shape.friction = 0.5
        self.shape.elasticity = 0.1
        self.shape.collision_type = 1  # Car collision type
        self.shape.filter = pymunk.ShapeFilter(
            categories=CATEGORY_CAR, mask=CATEGORY_WALL
        )

        space.add(self.body, self.shape)

        # Track collision state
        self.collided = False
        self.touching_wall = False
        # When True, collision is detected but physics response is skipped (car doesn't stop)
        self.sensor_only = False
        self._last_steering = 0.0
        self._last_throttle = 0.0
        self._smoothed_throttle = 0.0
        if setup_collision_handler:
            self._setup_collision_handler()

    def _setup_collision_handler(self):
        """Set up collision detection between car and walls."""
        self.space.on_collision(
            collision_type_a=1,  # Car
            collision_type_b=2,  # Wall
            begin=self._on_collision_begin,
            pre_solve=self._on_collision_pre_solve,
            data={"car": self},
        )

    def _on_collision_begin(self, arbiter, space, data):
        """Called when car collides with a wall."""
        car = data["car"]
        # Verify this car's shape is actually involved in this collision
        # (needed when multiple cars share the same space)
        if arbiter.shapes[0] == car.shape or arbiter.shapes[1] == car.shape:
            car.collided = True
            # Skip physics response if sensor_only mode (car passes through but detects collision)
            arbiter.process_collision = not car.sensor_only
            return not car.sensor_only
        # Not this car's collision - don't interfere
        return True

    def _on_collision_pre_solve(self, arbiter, space, data):
        """Called each physics step while car is in contact with a wall."""
        car = data["car"]
        if arbiter.shapes[0] == car.shape or arbiter.shapes[1] == car.shape:
            car.touching_wall = True
            return not car.sensor_only
        return True

    def reset(self, position: tuple, angle: float = 0.0):
        """Reset car to initial state."""
        self.body.position = Vec2d(*position)
        self.body.angle = angle
        self.body.velocity = Vec2d(0, 0)
        self.body.angular_velocity = 0
        self.collided = False
        self.touching_wall = False
        self._last_steering = 0.0
        self._last_throttle = 0.0
        self._smoothed_throttle = 0.0

    def update_friction(self, dt: float = 1.0 / 60.0):
        """
        Apply lateral friction to simulate tire grip.

        This prevents the car from sliding sideways while allowing
        forward/backward movement.
        """
        # Get forward and lateral directions
        forward = self.body.rotation_vector
        lateral = Vec2d(-forward.y, forward.x)

        velocity = self.body.velocity

        # Decompose velocity into forward and lateral components
        lateral_vel = lateral.dot(velocity) * lateral
        forward_vel = forward.dot(velocity) * forward

        # Apply high friction to lateral velocity (tire grip)
        lateral_impulse = (
            -lateral_vel * self.config.lateral_friction * self.body.mass * dt
        )
        self.body.apply_impulse_at_world_point(lateral_impulse, self.body.position)

        # Apply rolling friction to forward velocity
        forward_impulse = (
            -forward_vel * self.config.rolling_friction * self.body.mass * dt
        )
        self.body.apply_impulse_at_world_point(forward_impulse, self.body.position)

        # Apply air drag to make high-speed acceleration taper off
        if self.config.air_drag > 0.0:
            speed = velocity.length
            if speed > 0.0:
                drag_mag = self.config.air_drag * speed * speed * self.body.mass
                drag_force = -velocity.normalized() * drag_mag
                self.body.apply_force_at_world_point(drag_force, self.body.position)

        # Apply cornering drag when steering to bleed speed in turns
        steer_mag = abs(self._last_steering)
        if steer_mag > 0.0:
            velocity = self.body.velocity
            speed = velocity.length
            if speed > 0.0:
                drag_mag = (
                    self.config.turning_drag * steer_mag * speed * self.body.mass * dt
                )
                drag_mag = min(drag_mag, speed * self.body.mass)
                drag_impulse = -velocity.normalized() * drag_mag
                self.body.apply_impulse_at_world_point(drag_impulse, self.body.position)

        # Apply angular damping for stability
        self.body.angular_velocity *= 1.0 - self.config.angular_damping * 0.1

        # Remove any lateral velocity to prevent sideways sliding.
        velocity = self.body.velocity
        forward = self.body.rotation_vector
        forward_speed = forward.dot(velocity)
        self.body.velocity = forward * forward_speed

    def apply_control(self, steering: float, throttle: float):
        """
        Apply steering and throttle control to the car.

        Args:
            steering: Steering input from -1 (left) to 1 (right)
            throttle: Throttle input from 0 to 1
        """
        # Clamp inputs
        steering = max(-1.0, min(1.0, steering))
        throttle = max(0.0, min(1.0, throttle))
        self._last_steering = steering
        self._last_throttle = throttle
        response = max(0.0, min(1.0, self.config.throttle_response))
        if response >= 1.0:
            self._smoothed_throttle = throttle
        else:
            self._smoothed_throttle += (throttle - self._smoothed_throttle) * response

        # Apply forward force (throttle)
        forward = self.body.rotation_vector
        speed = self.body.velocity.length
        max_speed = max(self.config.max_speed, 1e-6)
        speed_ratio = min(1.0, speed / max_speed)
        accel_multiplier = 1.0 + self.config.accel_boost * (1.0 - speed_ratio)
        force = (
            forward
            * self._smoothed_throttle
            * self.config.engine_power
            * accel_multiplier
        )
        self.body.apply_force_at_world_point(force, self.body.position)

        # Apply steering torque (speed-dependent for realism)
        steering_ref = max(self.config.steering_speed_ref, 1e-6)
        speed_ratio = min(1.0, speed / steering_ref)
        speed_factor = self.config.steering_min_factor + (
            1.0 - self.config.steering_min_factor
        ) * (1.0 - speed_ratio)
        torque = (
            -steering * self.config.steering_power * speed_factor
        )  # Negate to fix inversion
        self.body.torque = torque

        # Limit max speed
        if speed > self.config.max_speed:
            self.body.velocity = self.body.velocity.normalized() * self.config.max_speed

    @property
    def position(self) -> Vec2d:
        """Get car position."""
        return self.body.position

    @property
    def angle(self) -> float:
        """Get car angle in radians."""
        return self.body.angle

    @property
    def velocity(self) -> Vec2d:
        """Get car velocity."""
        return self.body.velocity

    @property
    def speed(self) -> float:
        """Get car speed (velocity magnitude)."""
        return self.body.velocity.length

    @property
    def forward_speed(self) -> float:
        """Get forward component of velocity."""
        forward = self.body.rotation_vector
        return forward.dot(self.body.velocity)

    def get_corners(self) -> list:
        """Get the four corners of the car in world coordinates."""
        vertices = self.shape.get_vertices()
        return [self.body.local_to_world(v) for v in vertices]
