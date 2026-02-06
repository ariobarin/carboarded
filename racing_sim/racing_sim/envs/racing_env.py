"""Main Gymnasium environment for the racing simulation."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
from typing import Optional, Tuple, Dict, Any
from racing_sim.config.config import EnvConfig
from racing_sim.physics.car import Car
from racing_sim.physics.track import Track
from racing_sim.editor.node_track import NodeTrack
from racing_sim.sensors.lidar import Lidar
from racing_sim.rendering.renderer import Renderer
from racing_sim.sensors.grid import compute_grid
from racing_sim.utils.progress import progress_delta_cyclic
from racing_sim.utils.off_track import compute_off_track_state
from racing_sim.utils.reward import compute_slowdown_penalty


class RacingEnv(gym.Env):
    """
    2D Racing Environment.

    Observation Space:
        - "lidar" mode: Box(0, 1, shape=(num_rays,)) - normalized lidar distances
        - "grid" mode: Box(0, 255, shape=(grid_size, grid_size, 1)) - binary occupancy grid

    Action Space: Box([-1, 0], [1, 1]) - [steering, throttle]

    Reward: checkpoint bonus + progress shaping + speed bonus + collision penalty.
    See CLAUDE.md or configs/README.md for the full reward breakdown.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the racing environment.

        Args:
            config: Environment configuration (uses defaults if None)
            render_mode: "human" for display, "rgb_array" for numpy array, None to disable
        """
        super().__init__()

        self.config = config or EnvConfig.default()
        self.render_mode = render_mode

        # Observation space depends on obs_type
        if self.config.obs_type == "grid":
            gs = self.config.grid.grid_size
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(gs, gs, 1),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.config.lidar.num_rays,),
                dtype=np.float32,
            )

        # Action space: [steering, throttle]
        # steering: -1 (left) to 1 (right)
        # throttle: 0 to 1
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Physics space
        self.space: Optional[pymunk.Space] = None
        self.car: Optional[Car] = None
        self.track: Optional[Track] = None
        self.lidar: Optional[Lidar] = None
        self.renderer: Optional[Renderer] = None

        # Episode state
        self.current_step = 0
        self.last_checkpoint = 0
        self.checkpoints_passed = 0
        self.episode_reward = 0.0
        self.last_progress_angle = 0.0
        self.last_observation: Optional[np.ndarray] = None
        self.off_track_steps = 0
        self.wall_contact_steps = 0
        self.last_on_track = True

        # Cohort spawn: when set, reset() spawns at this checkpoint
        # instead of random. Used by CohortSpawnCallback for PPO.
        self.spawn_checkpoint: Optional[int] = None

        # Initialize components
        self._setup()

    def _setup(self):
        """Set up physics world and components."""
        # Create physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # Top-down view, no gravity

        # Create track (elliptical or custom node-based)
        if self.config.track.track_type == "custom" and self.config.track.custom is not None:
            custom = self.config.track.custom
            nodes = [(n.x, n.y, n.radius) for n in custom.nodes]
            self.track = NodeTrack(
                self.space,
                nodes=nodes,
                width=custom.width,
                num_checkpoints=custom.num_checkpoints,
                start_node_index=custom.start_node_index,
                start_offset=custom.start_offset,
            )
        else:
            self.track = Track(self.space, self.config.track)

        # Create car at start position
        self.car = Car(
            self.space,
            self.config.car,
            position=(self.track.start_position.x, self.track.start_position.y),
            angle=self.track.start_angle
        )
        if self.config.max_off_track_steps > 0:
            self.car.sensor_only = True

        # Create lidar sensor
        self.lidar = Lidar(self.space, self.config.lidar)

        # Create renderer if needed
        if self.render_mode is not None:
            self.renderer = Renderer(self.config.render, self.render_mode,
                                     grid_config=self.config.grid)

    def _apply_lateral_spawn_offset(self, start_pos: pymunk.Vec2d, checkpoint_idx: int) -> pymunk.Vec2d:
        """Apply a random lateral offset from the checkpoint centerline."""
        fraction = float(self.config.random_start_lateral_fraction)
        if fraction <= 0.0:
            return start_pos
        if not self.track.checkpoints:
            return start_pos

        idx = checkpoint_idx % len(self.track.checkpoints)
        inner, outer = self.track.checkpoints[idx]
        normal = outer - inner
        width = normal.length
        if width <= 1e-6:
            return start_pos

        normal = normal.normalized()
        car_half_width = self.config.car.height * 0.5
        margin = max(2.0, self.config.car.height * 0.05)
        safe_half_width = 0.5 * width - car_half_width - margin
        if safe_half_width <= 0.0:
            return start_pos

        fraction = max(0.0, min(1.0, fraction))
        max_offset = safe_half_width * fraction
        offset = self.np_random.uniform(-max_offset, max_offset)
        return pymunk.Vec2d(start_pos.x, start_pos.y) + normal * offset

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Determine starting position
        checkpoint_idx: Optional[int] = None
        if self.spawn_checkpoint is not None:
            # Cohort spawn: all envs use the same checkpoint per rollout
            checkpoint_idx = self.spawn_checkpoint
            start_pos, start_angle = self.track.get_spawn_position(checkpoint_idx)
        elif self.config.random_start:
            # Pick a random checkpoint and spawn there
            checkpoint_idx = self.np_random.integers(0, self.track.num_checkpoints)
            start_pos, start_angle = self.track.get_spawn_position(checkpoint_idx)
        else:
            # Use default start position
            start_pos = self.track.start_position
            start_angle = self.track.start_angle

        if checkpoint_idx is not None:
            start_pos = self._apply_lateral_spawn_offset(start_pos, int(checkpoint_idx))

        # Reset car to start position
        self.car.reset(
            position=(start_pos.x, start_pos.y),
            angle=start_angle
        )

        # Reset episode state
        self.current_step = 0
        self.last_checkpoint = self.track.get_checkpoint_index(self.car.position)
        self.checkpoints_passed = 0
        self.episode_reward = 0.0
        self.last_progress_angle = self.track.progress_coordinate(self.car.position)
        self.off_track_steps = 0
        self.wall_contact_steps = 0
        self.last_on_track = True

        # Get initial observation (always run lidar for reward computation)
        lidar_obs = self.lidar.scan(self.car)
        self.last_observation = lidar_obs

        if self.config.obs_type == "grid":
            grid = compute_grid(
                self.car.position, self.car.angle, self.track, self.config.grid,
            )
            observation = grid[:, :, np.newaxis]
        else:
            observation = lidar_obs

        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: [steering, throttle] array

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply control
        steering = float(action[0])
        throttle = float(action[1])
        self.car.apply_control(steering, throttle)

        # Update friction
        self.car.update_friction(self.config.physics_dt)

        # Clear per-step wall contact flag (pre_solve callback sets it during step)
        self.car.touching_wall = False

        # Step physics
        self.space.step(self.config.physics_dt)

        # Always run lidar (needed for reward computation)
        lidar_obs = self.lidar.scan(self.car)
        self.last_observation = lidar_obs

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Update step counter
        self.current_step += 1

        # Track continuous wall contact
        if self.car.touching_wall:
            self.wall_contact_steps += 1
        else:
            self.wall_contact_steps = 0

        # Check termination conditions
        if self.config.max_off_track_steps > 0:
            on_track = self.track.is_on_track(self.car.position)
            self.last_on_track = on_track
            self.off_track_steps, off_track_penalty, terminated = compute_off_track_state(
                on_track=on_track,
                collided=False,
                prev_off_track_steps=self.off_track_steps,
                off_track_penalty=self.config.off_track_penalty,
                max_off_track_steps=self.config.max_off_track_steps,
            )
            reward += off_track_penalty
        elif self.config.max_wall_contact_steps > 0:
            terminated = self.wall_contact_steps >= self.config.max_wall_contact_steps
        else:
            terminated = self.car.collided and self.config.terminate_on_collision
        truncated = self.current_step >= self.config.max_episode_steps

        # Build observation for the agent
        if self.config.obs_type == "grid":
            grid = compute_grid(
                self.car.position, self.car.angle, self.track, self.config.grid,
            )
            observation = grid[:, :, np.newaxis]
        else:
            observation = lidar_obs

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self) -> float:
        """Calculate reward for the current step."""
        reward = 0.0

        # Speed ratio (clamped) for shaping.
        speed_ratio = self.car.speed / max(self.config.car.max_speed, 1e-6)
        speed_ratio = min(max(speed_ratio, 0.0), 1.0)

        # Time penalty (scaled down as speed increases).
        reward += self.config.time_penalty * (1.0 - speed_ratio)

        # Speed bonus
        reward += self.config.speed_bonus_scale * speed_ratio

        # Checkpoint reward
        current_checkpoint, passed = self.track.get_progress(
            self.car.position,
            self.last_checkpoint,
            max_skip=self.config.max_checkpoint_skip,
        )

        if passed > 0:
            reward += self.config.checkpoint_reward * passed
            self.last_checkpoint = current_checkpoint
            self.checkpoints_passed += passed

        # Collision penalty applies only when collisions are terminal.
        if self.car.collided and self.config.terminate_on_collision:
            reward += self.config.collision_penalty

        # Wall contact penalty applies only for non-terminal collisions.
        if (
            not self.config.terminate_on_collision
            and self.config.wall_contact_penalty != 0.0
            and self.car.touching_wall
        ):
            reward += self.config.wall_contact_penalty

        # Progress reward (continuous around track)
        if self.config.progress_reward_scale != 0.0:
            current_progress = self.track.progress_coordinate(self.car.position)
            period = self.track.progress_period()
            delta = progress_delta_cyclic(self.last_progress_angle, current_progress, period)
            reward += self.config.progress_reward_scale * delta * self.track.progress_scale()
            self.last_progress_angle = current_progress

        # Slowdown penalty when close to walls and moving fast
        if self.last_observation is not None:
            speed_ratio = self.car.speed / max(self.config.car.max_speed, 1e-6)
            reward += compute_slowdown_penalty(
                self.last_observation,
                speed_ratio,
                self.config.slowdown_distance,
                self.config.slowdown_penalty_scale,
                ray_indices=(1, 2, 3),
            )

        return reward

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for debugging."""
        return {
            "step": self.current_step,
            "checkpoint": self.last_checkpoint,
            "checkpoints_passed": self.checkpoints_passed,
            "speed": self.car.speed,
            "episode_reward": self.episode_reward,
            "collided": self.car.collided,
            "touching_wall": self.car.touching_wall,
            "wall_contact_steps": self.wall_contact_steps,
            "on_track": self.last_on_track,
            "off_track_steps": self.off_track_steps,
        }

    def set_spawn_checkpoint(self, checkpoint: Optional[int]) -> None:
        """Set the spawn checkpoint for cohort spawning.

        Args:
            checkpoint: Checkpoint index to spawn at, or None to disable.
        """
        self.spawn_checkpoint = checkpoint

    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if self.renderer is None:
            return None

        return self.renderer.render(
            car=self.car,
            track=self.track,
            lidar=self.lidar,
            info=self._get_info(),
            obs_type=self.config.obs_type,
        )

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_env(config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
    """Factory function for creating the environment."""
    return RacingEnv(config=config, render_mode=render_mode)
