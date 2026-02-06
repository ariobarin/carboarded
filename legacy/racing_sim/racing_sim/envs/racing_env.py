"""Main Gymnasium environment for the racing simulation."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import math
from typing import Optional, Tuple, Dict, Any
from racing_sim.config.config import EnvConfig
from racing_sim.physics.car import Car
from racing_sim.physics.track import Track
from racing_sim.sensors.lidar import Lidar
from racing_sim.rendering.renderer import Renderer
from racing_sim.sensors.grid import compute_grid
from racing_sim.utils.progress import progress_delta
from racing_sim.utils.reward import compute_slowdown_penalty


class RacingEnv(gym.Env):
    """
    2D Racing Environment with Lidar observations.

    Observation Space: Box(0, 1, shape=(5,)) - 5 normalized lidar distances
    Action Space: Box([-1, 0], [1, 1]) - [steering, throttle]

    Reward:
        - +checkpoint_reward per checkpoint passed
        - +speed_bonus_scale * (speed / max_speed) per step
        - collision_penalty on wall hit (terminates episode)
        - time_penalty per step
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

        self.config = config or EnvConfig()
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

        # Create track
        self.track = Track(self.space, self.config.track)

        # Create car at start position
        self.car = Car(
            self.space,
            self.config.car,
            position=(self.track.start_position.x, self.track.start_position.y),
            angle=self.track.start_angle
        )

        # Create lidar sensor
        self.lidar = Lidar(self.space, self.config.lidar)

        # Create renderer if needed
        if self.render_mode is not None:
            self.renderer = Renderer(self.config.render, self.render_mode)

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
        if self.spawn_checkpoint is not None:
            # Cohort spawn: all envs use the same checkpoint per rollout
            start_pos, start_angle = self.track.get_spawn_position(self.spawn_checkpoint)
        elif self.config.random_start:
            # Pick a random checkpoint and spawn there
            checkpoint_idx = self.np_random.integers(0, self.track.num_checkpoints)
            start_pos, start_angle = self.track.get_spawn_position(checkpoint_idx)
        else:
            # Use default start position
            start_pos = self.track.start_position
            start_angle = self.track.start_angle

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
        self.last_progress_angle = self._get_progress_angle(self.car.position)

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
        self.car.update_friction()

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

        # Check termination conditions
        terminated = self.car.collided
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

        # Time penalty (encourages efficiency)
        reward += self.config.time_penalty

        # Speed bonus
        speed_ratio = self.car.speed / self.config.car.max_speed
        reward += self.config.speed_bonus_scale * speed_ratio

        # Checkpoint reward
        current_checkpoint, crossed = self.track.get_progress(
            self.car.position,
            self.last_checkpoint
        )

        if crossed:
            reward += self.config.checkpoint_reward
            self.last_checkpoint = current_checkpoint
            self.checkpoints_passed += 1

        # Collision penalty
        if self.car.collided:
            reward += self.config.collision_penalty

        # Progress reward (continuous around track)
        if self.config.progress_reward_scale != 0.0:
            current_angle = self._get_progress_angle(self.car.position)
            delta = progress_delta(self.last_progress_angle, current_angle)
            reward += self.config.progress_reward_scale * delta
            self.last_progress_angle = current_angle

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

    def _get_progress_angle(self, position) -> float:
        """Return normalized angle around track center in [0, 2pi)."""
        delta = position - self.track.center
        angle = math.atan2(delta.y, delta.x)
        if angle < 0:
            angle += math.tau
        return angle

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary for debugging."""
        return {
            "step": self.current_step,
            "checkpoint": self.last_checkpoint,
            "checkpoints_passed": self.checkpoints_passed,
            "speed": self.car.speed,
            "episode_reward": self.episode_reward,
            "collided": self.car.collided,
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
            info=self._get_info()
        )

    def close(self):
        """Clean up resources."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_env(config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
    """Factory function for creating the environment."""
    return RacingEnv(config=config, render_mode=render_mode)
