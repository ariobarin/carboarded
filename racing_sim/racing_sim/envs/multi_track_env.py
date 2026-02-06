"""Multi-track wrapper that cycles across multiple RacingEnv configs."""

from __future__ import annotations

from typing import Iterable, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv


class MultiTrackEnv(gym.Env):
    """Environment that trains on multiple tracks by switching configs per episode."""

    metadata = RacingEnv.metadata

    def __init__(
        self,
        configs: Iterable[EnvConfig],
        render_mode: Optional[str] = None,
        mode: str = "round_robin",
    ):
        super().__init__()

        self._configs: List[EnvConfig] = list(configs)
        if not self._configs:
            raise ValueError("MultiTrackEnv requires at least one config.")

        if mode not in ("round_robin", "random"):
            raise ValueError("MultiTrackEnv mode must be 'round_robin' or 'random'.")

        self._mode = mode
        self.render_mode = render_mode
        self._reset_count = 0
        self._active_env_index = 0
        self._spawn_checkpoint = None

        self._envs = [RacingEnv(config=cfg, render_mode=render_mode) for cfg in self._configs]

        self.observation_space = self._validate_observation_space()
        self.action_space = self._validate_action_space()

    @property
    def active_env_index(self) -> int:
        return self._active_env_index

    def _validate_observation_space(self) -> spaces.Space:
        base = self._envs[0].observation_space
        for env in self._envs[1:]:
            if not self._spaces_compatible(base, env.observation_space):
                raise ValueError("MultiTrackEnv requires all configs to share the same observation space.")
        return base

    def _validate_action_space(self) -> spaces.Space:
        base = self._envs[0].action_space
        for env in self._envs[1:]:
            if not self._spaces_compatible(base, env.action_space):
                raise ValueError("MultiTrackEnv requires all configs to share the same action space.")
        return base

    @staticmethod
    def _spaces_compatible(space_a: spaces.Space, space_b: spaces.Space) -> bool:
        if type(space_a) is not type(space_b):
            return False
        if isinstance(space_a, spaces.Box) and isinstance(space_b, spaces.Box):
            return (
                space_a.shape == space_b.shape
                and space_a.dtype == space_b.dtype
                and np.array_equal(space_a.low, space_b.low)
                and np.array_equal(space_a.high, space_b.high)
            )
        return space_a == space_b

    def _select_env_index(self) -> int:
        if self._mode == "random":
            return int(self.np_random.integers(0, len(self._envs)))
        if self._reset_count == 0:
            return self._active_env_index
        return (self._active_env_index + 1) % len(self._envs)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._active_env_index = self._select_env_index()
        self._reset_count += 1
        return self._envs[self._active_env_index].reset(seed=seed, options=options)

    def step(self, action):
        return self._envs[self._active_env_index].step(action)

    def set_spawn_checkpoint(self, checkpoint: Optional[int]) -> None:
        self._spawn_checkpoint = checkpoint
        for env in self._envs:
            env.set_spawn_checkpoint(checkpoint)

    def render(self):
        return self._envs[self._active_env_index].render()

    def close(self):
        for env in self._envs:
            env.close()
