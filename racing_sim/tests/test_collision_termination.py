import numpy as np

from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv


def test_collision_does_not_terminate_when_disabled():
    config = EnvConfig()
    config.collision_penalty = 0.0
    config.random_start = False
    config.max_off_track_steps = 0
    config.terminate_on_collision = False

    env = RacingEnv(config=config, render_mode=None)
    env.reset(seed=0)
    env.car.collided = True

    _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert terminated is False
    assert truncated is False
