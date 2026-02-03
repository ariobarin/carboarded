import numpy as np

from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv


def _make_env(time_penalty: float) -> RacingEnv:
    config = EnvConfig()
    config.time_penalty = time_penalty
    config.speed_bonus_scale = 0.0
    config.checkpoint_reward = 0.0
    config.collision_penalty = 0.0
    config.progress_reward_scale = 0.0
    config.slowdown_penalty_scale = 0.0
    config.random_start = False
    env = RacingEnv(config=config, render_mode=None)
    env.reset()
    return env


def test_time_penalty_scales_with_speed():
    env = _make_env(time_penalty=-1.0)

    env.car.body.velocity = (0.0, 0.0)
    reward_stopped = env._calculate_reward()

    env.car.body.velocity = (env.config.car.max_speed, 0.0)
    reward_fast = env._calculate_reward()

    assert reward_stopped < reward_fast
    assert abs(reward_fast) < 1e-6
