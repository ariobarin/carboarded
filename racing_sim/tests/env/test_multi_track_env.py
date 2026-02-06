import pytest

from racing_sim.config.config import EnvConfig
from racing_sim.envs.multi_track_env import MultiTrackEnv


def _default_config() -> EnvConfig:
    return EnvConfig.default()


def test_multi_track_env_round_robin_cycles_tracks():
    config_a = _default_config()
    config_b = _default_config()
    config_b.track.width = config_a.track.width + 10.0

    env = MultiTrackEnv([config_a, config_b], mode="round_robin")
    try:
        env.reset(seed=123)
        assert env.active_env_index == 0

        env.reset(seed=123)
        assert env.active_env_index == 1

        env.reset(seed=123)
        assert env.active_env_index == 0
    finally:
        env.close()


def test_multi_track_env_rejects_mismatched_observation_space():
    config_a = _default_config()
    config_b = _default_config()
    # Grid obs depends on grid_size; changing it creates a shape mismatch.
    config_b.grid.grid_size = config_a.grid.grid_size + 4

    with pytest.raises(ValueError, match="observation space"):
        MultiTrackEnv([config_a, config_b], mode="round_robin")
