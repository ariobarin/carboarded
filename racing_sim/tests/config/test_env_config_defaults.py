from pathlib import Path

from racing_sim.config.config import EnvConfig
from racing_sim.config.defaults import default_env_config_path


def test_env_config_defaults_match_default_yaml():
    repo_root = Path(__file__).resolve().parents[2]
    # default.yaml is the canonical env defaults source
    yaml_path = repo_root / "configs" / "default.yaml"
    file_config = EnvConfig.from_yaml(str(yaml_path))
    default_config = EnvConfig.default()

    assert str(default_env_config_path()) == str(yaml_path)
    assert default_config.car.max_speed == file_config.car.max_speed
    assert default_config.car.engine_power == file_config.car.engine_power
    assert default_config.car.steering_power == file_config.car.steering_power
    assert default_config.speed_bonus_scale == file_config.speed_bonus_scale
    assert default_config.progress_reward_scale == file_config.progress_reward_scale
    assert default_config.slowdown_distance == file_config.slowdown_distance
    assert default_config.slowdown_penalty_scale == file_config.slowdown_penalty_scale
    assert default_config.collision_penalty == file_config.collision_penalty
    assert default_config.time_penalty == file_config.time_penalty
    assert default_config.random_start == file_config.random_start
    assert default_config.random_start_lateral_fraction == file_config.random_start_lateral_fraction
