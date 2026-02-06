from pathlib import Path
import importlib.util


def load_env_config():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "racing_sim" / "config" / "config.py"
    spec = importlib.util.spec_from_file_location("racing_sim_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.EnvConfig


def test_env_config_defaults_match_default_yaml():
    repo_root = Path(__file__).resolve().parents[1]
    # default.yaml moved to deprecated/ but still used as baseline for EnvConfig() defaults
    yaml_path = repo_root / "configs" / "deprecated" / "default.yaml"
    EnvConfig = load_env_config()
    file_config = EnvConfig.from_yaml(str(yaml_path))
    default_config = EnvConfig()

    assert default_config.car.max_speed == file_config.car.max_speed
    assert default_config.car.engine_power == file_config.car.engine_power
    assert default_config.car.steering_power == file_config.car.steering_power
    assert default_config.speed_bonus_scale == file_config.speed_bonus_scale
    assert default_config.progress_reward_scale == file_config.progress_reward_scale
    assert default_config.slowdown_distance == file_config.slowdown_distance
    assert default_config.slowdown_penalty_scale == file_config.slowdown_penalty_scale
    assert default_config.collision_penalty == file_config.collision_penalty
    assert default_config.time_penalty == file_config.time_penalty
