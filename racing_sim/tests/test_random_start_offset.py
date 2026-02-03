from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv


def _spawn_offset(env: RacingEnv, checkpoint_idx: int) -> float:
    center_pos, _ = env.track.get_spawn_position(checkpoint_idx)
    inner, outer = env.track.checkpoints[checkpoint_idx]
    normal = outer - inner
    if normal.length <= 1e-6:
        return 0.0
    normal = normal.normalized()
    return (env.car.position - center_pos).dot(normal)


def _safe_half_width(env: RacingEnv, checkpoint_idx: int) -> float:
    inner, outer = env.track.checkpoints[checkpoint_idx]
    width = (outer - inner).length
    car_half_width = env.config.car.height * 0.5
    margin = max(2.0, env.config.car.height * 0.05)
    return 0.5 * width - car_half_width - margin


def test_spawn_lateral_offset_respects_bounds():
    checkpoint_idx = 0
    config = EnvConfig()
    config.random_start = False
    config.random_start_lateral_fraction = 1.0
    env = RacingEnv(config)
    env.set_spawn_checkpoint(checkpoint_idx)
    env.reset(seed=123)

    offset = _spawn_offset(env, checkpoint_idx)
    safe_half_width = _safe_half_width(env, checkpoint_idx)
    assert safe_half_width > 0.0
    assert abs(offset) <= safe_half_width + 1e-6
    env.close()


def test_spawn_lateral_offset_disabled():
    checkpoint_idx = 0
    config = EnvConfig()
    config.random_start = False
    config.random_start_lateral_fraction = 0.0
    env = RacingEnv(config)
    env.set_spawn_checkpoint(checkpoint_idx)
    env.reset(seed=123)

    offset = _spawn_offset(env, checkpoint_idx)
    assert abs(offset) <= 1e-6
    env.close()


def test_spawn_lateral_offset_changes_position():
    checkpoint_idx = 0
    seed = 321

    config_offset = EnvConfig()
    config_offset.random_start = False
    config_offset.random_start_lateral_fraction = 1.0
    env_offset = RacingEnv(config_offset)
    env_offset.set_spawn_checkpoint(checkpoint_idx)
    env_offset.reset(seed=seed)
    pos_offset = env_offset.car.position
    env_offset.close()

    config_center = EnvConfig()
    config_center.random_start = False
    config_center.random_start_lateral_fraction = 0.0
    env_center = RacingEnv(config_center)
    env_center.set_spawn_checkpoint(checkpoint_idx)
    env_center.reset(seed=seed)
    pos_center = env_center.car.position
    env_center.close()

    assert (pos_offset - pos_center).length > 1e-6
