import math

import pytest

from racing_sim.config.config import EnvConfig, NodeConfig, NodeTrackConfig
from racing_sim.envs.racing_env import RacingEnv


def _make_custom_env(nodes):
    config = EnvConfig.default()
    config.progress_reward_scale = 1.0
    config.speed_bonus_scale = 0.0
    config.checkpoint_reward = 0.0
    config.collision_penalty = 0.0
    config.time_penalty = 0.0
    config.slowdown_penalty_scale = 0.0
    config.random_start = False

    config.track.track_type = "custom"
    config.track.custom = NodeTrackConfig(
        nodes=[NodeConfig(x=x, y=y, radius=r) for x, y, r in nodes],
        width=40.0,
        num_checkpoints=16,
        start_node_index=0,
        start_offset=0.0,
    )

    env = RacingEnv(config=config, render_mode=None)
    env.reset()
    return env


def test_progress_reward_uses_centerline_distance_for_custom_tracks():
    nodes = [
        (-100.0, 100.0, 0.0),
        (100.0, 100.0, 0.0),
        (100.0, -100.0, 0.0),
        (-100.0, -100.0, 0.0),
    ]
    env = _make_custom_env(nodes)
    track = env.track
    assert track.__class__.__name__ == "NodeTrack"

    pos1, _ = track._sample_centerline_at_distance(10.0)
    pos2, _ = track._sample_centerline_at_distance(30.0)

    along1, _ = track._project_to_centerline_distance(pos1)
    along2, _ = track._project_to_centerline_distance(pos2)
    length = track._centerline_length
    assert length > 0.0

    # Set previous progress using centerline distance (not angle).
    env.last_progress_angle = along1
    env.car.reset(position=(pos2.x, pos2.y), angle=0.0)

    reward = env._calculate_reward()

    delta = along2 - along1
    half = length * 0.5
    if delta > half:
        delta -= length
    elif delta < -half:
        delta += length

    expected = delta * (math.tau / length)
    assert reward == pytest.approx(expected)
