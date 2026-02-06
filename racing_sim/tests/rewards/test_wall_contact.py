import numpy as np
from unittest.mock import patch

from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv


def _make_env(**overrides):
    config = EnvConfig.default()
    config.random_start = False
    config.terminate_on_collision = False
    config.collision_penalty = 0.0
    for k, v in overrides.items():
        setattr(config, k, v)
    env = RacingEnv(config=config, render_mode=None)
    env.reset(seed=0)
    return env


def _simulate_wall_contact(env):
    """Patch space.step to simulate wall contact (sets touching_wall after clear)."""
    original_step = env.space.step

    def step_with_contact(dt):
        original_step(dt)
        env.car.touching_wall = True

    return patch.object(env.space, "step", side_effect=step_with_contact)


def test_wall_contact_penalty_applied_when_touching():
    env = _make_env(wall_contact_penalty=-0.5)
    baseline_env = _make_env(wall_contact_penalty=0.0)

    with _simulate_wall_contact(env):
        _, reward, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    _, baseline_reward, _, _, _ = baseline_env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert abs((reward - baseline_reward) - (-0.5)) < 1e-5
    assert terminated is False
    assert info["touching_wall"] is True


def test_wall_contact_penalty_not_applied_when_not_touching():
    env = _make_env(wall_contact_penalty=-0.5)
    assert env.car.touching_wall is False

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))

    baseline_env = _make_env(wall_contact_penalty=0.0)
    _, baseline_reward, _, _, _ = baseline_env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert abs(reward - baseline_reward) < 1e-5


def test_wall_contact_steps_tracked():
    env = _make_env(wall_contact_penalty=-1.0)

    # Simulate touching for 3 steps
    with _simulate_wall_contact(env):
        for i in range(3):
            _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["wall_contact_steps"] == 3

    # Release from wall (no patch, touching_wall stays False after clear)
    _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["wall_contact_steps"] == 0


def test_max_wall_contact_steps_terminates():
    env = _make_env(wall_contact_penalty=-1.0, max_wall_contact_steps=3)

    with _simulate_wall_contact(env):
        # Steps 1 and 2: not yet terminated
        for i in range(2):
            _, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            assert terminated is False

        # Step 3: should terminate
        _, _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is True
    assert info["wall_contact_steps"] == 3


def test_max_wall_contact_steps_resets_on_release():
    env = _make_env(wall_contact_penalty=-1.0, max_wall_contact_steps=3)

    # Touch for 2 steps
    with _simulate_wall_contact(env):
        for _ in range(2):
            env.step(np.array([0.0, 0.0], dtype=np.float32))

    # Release (no patch -- touching_wall cleared, stays False)
    _, _, terminated, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is False
    assert info["wall_contact_steps"] == 0

    # Touch again for 3 steps -- should terminate
    with _simulate_wall_contact(env):
        for i in range(3):
            _, _, terminated, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert terminated is True


def test_collision_penalty_applies_on_terminal_collision():
    env = _make_env(
        wall_contact_penalty=-0.5,
        collision_penalty=-20.0,
        terminate_on_collision=True,
    )
    baseline_env = _make_env(
        wall_contact_penalty=-0.5,
        collision_penalty=0.0,
        terminate_on_collision=True,
    )

    env.car.collided = True
    baseline_env.car.collided = True

    with _simulate_wall_contact(env):
        _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    with _simulate_wall_contact(baseline_env):
        _, baseline_reward, _, _, _ = baseline_env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )

    assert abs((reward - baseline_reward) - (-20.0)) < 1e-5


def test_collision_penalty_ignored_when_non_terminal():
    env = _make_env(wall_contact_penalty=0.0, collision_penalty=-20.0)
    baseline_env = _make_env(wall_contact_penalty=0.0, collision_penalty=0.0)

    env.car.collided = True
    baseline_env.car.collided = True

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    _, baseline_reward, _, _, _ = baseline_env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert abs(reward - baseline_reward) < 1e-5
