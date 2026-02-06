import math

from racing_sim.utils.reward import compute_slowdown_penalty


def test_slowdown_penalty_zero_when_far():
    obs = [1.0, 0.9, 0.8, 0.9, 1.0]
    penalty = compute_slowdown_penalty(obs, speed_ratio=1.0, threshold=0.4, scale=2.0, ray_indices=[1, 2, 3])
    assert penalty == 0.0


def test_slowdown_penalty_negative_when_close():
    obs = [0.2, 0.2, 0.2, 0.2, 0.2]
    penalty = compute_slowdown_penalty(obs, speed_ratio=1.0, threshold=0.4, scale=2.0, ray_indices=[1, 2, 3])
    assert penalty < 0.0


def test_slowdown_penalty_scales_with_speed():
    obs = [0.2, 0.2, 0.2, 0.2, 0.2]
    slow = compute_slowdown_penalty(obs, speed_ratio=0.25, threshold=0.4, scale=2.0, ray_indices=[1, 2, 3])
    fast = compute_slowdown_penalty(obs, speed_ratio=1.0, threshold=0.4, scale=2.0, ray_indices=[1, 2, 3])
    assert fast < slow


def test_slowdown_penalty_disabled_with_zero_scale():
    obs = [0.2, 0.2, 0.2, 0.2, 0.2]
    penalty = compute_slowdown_penalty(obs, speed_ratio=1.0, threshold=0.4, scale=0.0, ray_indices=[1, 2, 3])
    assert penalty == 0.0
