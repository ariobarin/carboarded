"""Reward helpers for shaping driving behavior."""

from __future__ import annotations

from typing import Iterable, Sequence


def compute_slowdown_penalty(
    observation: Sequence[float],
    speed_ratio: float,
    threshold: float,
    scale: float,
    ray_indices: Iterable[int],
) -> float:
    """Penalty for going fast when close to walls.

    Returns 0 when disabled or when the nearest selected ray is beyond the threshold.
    """
    if scale <= 0.0 or threshold <= 0.0:
        return 0.0

    if speed_ratio <= 0.0:
        return 0.0

    selected = [observation[i] for i in ray_indices if i < len(observation)]
    if not selected:
        return 0.0

    min_dist = min(selected)
    if min_dist >= threshold:
        return 0.0

    closeness = (threshold - min_dist) / max(threshold, 1e-6)
    return -scale * speed_ratio * closeness
