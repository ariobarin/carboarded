"""Utilities for tracking angular progress around the track."""

import math


def progress_delta(last_angle: float, current_angle: float) -> float:
    """Return signed angular delta with wrap-around in [-pi, pi]."""
    delta = current_angle - last_angle
    if delta > math.pi:
        delta -= math.tau
    elif delta < -math.pi:
        delta += math.tau
    return delta


def progress_delta_cyclic(last_value: float, current_value: float, period: float) -> float:
    """Return signed delta with wrap-around in [-period/2, period/2]."""
    if period <= 0.0:
        return 0.0

    delta = current_value - last_value
    half = period * 0.5
    if delta > half:
        delta -= period
    elif delta < -half:
        delta += period
    return delta
