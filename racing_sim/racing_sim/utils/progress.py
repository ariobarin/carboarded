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
