"""Utilities for off-track penalty and termination handling."""

from typing import Tuple


def compute_off_track_state(
    on_track: bool,
    collided: bool,
    prev_off_track_steps: int,
    off_track_penalty: float,
    max_off_track_steps: int,
) -> Tuple[int, float, bool]:
    """Update off-track step count, penalty, and termination flag."""
    if on_track and not collided:
        return 0, 0.0, False

    off_track_steps = prev_off_track_steps + 1
    penalty = off_track_penalty * off_track_steps if off_track_penalty else 0.0
    terminated = max_off_track_steps > 0 and off_track_steps >= max_off_track_steps
    return off_track_steps, penalty, terminated
