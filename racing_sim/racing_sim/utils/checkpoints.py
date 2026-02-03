"""Checkpoint utilities for rendering and visualization."""

from typing import List, Tuple

from pymunk import Vec2d


def smoothed_checkpoint_lines(
    checkpoints: List[Tuple[Vec2d, Vec2d]],
) -> List[Tuple[Vec2d, Vec2d]]:
    """Return checkpoint lines with orientation smoothed by neighboring midpoints."""
    if len(checkpoints) < 3:
        return checkpoints

    midpoints = []
    widths = []
    for inner, outer in checkpoints:
        mid = (inner + outer) * 0.5
        width = (outer - inner).length
        midpoints.append(mid)
        widths.append(width)

    smoothed = []
    count = len(checkpoints)
    for i in range(count):
        prev_mid = midpoints[(i - 1) % count]
        next_mid = midpoints[(i + 1) % count]
        tangent = next_mid - prev_mid
        if tangent.length < 1e-6:
            smoothed.append(checkpoints[i])
            continue

        tangent = tangent.normalized()
        perp = Vec2d(-tangent.y, tangent.x)
        half_width = widths[i] * 0.5
        if half_width < 1e-6:
            smoothed.append(checkpoints[i])
            continue

        mid = midpoints[i]
        smoothed.append((mid - perp * half_width, mid + perp * half_width))

    return smoothed
