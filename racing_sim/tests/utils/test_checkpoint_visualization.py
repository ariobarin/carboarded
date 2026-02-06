import math

from pymunk import Vec2d

from racing_sim.utils.checkpoints import smoothed_checkpoint_lines


def _line_angle(inner: Vec2d, outer: Vec2d) -> float:
    direction = outer - inner
    return math.degrees(math.atan2(direction.y, direction.x))


def test_smoothed_checkpoint_lines_rotate_with_curved_midpoints():
    center = Vec2d(0.0, 0.0)
    radius = 100.0
    width = 20.0

    checkpoints = []
    for i in range(9):
        angle = (math.pi / 2.0) * (i / 8.0)
        midpoint = center + Vec2d(math.cos(angle), math.sin(angle)) * radius
        inner = midpoint + Vec2d(0.0, -width * 0.5)
        outer = midpoint + Vec2d(0.0, width * 0.5)
        checkpoints.append((inner, outer))

    smoothed = smoothed_checkpoint_lines(checkpoints)
    angles = [_line_angle(inner, outer) for inner, outer in smoothed]

    assert max(angles) - min(angles) > 45.0
