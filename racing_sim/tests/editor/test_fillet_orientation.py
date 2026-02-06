import math
from pymunk import Vec2d
from racing_sim.editor.geometry import compute_fillet


def signed_distance_to_line(point: Vec2d, line_p1: Vec2d, line_p2: Vec2d) -> float:
    dx = line_p2.x - line_p1.x
    dy = line_p2.y - line_p1.y
    denom = math.hypot(dx, dy)
    if denom < 1e-9:
        return 0.0
    return ((point.x - line_p1.x) * dy - (point.y - line_p1.y) * dx) / denom


def test_fillet_center_on_turn_side_for_right_turn():
    prev = Vec2d(1.3776874061001925, 1.03181761176121)
    curr = Vec2d(-0.31771367667662, -0.9643329988281466)
    next_pt = Vec2d(0.04509888547443408, -0.38026345019834284)

    fillet = compute_fillet(prev, curr, next_pt, radius=0.5)

    assert not fillet.is_collinear
    assert fillet.radius > 1e-6

    d1 = (curr - prev).normalized()
    d2 = (next_pt - curr).normalized()
    cross = d1.x * d2.y - d1.y * d2.x
    assert cross < 0  # right turn

    dist_in = signed_distance_to_line(fillet.center, prev, curr)
    dist_out = signed_distance_to_line(fillet.center, curr, next_pt)

    # Right turn -> center should be on the right side of both edges (positive distance)
    assert dist_in > 1e-6
    assert dist_out > 1e-6


def test_fillet_tangent_points_match_radius():
    prev = Vec2d(1.3776874061001925, 1.03181761176121)
    curr = Vec2d(-0.31771367667662, -0.9643329988281466)
    next_pt = Vec2d(0.04509888547443408, -0.38026345019834284)

    fillet = compute_fillet(prev, curr, next_pt, radius=0.5)

    assert not fillet.is_collinear
    assert fillet.radius > 1e-6

    dist_in = (fillet.tangent_in - fillet.center).length
    dist_out = (fillet.tangent_out - fillet.center).length

    assert abs(dist_in - fillet.radius) < 1e-6
    assert abs(dist_out - fillet.radius) < 1e-6
