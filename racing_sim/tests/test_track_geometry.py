import math
from pymunk import Vec2d
import racing_sim.editor.geometry as geom
from racing_sim.editor.geometry import offset_arc, offset_line
from racing_sim.editor.validation import validate_track
from racing_sim.editor.node_track import NodeTrack
import pymunk


def test_polygon_signed_area_ccw_and_cw():
    assert hasattr(geom, "polygon_signed_area")

    ccw = [Vec2d(0, 0), Vec2d(1, 0), Vec2d(1, 1), Vec2d(0, 1)]
    cw = list(reversed(ccw))

    assert geom.polygon_signed_area(ccw) > 0
    assert geom.polygon_signed_area(cw) < 0


def test_offset_arc_left_side_semantics():
    center = Vec2d(0, 0)
    radius = 10.0
    offset = 2.0

    # CCW arc: left side is inside -> radius should shrink
    _, new_radius_ccw, _, _ = offset_arc(center, radius, 0.0, math.pi / 2, offset)
    assert new_radius_ccw == 8.0

    # CW arc: left side is outside -> radius should grow
    _, new_radius_cw, _, _ = offset_arc(center, radius, 0.0, -math.pi / 2, offset)
    assert new_radius_cw == 12.0


def test_inside_offset_points_toward_centroid():
    assert hasattr(geom, "polygon_signed_area")

    nodes = [Vec2d(0, 0), Vec2d(2, 0), Vec2d(2, 2), Vec2d(0, 2)]
    centroid = Vec2d(1, 1)
    half_width = 0.25

    winding = geom.polygon_signed_area(nodes)
    inside_offset = half_width if winding > 0 else -half_width
    outside_offset = -inside_offset

    start = nodes[0]
    end = nodes[1]

    inner_p1, inner_p2 = offset_line(start, end, inside_offset)
    outer_p1, outer_p2 = offset_line(start, end, outside_offset)

    inner_mid = (inner_p1 + inner_p2) * 0.5
    outer_mid = (outer_p1 + outer_p2) * 0.5

    assert (inner_mid - centroid).length < (outer_mid - centroid).length

    nodes_cw = list(reversed(nodes))
    winding_cw = geom.polygon_signed_area(nodes_cw)
    inside_offset_cw = half_width if winding_cw > 0 else -half_width
    outside_offset_cw = -inside_offset_cw

    start_cw = nodes_cw[0]
    end_cw = nodes_cw[1]
    inner_p1, inner_p2 = offset_line(start_cw, end_cw, inside_offset_cw)
    outer_p1, outer_p2 = offset_line(start_cw, end_cw, outside_offset_cw)

    inner_mid = (inner_p1 + inner_p2) * 0.5
    outer_mid = (outer_p1 + outer_p2) * 0.5

    assert (inner_mid - centroid).length < (outer_mid - centroid).length


def test_validate_track_square_valid():
    nodes = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0),
        (0.0, 10.0, 0.0),
    ]

    result = validate_track(nodes, width=5.0)
    assert result.valid
    assert result.issues == []


def test_validate_track_inner_corner_warning():
    nodes = [
        (0.0, 0.0, 5.0),
        (10.0, 0.0, 5.0),
        (10.0, 10.0, 5.0),
        (0.0, 10.0, 5.0),
    ]

    result = validate_track(nodes, width=20.0)
    assert result.valid
    assert any(issue.issue_type == "warning" for issue in result.issues)
    assert any("inner corner" in issue.message.lower() for issue in result.issues)


def test_node_track_zero_radius_preserved():
    space = pymunk.Space()
    nodes = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0),
        (0.0, 10.0, 0.0),
    ]

    track = NodeTrack(space, nodes, width=4.0)
    fillets = track.get_fillets()

    assert len(fillets) == len(nodes)
    for fillet in fillets:
        assert fillet.radius == 0.0


def test_node_track_sharp_corners_close_walls():
    space = pymunk.Space()
    nodes = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0),
        (0.0, 10.0, 0.0),
    ]

    track = NodeTrack(space, nodes, width=4.0)

    def endpoint_counts(segments, precision=6):
        counts = {}
        for segment in segments:
            for point in (segment.a, segment.b):
                key = (round(point.x, precision), round(point.y, precision))
                counts[key] = counts.get(key, 0) + 1
        return counts

    for segments in (track.outer_walls, track.inner_walls):
        counts = endpoint_counts(segments)
        assert counts, "Expected wall segments to be generated."
        assert all(count >= 2 for count in counts.values())


def test_checkpoints_avoid_sharp_corners():
    space = pymunk.Space()
    nodes = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (10.0, 10.0, 0.0),
        (0.0, 10.0, 0.0),
    ]

    track = NodeTrack(space, nodes, width=4.0, num_checkpoints=4)
    node_positions = [Vec2d(x, y) for x, y, _ in nodes]

    assert len(track.checkpoints) == 4
    for inner, outer in track.checkpoints:
        midpoint = (inner + outer) * 0.5
        assert all((midpoint - node).length > 1e-3 for node in node_positions)
