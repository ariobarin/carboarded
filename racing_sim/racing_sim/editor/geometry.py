"""Geometry functions for node-based track construction.

This module provides pure math functions for computing fillets (rounded corners),
offset curves (for wall generation), and intersection detection.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from pymunk import Vec2d


@dataclass
class FilletResult:
    """Result of computing a fillet at a node.

    Attributes:
        center: Center point of the fillet arc.
        radius: Radius of the fillet arc.
        start_angle: Starting angle of the arc (radians).
        sweep_angle: Sweep angle of the arc (positive = CCW, negative = CW).
        tangent_in: Point where the incoming edge meets the arc.
        tangent_out: Point where the arc meets the outgoing edge.
        max_radius: Maximum possible radius for this geometry.
        is_collinear: True if the three points are collinear (no fillet needed).
    """
    center: Vec2d
    radius: float
    start_angle: float
    sweep_angle: float
    tangent_in: Vec2d
    tangent_out: Vec2d
    max_radius: float
    is_collinear: bool = False


@dataclass
class CenterlineElement:
    """An element of the track centerline (either a line segment or an arc).

    Attributes:
        element_type: "line" or "arc".
        start: Start point of the element.
        end: End point of the element.
        center: Center of arc (None for lines).
        radius: Radius of arc (None for lines).
        start_angle: Starting angle of arc (None for lines).
        sweep_angle: Sweep angle of arc (None for lines).
        length: Arc length of this element.
    """
    element_type: str  # "line" or "arc"
    start: Vec2d
    end: Vec2d
    center: Optional[Vec2d] = None
    radius: Optional[float] = None
    start_angle: Optional[float] = None
    sweep_angle: Optional[float] = None
    length: float = 0.0


def normalize(v: Vec2d) -> Vec2d:
    """Normalize a vector to unit length."""
    length = v.length
    if length < 1e-10:
        return Vec2d(0, 0)
    return v / length


def polygon_signed_area(points: List[Vec2d]) -> float:
    """Compute signed area of a polygon (positive for CCW winding)."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        area += p1.x * p2.y - p2.x * p1.y

    return 0.5 * area


def compute_fillet(
    prev: Vec2d, curr: Vec2d, next_pt: Vec2d, radius: float
) -> FilletResult:
    """Compute fillet arc parameters at a node.

    Given three consecutive nodes (prev, curr, next) and a radius at curr,
    computes the arc that smoothly connects the two edges.

    Args:
        prev: Previous node position.
        curr: Current node position (where fillet is applied).
        next_pt: Next node position.
        radius: Desired fillet radius.

    Returns:
        FilletResult with arc parameters.
    """
    # Edge directions (along the path)
    d1 = normalize(curr - prev)  # Incoming direction
    d2 = normalize(next_pt - curr)  # Outgoing direction

    # Check for collinear points
    cross = d1.x * d2.y - d1.y * d2.x
    dot = d1.x * d2.x + d1.y * d2.y  # Dot product of d1 and d2

    if abs(cross) < 1e-10:
        # Collinear - no fillet needed
        return FilletResult(
            center=curr,
            radius=0.0,
            start_angle=0.0,
            sweep_angle=0.0,
            tangent_in=curr,
            tangent_out=curr,
            max_radius=0.0,
            is_collinear=True,
        )

    # Half angle between edges
    # dot = cos(angle between d1 and d2)
    dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability
    full_angle = math.acos(dot)
    half_angle = full_angle / 2.0

    # Tangent distance from node to arc endpoints
    if abs(math.tan(half_angle)) < 1e-10:
        tan_dist = 0.0
    else:
        tan_dist = radius / math.tan(half_angle)

    # Maximum radius constraint
    dist_to_prev = (curr - prev).length
    dist_to_next = (next_pt - curr).length
    max_tan_dist = min(dist_to_prev, dist_to_next) * 0.999  # Leave small margin
    if abs(math.tan(half_angle)) > 1e-10:
        max_radius = max_tan_dist * math.tan(half_angle)
    else:
        max_radius = 0.0

    # Clamp radius to maximum
    actual_radius = min(radius, max_radius)
    if actual_radius < 1e-6:
        return FilletResult(
            center=curr,
            radius=0.0,
            start_angle=0.0,
            sweep_angle=0.0,
            tangent_in=curr,
            tangent_out=curr,
            max_radius=max_radius,
            is_collinear=False,
        )

    # Recompute tangent distance with actual radius
    tan_dist = actual_radius / math.tan(half_angle)

    # Determine turn side (left turn = CCW, right turn = CW)
    turn_sign = 1.0 if cross > 0 else -1.0

    def left_normal(v: Vec2d) -> Vec2d:
        return Vec2d(-v.y, v.x)

    def right_normal(v: Vec2d) -> Vec2d:
        return Vec2d(v.y, -v.x)

    n1 = left_normal(d1) if turn_sign > 0 else right_normal(d1)
    n2 = left_normal(d2) if turn_sign > 0 else right_normal(d2)

    # Arc center from intersection of offset lines
    p1 = curr + n1 * actual_radius
    p2 = curr + n2 * actual_radius
    center = line_line_intersection(p1, p1 + d1, p2, p2 + d2)
    if center is None:
        # Fallback to bisector-based center when lines are nearly parallel
        bisector = normalize(Vec2d(d1.x + d2.x, d1.y + d2.y))
        if abs(math.sin(half_angle)) < 1e-10:
            center_dist = actual_radius
        else:
            center_dist = actual_radius / math.sin(half_angle)
        center = curr + bisector * center_dist

    # Tangent points (closest points on each edge)
    tangent_in = center - n1 * actual_radius
    tangent_out = center - n2 * actual_radius

    # Arc angles
    start_angle = math.atan2(tangent_in.y - center.y, tangent_in.x - center.x)
    end_angle = math.atan2(tangent_out.y - center.y, tangent_out.x - center.x)

    # Sweep direction: determined by cross product (left turn = CCW, right turn = CW)
    # cross > 0 means left turn (CCW when looking from above)
    # cross < 0 means right turn (CW)

    # Calculate sweep angle
    sweep_angle = end_angle - start_angle

    # Normalize sweep angle based on turn direction
    if cross > 0:  # Left turn - should be CCW (positive sweep)
        if sweep_angle < 0:
            sweep_angle += 2 * math.pi
    else:  # Right turn - should be CW (negative sweep)
        if sweep_angle > 0:
            sweep_angle -= 2 * math.pi

    return FilletResult(
        center=center,
        radius=actual_radius,
        start_angle=start_angle,
        sweep_angle=sweep_angle,
        tangent_in=tangent_in,
        tangent_out=tangent_out,
        max_radius=max_radius,
        is_collinear=False,
    )


def offset_line(p1: Vec2d, p2: Vec2d, offset: float) -> Tuple[Vec2d, Vec2d]:
    """Compute a parallel offset of a line segment.

    Positive offset is to the left of the direction from p1 to p2.

    Args:
        p1: Start point of the line.
        p2: End point of the line.
        offset: Offset distance (positive = left, negative = right).

    Returns:
        Tuple of (offset_p1, offset_p2).
    """
    direction = normalize(p2 - p1)
    # Perpendicular (90 degrees CCW)
    perp = Vec2d(-direction.y, direction.x)

    return (p1 + perp * offset, p2 + perp * offset)


def offset_arc(
    center: Vec2d,
    radius: float,
    start_angle: float,
    sweep_angle: float,
    offset: float,
) -> Tuple[Vec2d, float, float, float]:
    """Compute a parallel offset of an arc.

    Positive offset is to the left of the arc direction.
    For CCW arcs (positive sweep), left is inside (smaller radius).
    For CW arcs (negative sweep), left is outside (larger radius).

    Args:
        center: Center of the arc.
        radius: Radius of the arc.
        start_angle: Starting angle (radians).
        sweep_angle: Sweep angle (positive = CCW, negative = CW).
        offset: Offset distance.

    Returns:
        Tuple of (center, new_radius, start_angle, sweep_angle).
        Center and angles are unchanged; only radius changes.
    """
    # For CCW arcs, left is inside (smaller radius)
    # For CW arcs, left is outside (larger radius)
    if sweep_angle >= 0:  # CCW
        new_radius = radius - offset
    else:  # CW
        new_radius = radius + offset

    return (center, new_radius, start_angle, sweep_angle)


def discretize_arc(
    center: Vec2d,
    radius: float,
    start_angle: float,
    sweep_angle: float,
    num_segments: int = 8,
) -> List[Vec2d]:
    """Discretize an arc into line segment endpoints.

    Args:
        center: Center of the arc.
        radius: Radius of the arc.
        start_angle: Starting angle (radians).
        sweep_angle: Sweep angle (positive = CCW, negative = CW).
        num_segments: Number of line segments to use.

    Returns:
        List of points along the arc (num_segments + 1 points).
    """
    if abs(radius) < 1e-10 or num_segments < 1:
        return [center]

    points = []
    for i in range(num_segments + 1):
        t = i / num_segments
        angle = start_angle + sweep_angle * t
        point = center + Vec2d(math.cos(angle), math.sin(angle)) * radius
        points.append(point)

    return points


def arc_length(radius: float, sweep_angle: float) -> float:
    """Compute the arc length."""
    return abs(radius * sweep_angle)


def point_to_segment_distance(
    point: Vec2d, seg_start: Vec2d, seg_end: Vec2d
) -> float:
    """Compute the minimum distance from a point to a line segment.

    Args:
        point: The query point.
        seg_start: Start of the segment.
        seg_end: End of the segment.

    Returns:
        Minimum distance from point to segment.
    """
    v = seg_end - seg_start
    w = point - seg_start

    c1 = w.dot(v)
    if c1 <= 0:
        return (point - seg_start).length

    c2 = v.dot(v)
    if c2 <= c1:
        return (point - seg_end).length

    t = c1 / c2
    projection = seg_start + v * t
    return (point - projection).length


def point_to_segment_distance_batch(
    points_x: np.ndarray,
    points_y: np.ndarray,
    seg_start: Vec2d,
    seg_end: Vec2d,
) -> np.ndarray:
    """Vectorized distance from multiple points to a line segment.

    Args:
        points_x: Array of x coordinates.
        points_y: Array of y coordinates.
        seg_start: Start of the segment.
        seg_end: End of the segment.

    Returns:
        Array of distances.
    """
    vx = seg_end.x - seg_start.x
    vy = seg_end.y - seg_start.y
    wx = points_x - seg_start.x
    wy = points_y - seg_start.y

    c1 = wx * vx + wy * vy
    c2 = vx * vx + vy * vy

    # Compute projection parameter t, clamped to [0, 1]
    t = np.clip(c1 / max(c2, 1e-10), 0.0, 1.0)

    # Closest points on segment
    closest_x = seg_start.x + t * vx
    closest_y = seg_start.y + t * vy

    # Distances
    dx = points_x - closest_x
    dy = points_y - closest_y
    return np.sqrt(dx * dx + dy * dy)


def point_to_arc_distance(
    point: Vec2d,
    center: Vec2d,
    radius: float,
    start_angle: float,
    sweep_angle: float,
) -> float:
    """Compute the minimum distance from a point to an arc.

    Args:
        point: The query point.
        center: Center of the arc.
        radius: Radius of the arc.
        start_angle: Starting angle (radians).
        sweep_angle: Sweep angle (positive = CCW, negative = CW).

    Returns:
        Minimum distance from point to arc.
    """
    if abs(radius) < 1e-10:
        return (point - center).length

    # Vector from center to point
    to_point = point - center
    dist_to_center = to_point.length

    if dist_to_center < 1e-10:
        return radius

    # Angle of point relative to center
    point_angle = math.atan2(to_point.y, to_point.x)

    # Normalize angles
    end_angle = start_angle + sweep_angle

    # Check if point's angle is within the arc's angular range
    def angle_in_range(angle, start, sweep):
        """Check if angle is within arc range."""
        # Normalize angle to [0, 2*pi)
        angle = angle % (2 * math.pi)
        start = start % (2 * math.pi)

        if sweep >= 0:  # CCW
            end = (start + sweep) % (2 * math.pi)
            if end >= start:
                return start <= angle <= end
            else:  # Wraps around
                return angle >= start or angle <= end
        else:  # CW
            end = (start + sweep) % (2 * math.pi)
            if end <= start:
                return end <= angle <= start
            else:  # Wraps around
                return angle <= start or angle >= end

    if angle_in_range(point_angle, start_angle, sweep_angle):
        # Closest point is on the arc
        return abs(dist_to_center - radius)
    else:
        # Closest point is one of the arc endpoints
        start_point = center + Vec2d(math.cos(start_angle), math.sin(start_angle)) * radius
        end_point = center + Vec2d(math.cos(end_angle), math.sin(end_angle)) * radius
        return min((point - start_point).length, (point - end_point).length)


def point_to_arc_distance_batch(
    points_x: np.ndarray,
    points_y: np.ndarray,
    center: Vec2d,
    radius: float,
    start_angle: float,
    sweep_angle: float,
) -> np.ndarray:
    """Vectorized distance from multiple points to an arc.

    Args:
        points_x: Array of x coordinates.
        points_y: Array of y coordinates.
        center: Center of the arc.
        radius: Radius of the arc.
        start_angle: Starting angle (radians).
        sweep_angle: Sweep angle (positive = CCW, negative = CW).

    Returns:
        Array of distances.
    """
    if abs(radius) < 1e-10:
        dx = points_x - center.x
        dy = points_y - center.y
        return np.sqrt(dx * dx + dy * dy)

    # Vector from center to points
    to_x = points_x - center.x
    to_y = points_y - center.y
    dist_to_center = np.sqrt(to_x * to_x + to_y * to_y)

    # Angle of each point relative to center
    point_angles = np.arctan2(to_y, to_x)

    # Check if each point's angle is within the arc's angular range
    end_angle = start_angle + sweep_angle

    # Normalize to [0, 2*pi)
    point_angles_norm = point_angles % (2 * np.pi)
    start_norm = start_angle % (2 * np.pi)

    if sweep_angle >= 0:  # CCW
        end_norm = (start_angle + sweep_angle) % (2 * np.pi)
        if end_norm >= start_norm:
            in_range = (point_angles_norm >= start_norm) & (point_angles_norm <= end_norm)
        else:  # Wraps around
            in_range = (point_angles_norm >= start_norm) | (point_angles_norm <= end_norm)
    else:  # CW
        end_norm = (start_angle + sweep_angle) % (2 * np.pi)
        if end_norm <= start_norm:
            in_range = (point_angles_norm >= end_norm) & (point_angles_norm <= start_norm)
        else:  # Wraps around
            in_range = (point_angles_norm <= start_norm) | (point_angles_norm >= end_norm)

    # Distance when in range: |dist_to_center - radius|
    dist_in_range = np.abs(dist_to_center - radius)

    # Distance when out of range: min distance to endpoints
    start_x = center.x + math.cos(start_angle) * radius
    start_y = center.y + math.sin(start_angle) * radius
    end_x = center.x + math.cos(end_angle) * radius
    end_y = center.y + math.sin(end_angle) * radius

    dist_to_start = np.sqrt((points_x - start_x) ** 2 + (points_y - start_y) ** 2)
    dist_to_end = np.sqrt((points_x - end_x) ** 2 + (points_y - end_y) ** 2)
    dist_out_range = np.minimum(dist_to_start, dist_to_end)

    return np.where(in_range, dist_in_range, dist_out_range)


def line_line_intersection(
    p1: Vec2d, p2: Vec2d, p3: Vec2d, p4: Vec2d
) -> Optional[Vec2d]:
    """Find the intersection point of two lines (not segments).

    Args:
        p1, p2: Points defining the first line.
        p3, p4: Points defining the second line.

    Returns:
        Intersection point, or None if lines are parallel.
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return Vec2d(x, y)


def segments_intersect(
    p1: Vec2d, p2: Vec2d, p3: Vec2d, p4: Vec2d, epsilon: float = 1e-10
) -> bool:
    """Check if two line segments intersect.

    Args:
        p1, p2: Endpoints of the first segment.
        p3, p4: Endpoints of the second segment.
        epsilon: Tolerance for endpoint touches.

    Returns:
        True if segments intersect (not counting shared endpoints).
    """
    def ccw(a: Vec2d, b: Vec2d, c: Vec2d) -> float:
        """Return positive if CCW, negative if CW, zero if collinear."""
        return (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if ((d1 > epsilon and d2 < -epsilon) or (d1 < -epsilon and d2 > epsilon)) and \
       ((d3 > epsilon and d4 < -epsilon) or (d3 < -epsilon and d4 > epsilon)):
        return True

    return False


def point_to_centerline_distance(
    point: Vec2d, elements: List[CenterlineElement]
) -> float:
    """Compute minimum distance from a point to any centerline element.

    Args:
        point: The query point.
        elements: List of centerline elements.

    Returns:
        Minimum distance to any element.
    """
    if not elements:
        return float('inf')

    min_dist = float('inf')

    for elem in elements:
        if elem.element_type == "line":
            dist = point_to_segment_distance(point, elem.start, elem.end)
        else:  # arc
            dist = point_to_arc_distance(
                point, elem.center, elem.radius, elem.start_angle, elem.sweep_angle
            )
        min_dist = min(min_dist, dist)

    return min_dist


def point_to_centerline_distance_batch(
    points_x: np.ndarray,
    points_y: np.ndarray,
    elements: List[CenterlineElement],
) -> np.ndarray:
    """Vectorized distance from multiple points to centerline.

    Args:
        points_x: Array of x coordinates.
        points_y: Array of y coordinates.
        elements: List of centerline elements.

    Returns:
        Array of minimum distances to any element.
    """
    if not elements:
        return np.full_like(points_x, float('inf'))

    min_dist = np.full_like(points_x, float('inf'), dtype=np.float64)

    for elem in elements:
        if elem.element_type == "line":
            dist = point_to_segment_distance_batch(
                points_x, points_y, elem.start, elem.end
            )
        else:  # arc
            dist = point_to_arc_distance_batch(
                points_x, points_y, elem.center, elem.radius,
                elem.start_angle, elem.sweep_angle
            )
        min_dist = np.minimum(min_dist, dist)

    return min_dist


def sample_centerline(
    elements: List[CenterlineElement], num_samples: int
) -> List[Tuple[Vec2d, Vec2d]]:
    """Sample points and tangent directions along the centerline.

    Args:
        elements: List of centerline elements.
        num_samples: Number of samples to take.

    Returns:
        List of (position, tangent_direction) tuples.
    """
    if not elements:
        return []

    # Compute total arc length
    total_length = sum(elem.length for elem in elements)
    if total_length < 1e-10:
        return []

    samples = []
    sample_spacing = total_length / num_samples

    current_dist = 0.0
    elem_idx = 0
    elem_progress = 0.0  # Progress within current element

    for i in range(num_samples):
        target_dist = i * sample_spacing

        # Advance to the element containing this distance
        while elem_idx < len(elements) and current_dist + elements[elem_idx].length < target_dist:
            current_dist += elements[elem_idx].length
            elem_idx += 1

        if elem_idx >= len(elements):
            elem_idx = len(elements) - 1
            elem_progress = elements[elem_idx].length
        else:
            elem_progress = target_dist - current_dist

        elem = elements[elem_idx]

        if elem.element_type == "line":
            # Linear interpolation along segment
            t = elem_progress / max(elem.length, 1e-10)
            t = max(0.0, min(1.0, t))
            pos = elem.start + (elem.end - elem.start) * t
            tangent = normalize(elem.end - elem.start)
        else:  # arc
            # Angular interpolation along arc
            t = elem_progress / max(elem.length, 1e-10)
            t = max(0.0, min(1.0, t))
            angle = elem.start_angle + elem.sweep_angle * t
            pos = elem.center + Vec2d(math.cos(angle), math.sin(angle)) * elem.radius
            # Tangent is perpendicular to radius
            if elem.sweep_angle >= 0:  # CCW
                tangent = Vec2d(-math.sin(angle), math.cos(angle))
            else:  # CW
                tangent = Vec2d(math.sin(angle), -math.cos(angle))

        samples.append((pos, tangent))

    return samples
