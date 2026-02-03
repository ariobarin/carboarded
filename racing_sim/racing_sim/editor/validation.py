"""Track validation for the editor.

Detects issues like self-intersections, impossible turns, and width violations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pymunk import Vec2d

from racing_sim.editor.geometry import (
    compute_fillet,
    segments_intersect,
    offset_line,
    offset_arc,
    discretize_arc,
    FilletResult,
    polygon_signed_area,
)


@dataclass
class ValidationIssue:
    """A single validation issue."""
    issue_type: str  # "intersection", "impossible_turn", "width_violation"
    message: str
    severity: str = "error"  # "error" or "warning"
    node_index: Optional[int] = None  # For node-specific issues
    edge_index: Optional[int] = None  # For edge-specific issues
    position: Optional[Vec2d] = None  # For rendering the issue location


@dataclass
class ValidationResult:
    """Result of track validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)


def validate_track(
    nodes: List[Tuple[float, float, float]],
    width: float,
) -> ValidationResult:
    """Validate a track for geometric issues.

    Args:
        nodes: List of (x, y, radius) tuples defining the track.
        width: Track width.

    Returns:
        ValidationResult with any issues found.
    """
    issues = []

    if len(nodes) < 3:
        issues.append(ValidationIssue(
            issue_type="insufficient_nodes",
            severity="error",
            message="Track needs at least 3 nodes",
        ))
        return ValidationResult(valid=False, issues=issues)

    n = len(nodes)
    node_positions = [Vec2d(x, y) for x, y, _ in nodes]
    node_radii = [r for _, _, r in nodes]

    half_width = width / 2.0
    winding = polygon_signed_area(node_positions)
    inside_offset = half_width if winding >= 0 else -half_width
    outside_offset = -inside_offset

    # Check for tight turns (inner radius <= 0 -> sharp inner corner)
    fillets = []
    for i in range(n):
        prev = node_positions[(i - 1) % n]
        curr = node_positions[i]
        next_pt = node_positions[(i + 1) % n]
        radius = node_radii[i]

        fillet = compute_fillet(prev, curr, next_pt, radius)
        fillets.append(fillet)

        if not fillet.is_collinear and fillet.radius > 0:
            # Check if inner wall radius would be negative
            _, inner_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, inside_offset
            )

            # Treat negative inner radius as a warning; the inner wall will corner.
            if inner_radius < -1e-6:
                issues.append(ValidationIssue(
                    issue_type="warning",
                    severity="warning",
                    message=f"Warning: inner corner at node {i} (radius < width/2)",
                    node_index=i,
                    position=curr,
                ))

            # Also check if radius exceeds maximum
            if fillet.radius < node_radii[i] - 1e-6:
                issues.append(ValidationIssue(
                    issue_type="radius_clamped",
                    severity="warning",
                    message=f"Radius at node {i} was reduced to fit geometry",
                    node_index=i,
                    position=curr,
                ))

    # Build wall segments for intersection checking
    outer_segments: List[Tuple[Vec2d, Vec2d]] = []
    inner_segments: List[Tuple[Vec2d, Vec2d]] = []

    for i in range(n):
        fillet = fillets[i]
        next_fillet = fillets[(i + 1) % n]

        # Line segment from this fillet's tangent_out to next fillet's tangent_in
        start = fillet.tangent_out
        end = next_fillet.tangent_in

        line_length = (end - start).length
        if line_length > 1e-6:
            outer_p1, outer_p2 = offset_line(start, end, outside_offset)
            inner_p1, inner_p2 = offset_line(start, end, inside_offset)
            outer_segments.append((outer_p1, outer_p2))
            inner_segments.append((inner_p1, inner_p2))

        # Arc at next node
        nf = next_fillet
        if not nf.is_collinear and nf.radius > 1e-6:
            _, outer_radius, _, _ = offset_arc(
                nf.center, nf.radius, nf.start_angle, nf.sweep_angle, outside_offset
            )
            _, inner_radius, _, _ = offset_arc(
                nf.center, nf.radius, nf.start_angle, nf.sweep_angle, inside_offset
            )

            # Only discretize arcs with positive radius
            if outer_radius > 1e-6:
                outer_pts = discretize_arc(
                    nf.center, outer_radius, nf.start_angle, nf.sweep_angle, 8
                )
                for j in range(len(outer_pts) - 1):
                    outer_segments.append((outer_pts[j], outer_pts[j + 1]))

            if inner_radius > 1e-6:
                inner_pts = discretize_arc(
                    nf.center, inner_radius, nf.start_angle, nf.sweep_angle, 8
                )
                for j in range(len(inner_pts) - 1):
                    inner_segments.append((inner_pts[j], inner_pts[j + 1]))

    def has_self_intersection(segments: List[Tuple[Vec2d, Vec2d]]) -> Optional[Vec2d]:
        for i in range(len(segments)):
            for j in range(i + 2, len(segments)):
                # Skip adjacent segments in the same loop
                if j == i + 1:
                    continue
                if i == 0 and j == len(segments) - 1:
                    continue

                seg1 = segments[i]
                seg2 = segments[j]

                if segments_intersect(seg1[0], seg1[1], seg2[0], seg2[1]):
                    mid1 = (seg1[0] + seg1[1]) * 0.5
                    mid2 = (seg2[0] + seg2[1]) * 0.5
                    return (mid1 + mid2) * 0.5
        return None

    # Check for self-intersections within each wall loop
    for loop in (outer_segments, inner_segments):
        pos = has_self_intersection(loop)
        if pos is not None:
            issues.append(ValidationIssue(
                issue_type="self_intersection",
                severity="error",
                message="Track walls intersect",
                position=pos,
            ))
            break

    # Check for outer/inner wall intersection (track crossing itself)
    for i, outer_seg in enumerate(outer_segments):
        for j, inner_seg in enumerate(inner_segments):
            # Skip nearby segments (they're expected to be close)
            if abs(i - j) <= 1 or (i == 0 and j == len(inner_segments) - 1) or (j == 0 and i == len(outer_segments) - 1):
                continue

            if segments_intersect(outer_seg[0], outer_seg[1], inner_seg[0], inner_seg[1]):
                mid = (outer_seg[0] + outer_seg[1] + inner_seg[0] + inner_seg[1]) * 0.25
                issues.append(ValidationIssue(
                    issue_type="track_crossing",
                    severity="error",
                    message="Track crosses itself",
                    position=mid,
                ))
                break
        else:
            continue
        break

    # Check for very short segments (potential numerical issues)
    for i in range(n):
        curr = node_positions[i]
        next_pt = node_positions[(i + 1) % n]
        dist = (next_pt - curr).length

        if dist < width * 0.5:
            issues.append(ValidationIssue(
                issue_type="short_segment",
                severity="warning",
                message=f"Edge {i} is very short, may cause issues",
                edge_index=i,
                position=(curr + next_pt) * 0.5,
            ))

    has_error = any(issue.severity == "error" for issue in issues)
    return ValidationResult(
        valid=not has_error,
        issues=issues,
    )


def compute_curvature_at_point(
    fillets: List[FilletResult],
    node_index: int,
) -> float:
    """Compute curvature at a node.

    Args:
        fillets: List of fillet results.
        node_index: Index of the node.

    Returns:
        Curvature (1/radius), or 0 for straight sections.
    """
    if not fillets or node_index >= len(fillets):
        return 0.0

    fillet = fillets[node_index]
    if fillet.is_collinear or fillet.radius < 1e-6:
        return 0.0

    return 1.0 / fillet.radius


def get_curvature_color(curvature: float, max_curvature: float = 0.02) -> Tuple[int, int, int]:
    """Get a color representing curvature (for heatmap).

    Args:
        curvature: Curvature value (1/radius).
        max_curvature: Maximum curvature for color scaling.

    Returns:
        RGB color tuple.
    """
    # Normalize curvature to [0, 1]
    t = min(1.0, curvature / max_curvature)

    # Color gradient: green (low) -> yellow -> red (high)
    if t < 0.5:
        # Green to yellow
        r = int(255 * (t * 2))
        g = 255
    else:
        # Yellow to red
        r = 255
        g = int(255 * (1 - (t - 0.5) * 2))

    return (r, g, 0)
