"""NodeTrack class for node-based custom track geometry.

Implements the same interface as physics/track.py Track class,
allowing custom tracks to be used interchangeably with elliptical tracks.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
import pymunk
from pymunk import Vec2d

from racing_sim.editor.geometry import (
    compute_fillet,
    offset_line,
    offset_arc,
    discretize_arc,
    line_line_intersection,
    point_to_centerline_distance_batch,
    polygon_signed_area,
    CenterlineElement,
    FilletResult,
    arc_length,
)
from racing_sim.physics.car import CATEGORY_WALL


class NodeTrack:
    """Track defined by a series of nodes with fillet radii.

    Nodes connect in series forming a closed circuit. Each node's corner
    is filleted with a circular arc of configurable radius.

    Implements the same interface as Track for compatibility with RacingEnv.
    """

    def __init__(
        self,
        space: pymunk.Space,
        nodes: List[Tuple[float, float, float]],
        width: float = 100.0,
        num_checkpoints: int = 64,
        start_node_index: int = 0,
        start_offset: float = 0.0,
        build_bitmap: bool = True,
    ):
        """Initialize a node-based track.

        Args:
            space: Pymunk physics space.
            nodes: List of (x, y, radius) tuples defining node positions and fillet radii.
            width: Track width (distance between inner and outer walls).
            num_checkpoints: Number of checkpoints to generate.
            start_node_index: Index of the node nearest to start position.
            start_offset: Distance along centerline from node to start position.
        """
        self.space = space
        self.width = width
        self.num_checkpoints_target = num_checkpoints
        self.start_node_index = start_node_index
        self.start_offset = start_offset

        # Convert nodes to Vec2d with radii (allow sharp corners with radius 0)
        self.nodes: List[Tuple[Vec2d, float]] = [
            (Vec2d(x, y), max(0.0, r)) for x, y, r in nodes
        ]

        # Wall segments
        self.outer_walls: List[pymunk.Segment] = []
        self.inner_walls: List[pymunk.Segment] = []

        # Checkpoints
        self.checkpoints: List[Tuple[Vec2d, Vec2d]] = []

        # Centerline elements (for distance calculations)
        self._centerline: List[CenterlineElement] = []
        self._centerline_length: float = 0.0
        self._node_start_distances: List[float] = []

        # Computed fillet results
        self._fillets: List[FilletResult] = []

        # Start position (set in _compute_start_position)
        self.start_position: Vec2d = Vec2d(0, 0)
        self.start_angle: float = 0.0

        # Bitmap for fast track lookups
        self._track_bitmap: Optional[np.ndarray] = None
        self._bitmap_min_x: float = 0.0
        self._bitmap_max_x: float = 0.0
        self._bitmap_min_y: float = 0.0
        self._bitmap_max_y: float = 0.0
        self._bitmap_resolution: float = 1.0

        # Track center (approximate, for checkpoint calculations)
        self._center: Vec2d = Vec2d(0, 0)

        # Build the track
        if len(self.nodes) >= 3:
            self._build_centerline()
            self._create_walls()
            self._create_checkpoints()
            self._compute_start_position()
            if build_bitmap:
                self._create_bitmap()

    def _build_centerline(self):
        """Build centerline from nodes with fillets."""
        n = len(self.nodes)
        if n < 3:
            return

        # Compute fillets at each node
        self._fillets = []
        for i in range(n):
            prev = self.nodes[(i - 1) % n][0]
            curr = self.nodes[i][0]
            next_pt = self.nodes[(i + 1) % n][0]
            radius = self.nodes[i][1]

            fillet = compute_fillet(prev, curr, next_pt, radius)
            self._fillets.append(fillet)

        # Compute track center (centroid of nodes)
        cx = sum(node[0].x for node in self.nodes) / n
        cy = sum(node[0].y for node in self.nodes) / n
        self._center = Vec2d(cx, cy)

        # Build centerline elements: alternating lines and arcs
        self._centerline = []
        self._centerline_length = 0.0
        self._node_start_distances = [0.0 for _ in range(n)]

        for i in range(n):
            fillet = self._fillets[i]
            next_fillet = self._fillets[(i + 1) % n]

            # Start of this node's contribution along the centerline
            self._node_start_distances[i] = self._centerline_length

            # Line from this fillet's tangent_out to next fillet's tangent_in
            start = fillet.tangent_out
            end = next_fillet.tangent_in

            line_length = (end - start).length
            if line_length > 1e-6:
                self._centerline.append(CenterlineElement(
                    element_type="line",
                    start=start,
                    end=end,
                    length=line_length,
                ))
                self._centerline_length += line_length

            # Arc at next node
            if not next_fillet.is_collinear and next_fillet.radius > 1e-6:
                arc_len = arc_length(next_fillet.radius, abs(next_fillet.sweep_angle))
                self._centerline.append(CenterlineElement(
                    element_type="arc",
                    start=next_fillet.tangent_in,
                    end=next_fillet.tangent_out,
                    center=next_fillet.center,
                    radius=next_fillet.radius,
                    start_angle=next_fillet.start_angle,
                    sweep_angle=next_fillet.sweep_angle,
                    length=arc_len,
                ))
                self._centerline_length += arc_len

    def _create_walls(self, segments_per_arc: int = 8):
        """Create inner and outer walls from centerline."""
        half_width = self.width / 2.0
        static_body = self.space.static_body
        n = len(self._fillets)
        if n < 2:
            return

        node_positions = [node[0] for node in self.nodes]
        winding = polygon_signed_area(node_positions)
        inside_offset = half_width if winding >= 0 else -half_width
        outside_offset = -inside_offset

        # Precompute arc endpoints for offset walls
        outer_arc_pts: List[Optional[Tuple[Vec2d, Vec2d]]] = [None] * n
        inner_arc_pts: List[Optional[Tuple[Vec2d, Vec2d]]] = [None] * n

        for i in range(n):
            fillet = self._fillets[i]
            if fillet.is_collinear or fillet.radius <= 1e-6:
                continue

            _, outer_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, outside_offset
            )
            _, inner_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, inside_offset
            )

            end_angle = fillet.start_angle + fillet.sweep_angle

            if outer_radius > 1e-6:
                outer_start = fillet.center + Vec2d(math.cos(fillet.start_angle), math.sin(fillet.start_angle)) * outer_radius
                outer_end = fillet.center + Vec2d(math.cos(end_angle), math.sin(end_angle)) * outer_radius
                outer_arc_pts[i] = (outer_start, outer_end)

            if inner_radius > 1e-6:
                inner_start = fillet.center + Vec2d(math.cos(fillet.start_angle), math.sin(fillet.start_angle)) * inner_radius
                inner_end = fillet.center + Vec2d(math.cos(end_angle), math.sin(end_angle)) * inner_radius
                inner_arc_pts[i] = (inner_start, inner_end)

        # Build offset line segments between fillet tangents (per edge)
        outer_lines: List[Optional[Tuple[Vec2d, Vec2d]]] = [None] * n
        inner_lines: List[Optional[Tuple[Vec2d, Vec2d]]] = [None] * n

        for i in range(n):
            fillet = self._fillets[i]
            next_fillet = self._fillets[(i + 1) % n]
            start = fillet.tangent_out
            end = next_fillet.tangent_in
            if (end - start).length > 1e-6:
                outer_p1, outer_p2 = offset_line(start, end, outside_offset)
                inner_p1, inner_p2 = offset_line(start, end, inside_offset)
                outer_start = outer_arc_pts[i][1] if outer_arc_pts[i] else outer_p1
                outer_end = outer_arc_pts[(i + 1) % n][0] if outer_arc_pts[(i + 1) % n] else outer_p2
                inner_start = inner_arc_pts[i][1] if inner_arc_pts[i] else inner_p1
                inner_end = inner_arc_pts[(i + 1) % n][0] if inner_arc_pts[(i + 1) % n] else inner_p2
                outer_lines[i] = (outer_start, outer_end)
                inner_lines[i] = (inner_start, inner_end)

        # Add arc segments and adjust lines at tight or sharp corners
        inner_bevels: List[Tuple[Vec2d, Vec2d]] = []
        outer_bevels: List[Tuple[Vec2d, Vec2d]] = []

        def _apply_corner_join(
            line_list: List[Optional[Tuple[Vec2d, Vec2d]]],
            corner_index: int,
            bevels: List[Tuple[Vec2d, Vec2d]],
        ) -> None:
            prev_idx = (corner_index - 1) % n
            next_idx = corner_index
            prev_line = line_list[prev_idx]
            next_line = line_list[next_idx]
            if not prev_line or not next_line:
                return

            intersection = line_line_intersection(
                prev_line[0], prev_line[1], next_line[0], next_line[1]
            )
            if intersection is not None:
                dist_prev = (intersection - prev_line[1]).length
                dist_next = (intersection - next_line[0]).length
                prev_len = (prev_line[1] - prev_line[0]).length
                next_len = (next_line[1] - next_line[0]).length
                miter_limit = max(self.width * 1.5, max(prev_len, next_len) * 1.5)
                if dist_prev <= miter_limit and dist_next <= miter_limit:
                    line_list[prev_idx] = (prev_line[0], intersection)
                    line_list[next_idx] = (intersection, next_line[1])
                    return

            bevels.append((prev_line[1], next_line[0]))

        for i in range(n):
            fillet = self._fillets[i]
            if fillet.is_collinear or fillet.radius <= 1e-6:
                continue

            _, outer_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, outside_offset
            )
            _, inner_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, inside_offset
            )

            if outer_radius > 1e-6:
                outer_points = discretize_arc(
                    fillet.center, outer_radius, fillet.start_angle, fillet.sweep_angle, segments_per_arc
                )
                for j in range(len(outer_points) - 1):
                    self._add_wall_segment(
                        static_body, outer_points[j], outer_points[j + 1], self.outer_walls
                    )

            if inner_radius > 1e-6:
                inner_points = discretize_arc(
                    fillet.center, inner_radius, fillet.start_angle, fillet.sweep_angle, segments_per_arc
                )
                for j in range(len(inner_points) - 1):
                    self._add_wall_segment(
                        static_body, inner_points[j], inner_points[j + 1], self.inner_walls
                    )
            else:
                # Tight inner corner: trim adjacent inner lines or bevel.
                _apply_corner_join(inner_lines, i, inner_bevels)

        # Sharp corners (radius ~ 0): join both inner and outer lines.
        for i in range(n):
            fillet = self._fillets[i]
            if fillet.is_collinear or fillet.radius > 1e-6:
                continue
            _apply_corner_join(inner_lines, i, inner_bevels)
            _apply_corner_join(outer_lines, i, outer_bevels)

        # Add line segments after any adjustments
        for line in outer_lines:
            if line:
                self._add_wall_segment(static_body, line[0], line[1], self.outer_walls)
        for line in inner_lines:
            if line:
                self._add_wall_segment(static_body, line[0], line[1], self.inner_walls)
        for bevel in inner_bevels:
            self._add_wall_segment(static_body, bevel[0], bevel[1], self.inner_walls)
        for bevel in outer_bevels:
            self._add_wall_segment(static_body, bevel[0], bevel[1], self.outer_walls)

    def _add_wall_segment(
        self,
        body: pymunk.Body,
        p1: Vec2d,
        p2: Vec2d,
        wall_list: List[pymunk.Segment],
    ):
        """Add a wall segment to the physics space."""
        segment = pymunk.Segment(body, p1, p2, 2.0)
        segment.friction = 0.8
        segment.elasticity = 0.5
        segment.collision_type = 2  # Wall collision type
        segment.filter = pymunk.ShapeFilter(categories=CATEGORY_WALL)
        self.space.add(segment)
        wall_list.append(segment)

    def _create_checkpoints(self):
        """Create checkpoints evenly spaced along the centerline."""
        if not self._centerline or self._centerline_length < 1e-6:
            return

        num_samples = max(0, int(self.num_checkpoints_target))
        if num_samples <= 0:
            return

        sample_spacing = self._centerline_length / num_samples
        sharp_node_distances = [
            self._node_start_distances[i]
            for i, fillet in enumerate(self._fillets)
            if not fillet.is_collinear and fillet.radius <= 1e-6
        ]
        nudge = min(sample_spacing * 0.25, 1.0) if sample_spacing > 0 else 0.0
        half_width = self.width / 2.0

        self.checkpoints = []
        for i in range(num_samples):
            target_dist = i * sample_spacing
            if sharp_node_distances and nudge > 0.0:
                for sharp_dist in sharp_node_distances:
                    if abs(target_dist - sharp_dist) <= 1e-6:
                        target_dist = (target_dist + nudge) % self._centerline_length
                        break

            pos, tangent = self._sample_centerline_at_distance(target_dist)
            # Perpendicular direction
            perp = Vec2d(-tangent.y, tangent.x)
            inner_point = pos - perp * half_width
            outer_point = pos + perp * half_width
            self.checkpoints.append((inner_point, outer_point))

    def _compute_start_position(self):
        """Compute start position and angle."""
        if not self._centerline:
            if self.nodes:
                self.start_position = self.nodes[0][0]
                self.start_angle = 0.0
            return

        if self._centerline_length < 1e-6:
            self.start_position = self.nodes[0][0]
            self.start_angle = 0.0
            return

        # Compute start distance along centerline
        n = len(self._node_start_distances)
        start_idx = self.start_node_index % max(n, 1)
        base_dist = self._node_start_distances[start_idx] if n else 0.0
        target_dist = (base_dist + self.start_offset) % self._centerline_length

        pos, tangent = self._sample_centerline_at_distance(target_dist)
        self.start_position = pos
        self.start_angle = math.atan2(tangent.y, tangent.x)

    def _sample_centerline_at_distance(self, distance: float) -> Tuple[Vec2d, Vec2d]:
        """Sample position and tangent along centerline by distance."""
        if not self._centerline:
            return self.nodes[0][0], Vec2d(1, 0)

        remaining = max(0.0, distance)
        for elem in self._centerline:
            if remaining <= elem.length + 1e-9:
                t = remaining / max(elem.length, 1e-10)
                if elem.element_type == "line":
                    pos = elem.start + (elem.end - elem.start) * t
                    tangent = (elem.end - elem.start).normalized()
                else:
                    angle = elem.start_angle + elem.sweep_angle * t
                    pos = elem.center + Vec2d(math.cos(angle), math.sin(angle)) * elem.radius
                    if elem.sweep_angle >= 0:
                        tangent = Vec2d(-math.sin(angle), math.cos(angle))
                    else:
                        tangent = Vec2d(math.sin(angle), -math.cos(angle))
                return pos, tangent
            remaining -= elem.length

        # Fallback to end of last element
        last = self._centerline[-1]
        if last.element_type == "line":
            tangent = (last.end - last.start).normalized()
            return last.end, tangent
        angle = last.start_angle + last.sweep_angle
        pos = last.center + Vec2d(math.cos(angle), math.sin(angle)) * last.radius
        if last.sweep_angle >= 0:
            tangent = Vec2d(-math.sin(angle), math.cos(angle))
        else:
            tangent = Vec2d(math.sin(angle), -math.cos(angle))
        return pos, tangent

    def _create_bitmap(self, resolution: float = 1.0):
        """Precompute track occupancy bitmap for fast lookups."""
        if not self._centerline:
            return

        # Compute bounding box
        all_x = []
        all_y = []
        for elem in self._centerline:
            all_x.extend([elem.start.x, elem.end.x])
            all_y.extend([elem.start.y, elem.end.y])
            if elem.element_type == "arc" and elem.center is not None:
                # Include arc extents
                all_x.append(elem.center.x - elem.radius)
                all_x.append(elem.center.x + elem.radius)
                all_y.append(elem.center.y - elem.radius)
                all_y.append(elem.center.y + elem.radius)

        padding = self.width + 50
        self._bitmap_min_x = min(all_x) - padding
        self._bitmap_max_x = max(all_x) + padding
        self._bitmap_min_y = min(all_y) - padding
        self._bitmap_max_y = max(all_y) + padding
        self._bitmap_resolution = resolution

        # Compute bitmap dimensions
        width = int((self._bitmap_max_x - self._bitmap_min_x) * resolution)
        height = int((self._bitmap_max_y - self._bitmap_min_y) * resolution)

        # Ensure minimum size
        width = max(width, 1)
        height = max(height, 1)

        # Create grid of world coordinates
        x_coords = np.linspace(self._bitmap_min_x, self._bitmap_max_x, width)
        y_coords = np.linspace(self._bitmap_min_y, self._bitmap_max_y, height)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Compute distance to centerline for all points
        distances = point_to_centerline_distance_batch(
            xx.flatten(), yy.flatten(), self._centerline
        )
        distances = distances.reshape(height, width)

        # On track if distance <= half_width
        half_width = self.width / 2.0
        self._track_bitmap = (distances <= half_width).astype(np.uint8)

    def get_all_wall_segments(self) -> List[Tuple[Vec2d, Vec2d]]:
        """Get all wall segments as point pairs for rendering."""
        segments = []
        for wall in self.outer_walls + self.inner_walls:
            segments.append((wall.a, wall.b))
        return segments

    def get_checkpoint_index(self, position: Vec2d) -> int:
        """Get the index of the checkpoint closest to the given position.

        Args:
            position: Position to check.

        Returns:
            Index of the closest checkpoint.
        """
        if not self.checkpoints:
            return 0

        distance_along, _ = self._project_to_centerline_distance(position)
        num_checkpoints = len(self.checkpoints)
        if num_checkpoints == 0 or self._centerline_length < 1e-6:
            return 0
        distance_along = min(distance_along, self._centerline_length - 1e-6)
        index = int((distance_along / self._centerline_length) * num_checkpoints)
        return index % num_checkpoints

    def _project_to_centerline_distance(
        self,
        position: Vec2d,
        reference_distance: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Project a position onto the centerline, returning distance along and lateral error."""
        if not self._centerline or self._centerline_length < 1e-6:
            return 0.0, float("inf")

        def _angle_in_range(angle: float, start: float, sweep: float) -> bool:
            angle = angle % (2 * math.pi)
            start = start % (2 * math.pi)
            end = (start + sweep) % (2 * math.pi)
            if sweep >= 0:
                if end >= start:
                    return start <= angle <= end
                return angle >= start or angle <= end
            if end <= start:
                return end <= angle <= start
            return angle <= start or angle >= end

        def _angle_delta(angle: float, start: float, sweep: float) -> float:
            if sweep >= 0:
                return (angle - start) % (2 * math.pi)
            return (start - angle) % (2 * math.pi)

        candidates: List[Tuple[float, float]] = []
        accumulated = 0.0

        for elem in self._centerline:
            along = accumulated
            if elem.element_type == "line":
                seg = elem.end - elem.start
                seg_len = elem.length if elem.length > 1e-10 else seg.length
                if seg_len > 1e-10:
                    t = (position - elem.start).dot(seg) / (seg_len * seg_len)
                    t = max(0.0, min(1.0, t))
                    closest = elem.start + seg * t
                    dist = (position - closest).length
                    along = accumulated + seg_len * t
                else:
                    dist = (position - elem.start).length
            else:
                center = elem.center
                radius = elem.radius or 0.0
                start_angle = elem.start_angle or 0.0
                sweep_angle = elem.sweep_angle or 0.0
                if center is None or abs(radius) < 1e-10:
                    dist = (position - elem.start).length
                else:
                    to_point = position - center
                    dist_to_center = to_point.length
                    point_angle = math.atan2(to_point.y, to_point.x)
                    end_angle = start_angle + sweep_angle
                    arc_length_total = elem.length
                    if _angle_in_range(point_angle, start_angle, sweep_angle):
                        delta = _angle_delta(point_angle, start_angle, sweep_angle)
                        along = accumulated + abs(radius) * delta
                        dist = abs(dist_to_center - abs(radius))
                    else:
                        start_pt = center + Vec2d(math.cos(start_angle), math.sin(start_angle)) * radius
                        end_pt = center + Vec2d(math.cos(end_angle), math.sin(end_angle)) * radius
                        dist_start = (position - start_pt).length
                        dist_end = (position - end_pt).length
                        if dist_start <= dist_end:
                            dist = dist_start
                        else:
                            dist = dist_end
                            along = accumulated + arc_length_total

            candidates.append((along, dist))

            accumulated += elem.length

        if not candidates:
            return 0.0, float("inf")

        if reference_distance is None:
            best_along, best_dist = min(candidates, key=lambda item: item[1])
            return best_along % self._centerline_length, best_dist

        ref = reference_distance % self._centerline_length
        min_dist = min(dist for _, dist in candidates)
        tolerance = self.width * 0.5
        eligible = [
            item for item in candidates
            if item[1] <= min_dist + tolerance
        ]
        if not eligible:
            eligible = candidates

        def forward_distance(along: float) -> float:
            return (along - ref) % self._centerline_length

        best_along, best_dist = min(eligible, key=lambda item: forward_distance(item[0]))
        return best_along % self._centerline_length, best_dist

    def get_progress(
        self,
        position: Vec2d,
        last_checkpoint: int,
        max_skip: int = 1,
    ) -> Tuple[int, int]:
        """Calculate progress and count checkpoints passed.

        Args:
            position: Current position.
            last_checkpoint: Previously reached checkpoint index.

        Returns:
            Tuple of (current_checkpoint_index, checkpoints_passed).
        """
        num_checkpoints = len(self.checkpoints)

        if num_checkpoints == 0:
            return 0, 0

        spacing = self._centerline_length / num_checkpoints if self._centerline_length > 1e-6 else 0.0
        if spacing <= 0.0:
            current_checkpoint = self.get_checkpoint_index(position)
        else:
            reference_distance = last_checkpoint * spacing
            distance_along, _ = self._project_to_centerline_distance(
                position, reference_distance=reference_distance
            )
            current_checkpoint = int((distance_along / self._centerline_length) * num_checkpoints) % num_checkpoints

        forward_distance = (current_checkpoint - last_checkpoint) % num_checkpoints
        if forward_distance == 0:
            return current_checkpoint, 0

        max_skip = max(1, int(max_skip))
        if forward_distance <= max_skip:
            return current_checkpoint, int(forward_distance)

        return current_checkpoint, 0

    def progress_coordinate(self, position: Vec2d) -> float:
        """Return a continuous progress coordinate for reward shaping."""
        along, _ = self._project_to_centerline_distance(position)
        return along

    def progress_period(self) -> float:
        """Return the wrap-around period for progress coordinates."""
        return self._centerline_length

    def progress_scale(self) -> float:
        """Return scale factor to keep progress in angle-equivalent units."""
        if self._centerline_length <= 1e-6:
            return 0.0
        return math.tau / self._centerline_length

    def is_on_track(self, position: Vec2d) -> bool:
        """Check if a position is on the track.

        Args:
            position: Position to check.

        Returns:
            True if position is on track.
        """
        return self.is_on_track_fast(position)

    def is_on_track_fast(self, position: Vec2d) -> bool:
        """Fast track check using precomputed bitmap.

        Args:
            position: Position to check.

        Returns:
            True if position is on track.
        """
        if self._track_bitmap is None:
            return False

        # Convert world position to bitmap coordinates
        bx = int((position.x - self._bitmap_min_x) * self._bitmap_resolution)
        by = int((position.y - self._bitmap_min_y) * self._bitmap_resolution)

        # Bounds check
        if bx < 0 or bx >= self._track_bitmap.shape[1]:
            return False
        if by < 0 or by >= self._track_bitmap.shape[0]:
            return False

        return self._track_bitmap[by, bx] > 0

    def is_on_track_batch(self, world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
        """Batch track check using precomputed bitmap.

        Args:
            world_x: Array of x coordinates.
            world_y: Array of y coordinates.

        Returns:
            Boolean array, True where positions are on track.
        """
        if self._track_bitmap is None:
            return np.zeros_like(world_x, dtype=bool)

        # Convert world positions to bitmap coordinates
        bx = ((world_x - self._bitmap_min_x) * self._bitmap_resolution).astype(np.int32)
        by = ((world_y - self._bitmap_min_y) * self._bitmap_resolution).astype(np.int32)

        # Bounds check
        h, w = self._track_bitmap.shape
        valid = (bx >= 0) & (bx < w) & (by >= 0) & (by < h)

        # Clip to valid range for lookup
        bx_safe = np.clip(bx, 0, w - 1)
        by_safe = np.clip(by, 0, h - 1)

        # Lookup and mask invalid positions
        result = self._track_bitmap[by_safe, bx_safe] > 0
        result = result & valid

        return result

    def get_spawn_position(self, checkpoint_index: int) -> Tuple[Vec2d, float]:
        """Get a valid spawn position and angle at a given checkpoint.

        Args:
            checkpoint_index: Index of the checkpoint.

        Returns:
            Tuple of (position, angle) where angle is in radians.
        """
        if not self.checkpoints:
            return self.start_position, self.start_angle

        num_checkpoints = len(self.checkpoints)
        checkpoint_index = checkpoint_index % num_checkpoints

        inner, outer = self.checkpoints[checkpoint_index]

        # Position at center of checkpoint
        position = (inner + outer) * 0.5

        # Angle tangent to track (perpendicular to checkpoint line)
        checkpoint_dir = outer - inner
        tangent = Vec2d(-checkpoint_dir.y, checkpoint_dir.x)
        tangent = tangent.normalized()

        # Determine correct direction (should point in direction of progress)
        # Use next checkpoint to determine direction
        next_idx = (checkpoint_index + 1) % num_checkpoints
        next_inner, next_outer = self.checkpoints[next_idx]
        next_mid = (next_inner + next_outer) * 0.5

        to_next = next_mid - position
        if to_next.dot(tangent) < 0:
            tangent = -tangent

        angle = math.atan2(tangent.y, tangent.x)

        return position, angle

    @property
    def num_checkpoints(self) -> int:
        """Return the number of checkpoints."""
        return len(self.checkpoints)

    @property
    def center(self) -> Vec2d:
        """Return the approximate track center."""
        return self._center

    def get_centerline_elements(self) -> List[CenterlineElement]:
        """Return centerline elements for external use (e.g., rendering)."""
        return self._centerline

    def get_fillets(self) -> List[FilletResult]:
        """Return fillet results for external use (e.g., rendering)."""
        return self._fillets
