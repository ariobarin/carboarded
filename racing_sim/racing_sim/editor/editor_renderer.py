"""Renderer for the track editor."""

import math
from typing import List, Optional, Tuple
import pygame
from pymunk import Vec2d

from racing_sim.editor.editor_state import EditorState, EditorMode, NodeData
from racing_sim.editor.geometry import (
    compute_fillet,
    offset_line,
    offset_arc,
    discretize_arc,
    CenterlineElement,
    arc_length,
    line_line_intersection,
    FilletResult,
    polygon_signed_area,
)
from racing_sim.editor.validation import (
    ValidationResult,
    ValidationIssue,
    compute_curvature_at_point,
    get_curvature_color,
)


class EditorRenderer:
    """Renderer for the track editor interface."""

    # Colors
    BACKGROUND = (40, 40, 40)
    GRID_COLOR = (60, 60, 60)
    NODE_COLOR = (100, 150, 255)
    NODE_SELECTED = (255, 200, 100)
    NODE_HOVER = (150, 200, 255)
    EDGE_COLOR = (80, 80, 80)
    CENTERLINE_COLOR = (100, 100, 100)
    WALL_OUTER_COLOR = (150, 150, 150)
    WALL_INNER_COLOR = (120, 120, 120)
    CHECKPOINT_COLOR = (50, 80, 50)
    START_COLOR = (0, 255, 100)
    ERROR_COLOR = (255, 80, 80)
    WARNING_COLOR = (255, 200, 80)
    RADIUS_HANDLE_COLOR = (255, 180, 100)
    TEXT_COLOR = (200, 200, 200)

    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the renderer.

        Args:
            screen_width: Width of the screen.
            screen_height: Height of the screen.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen: Optional[pygame.Surface] = None
        self._initialized = False

        # Cached fonts
        self._font_small: Optional[pygame.font.Font] = None
        self._font_medium: Optional[pygame.font.Font] = None

    def init(self):
        """Initialize pygame."""
        if self._initialized:
            return

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption("Track Editor")

        self._font_small = pygame.font.Font(None, 20)
        self._font_medium = pygame.font.Font(None, 28)
        self._initialized = True

    def resize(self, screen_width: int, screen_height: int):
        """Handle window resize."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        if self._initialized:
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height), pygame.DOUBLEBUF | pygame.RESIZABLE
            )

    def render(
        self,
        state: EditorState,
        validation: Optional[ValidationResult] = None,
        fillets: Optional[List[FilletResult]] = None,
    ):
        """Render the editor interface.

        Args:
            state: Current editor state.
            validation: Optional validation result for overlay.
            fillets: Optional precomputed fillets.
        """
        self.init()
        self.screen.fill(self.BACKGROUND)

        # Render grid if enabled
        if state.view.grid_snap:
            self._render_grid(state)

        # Compute fillets if not provided
        if fillets is None and len(state.track.nodes) >= 3:
            fillets = self._compute_fillets(state.track.nodes)

        # Render track elements (back to front)
        if state.show_checkpoints and fillets:
            self._render_checkpoints(state, fillets)

        if state.show_walls and fillets:
            self._render_walls(state, fillets)

        if state.show_curvature and fillets:
            self._render_curvature_heatmap(state, fillets)

        # Render centerline
        self._render_centerline(state, fillets)

        # Render edges (connections between nodes)
        self._render_edges(state)

        # Render validation issues
        if state.show_validation and validation:
            self._render_validation(state, validation)

        # Render nodes
        self._render_nodes(state, fillets)

        # Render start position marker
        if len(state.track.nodes) >= 3:
            self._render_start_marker(state, fillets)

        # Render HUD
        self._render_hud(state, validation)

        # Render help text
        self._render_help(state)

        pygame.display.flip()

    def _compute_fillets(self, nodes: List[NodeData]) -> List[FilletResult]:
        """Compute fillets for all nodes."""
        n = len(nodes)
        fillets = []
        for i in range(n):
            prev = Vec2d(nodes[(i - 1) % n].x, nodes[(i - 1) % n].y)
            curr = Vec2d(nodes[i].x, nodes[i].y)
            next_pt = Vec2d(nodes[(i + 1) % n].x, nodes[(i + 1) % n].y)
            radius = nodes[i].radius

            fillet = compute_fillet(prev, curr, next_pt, radius)
            fillets.append(fillet)
        return fillets

    def _render_grid(self, state: EditorState):
        """Render the snap grid."""
        grid_size = state.view.grid_size * state.view.zoom

        # Only render if grid cells are visible
        if grid_size < 5:
            return

        # Calculate visible range
        start_x = int((-state.view.offset_x) / grid_size) * grid_size
        start_y = int((-state.view.offset_y) / grid_size) * grid_size

        for x in range(int(start_x), self.screen_width + int(grid_size), int(grid_size)):
            screen_x = int(x + state.view.offset_x)
            if 0 <= screen_x < self.screen_width:
                pygame.draw.line(
                    self.screen, self.GRID_COLOR,
                    (screen_x, 0), (screen_x, self.screen_height), 1
                )

        for y in range(int(start_y), self.screen_height + int(grid_size), int(grid_size)):
            screen_y = int(self.screen_height - y - state.view.offset_y)
            if 0 <= screen_y < self.screen_height:
                pygame.draw.line(
                    self.screen, self.GRID_COLOR,
                    (0, screen_y), (self.screen_width, screen_y), 1
                )

    def _render_edges(self, state: EditorState):
        """Render edges between nodes."""
        n = len(state.track.nodes)
        if n < 2:
            return

        for i in range(n):
            node = state.track.nodes[i]
            next_node = state.track.nodes[(i + 1) % n]

            p1 = state.world_to_screen(node.x, node.y, self.screen_height)
            p2 = state.world_to_screen(next_node.x, next_node.y, self.screen_height)

            color = self.EDGE_COLOR
            if state.hovered_edge == i:
                color = self.NODE_HOVER

            pygame.draw.line(self.screen, color, p1, p2, 2)

    def _render_centerline(self, state: EditorState, fillets: Optional[List[FilletResult]]):
        """Render the track centerline with fillets."""
        if not fillets or len(fillets) < 3:
            return

        n = len(fillets)

        for i in range(n):
            fillet = fillets[i]
            next_fillet = fillets[(i + 1) % n]

            # Draw arc at this node
            if not fillet.is_collinear and fillet.radius > 1e-6:
                self._draw_arc(
                    state, fillet.center, fillet.radius,
                    fillet.start_angle, fillet.sweep_angle,
                    self.CENTERLINE_COLOR, 2
                )

            # Draw line to next node
            p1 = state.world_to_screen(fillet.tangent_out.x, fillet.tangent_out.y, self.screen_height)
            p2 = state.world_to_screen(next_fillet.tangent_in.x, next_fillet.tangent_in.y, self.screen_height)
            pygame.draw.line(self.screen, self.CENTERLINE_COLOR, p1, p2, 2)

    def _render_walls(self, state: EditorState, fillets: List[FilletResult]):
        """Render inner and outer walls."""
        half_width = state.track.width / 2.0
        n = len(fillets)
        if n < 2:
            return

        node_positions = [Vec2d(node.x, node.y) for node in state.track.nodes]
        winding = polygon_signed_area(node_positions)
        inside_offset = half_width if winding >= 0 else -half_width
        outside_offset = -inside_offset

        # Precompute arc endpoints for offset walls
        outer_arc_pts: List[Optional[Tuple[Vec2d, Vec2d]]] = [None] * n
        inner_arc_pts: List[Optional[Tuple[Vec2d, Vec2d]]] = [None] * n

        for i in range(n):
            fillet = fillets[i]
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
            fillet = fillets[i]
            next_fillet = fillets[(i + 1) % n]
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

        # Draw arcs and trim lines for tight or sharp corners
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
                miter_limit = max(state.track.width * 1.5, max(prev_len, next_len) * 1.5)
                if dist_prev <= miter_limit and dist_next <= miter_limit:
                    line_list[prev_idx] = (prev_line[0], intersection)
                    line_list[next_idx] = (intersection, next_line[1])
                    return

            bevels.append((prev_line[1], next_line[0]))
        for i in range(n):
            fillet = fillets[i]
            if fillet.is_collinear or fillet.radius <= 1e-6:
                continue

            _, outer_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, outside_offset
            )
            _, inner_radius, _, _ = offset_arc(
                fillet.center, fillet.radius, fillet.start_angle, fillet.sweep_angle, inside_offset
            )

            if outer_radius > 1e-6:
                self._draw_arc(
                    state, fillet.center, outer_radius,
                    fillet.start_angle, fillet.sweep_angle,
                    self.WALL_OUTER_COLOR, 2
                )
            if inner_radius > 1e-6:
                self._draw_arc(
                    state, fillet.center, inner_radius,
                    fillet.start_angle, fillet.sweep_angle,
                    self.WALL_INNER_COLOR, 2
                )
            else:
                _apply_corner_join(inner_lines, i, inner_bevels)

        # Sharp corners (radius ~ 0): join both inner and outer lines.
        for i in range(n):
            fillet = fillets[i]
            if fillet.is_collinear or fillet.radius > 1e-6:
                continue
            _apply_corner_join(inner_lines, i, inner_bevels)
            _apply_corner_join(outer_lines, i, outer_bevels)

        # Draw line segments after any adjustments
        for line in outer_lines:
            if line:
                self._draw_line(state, line[0], line[1], self.WALL_OUTER_COLOR, 2)
        for line in inner_lines:
            if line:
                self._draw_line(state, line[0], line[1], self.WALL_INNER_COLOR, 2)
        for bevel in inner_bevels:
            self._draw_line(state, bevel[0], bevel[1], self.WALL_INNER_COLOR, 2)
        for bevel in outer_bevels:
            self._draw_line(state, bevel[0], bevel[1], self.WALL_OUTER_COLOR, 2)

    def _render_checkpoints(self, state: EditorState, fillets: List[FilletResult]):
        """Render checkpoint lines."""
        if len(fillets) < 3:
            return

        # Sample points along centerline for checkpoints
        half_width = state.track.width / 2.0
        n = len(fillets)

        # Simple checkpoint: one at each node's fillet center
        for i in range(n):
            fillet = fillets[i]
            if fillet.is_collinear:
                pos = Vec2d(state.track.nodes[i].x, state.track.nodes[i].y)
                # Use direction to next node
                next_pos = Vec2d(state.track.nodes[(i + 1) % n].x, state.track.nodes[(i + 1) % n].y)
                tangent = (next_pos - pos).normalized()
            else:
                # Position at center of arc
                mid_angle = fillet.start_angle + fillet.sweep_angle / 2
                pos = fillet.center + Vec2d(math.cos(mid_angle), math.sin(mid_angle)) * fillet.radius

                # Tangent perpendicular to radius
                if fillet.sweep_angle >= 0:
                    tangent = Vec2d(-math.sin(mid_angle), math.cos(mid_angle))
                else:
                    tangent = Vec2d(math.sin(mid_angle), -math.cos(mid_angle))

            # Perpendicular direction
            perp = Vec2d(-tangent.y, tangent.x)
            inner = pos - perp * half_width
            outer = pos + perp * half_width

            self._draw_line(state, inner, outer, self.CHECKPOINT_COLOR, 1)

    def _render_curvature_heatmap(self, state: EditorState, fillets: List[FilletResult]):
        """Render curvature heatmap on centerline."""
        n = len(fillets)

        for i in range(n):
            fillet = fillets[i]
            if fillet.is_collinear or fillet.radius < 1e-6:
                continue

            curvature = compute_curvature_at_point(fillets, i)
            color = get_curvature_color(curvature)

            self._draw_arc(
                state, fillet.center, fillet.radius,
                fillet.start_angle, fillet.sweep_angle,
                color, 4
            )

    def _render_nodes(self, state: EditorState, fillets: Optional[List[FilletResult]]):
        """Render track nodes."""
        for i, node in enumerate(state.track.nodes):
            screen_pos = state.world_to_screen(node.x, node.y, self.screen_height)

            # Choose color based on state
            if i == state.selected_node:
                color = self.NODE_SELECTED
                radius = 12
            elif i == state.hovered_node:
                color = self.NODE_HOVER
                radius = 10
            else:
                color = self.NODE_COLOR
                radius = 8

            # Draw node circle
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, radius, 2)

            # Draw node index
            text = self._font_small.render(str(i), True, self.TEXT_COLOR)
            self.screen.blit(text, (screen_pos[0] + 12, screen_pos[1] - 8))

            # Draw radius handle for selected node
            if i == state.selected_node and node.radius > 0:
                # Show radius indicator
                radius_text = self._font_small.render(f"R:{node.radius:.0f}", True, self.RADIUS_HANDLE_COLOR)
                self.screen.blit(radius_text, (screen_pos[0] + 12, screen_pos[1] + 8))

                handle_pos = None
                if fillets and i < len(fillets):
                    fillet = fillets[i]
                    if not fillet.is_collinear and fillet.radius > 1e-6:
                        direction = Vec2d(node.x, node.y) - fillet.center
                        if direction.length > 1e-6:
                            direction = direction.normalized()
                            handle_pos = Vec2d(node.x, node.y) + direction * node.radius

                if state.adjusting_radius and state.radius_node == i:
                    mouse_pos = pygame.mouse.get_pos()
                    handle_pos = state.screen_to_world(
                        mouse_pos[0], mouse_pos[1], self.screen_height
                    )

                if handle_pos is not None:
                    handle_screen = state.world_to_screen(
                        handle_pos.x, handle_pos.y, self.screen_height
                    )
                    pygame.draw.line(
                        self.screen, self.RADIUS_HANDLE_COLOR,
                        screen_pos, handle_screen, 2
                    )
                    pygame.draw.circle(self.screen, self.RADIUS_HANDLE_COLOR, handle_screen, 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), handle_screen, 6, 2)

    def _render_start_marker(self, state: EditorState, fillets: Optional[List[FilletResult]]):
        """Render the start position marker."""
        if not fillets or len(fillets) < 3:
            return
        pos, direction = self._compute_start_marker(state, fillets)
        if pos is None or direction is None:
            return

        screen_pos = state.world_to_screen(pos.x, pos.y, self.screen_height)

        # Draw start marker (triangle)
        size = 15
        # Calculate triangle points
        forward = Vec2d(direction.x, -direction.y) * size  # Flip Y for screen
        left = Vec2d(direction.y, direction.x) * size * 0.5
        right = Vec2d(-direction.y, -direction.x) * size * 0.5

        points = [
            (screen_pos[0] + forward.x, screen_pos[1] + forward.y),
            (screen_pos[0] + left.x - forward.x * 0.3, screen_pos[1] + left.y - forward.y * 0.3),
            (screen_pos[0] + right.x - forward.x * 0.3, screen_pos[1] + right.y - forward.y * 0.3),
        ]

        pygame.draw.polygon(self.screen, self.START_COLOR, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

    def _compute_start_marker(
        self,
        state: EditorState,
        fillets: List[FilletResult],
    ) -> Tuple[Optional[Vec2d], Optional[Vec2d]]:
        """Compute start marker position and tangent direction."""
        n = len(fillets)
        if n < 3:
            return None, None

        # Build centerline elements and node start distances
        elements: List[CenterlineElement] = []
        node_start_distances = [0.0 for _ in range(n)]
        total_length = 0.0

        for i in range(n):
            fillet = fillets[i]
            next_fillet = fillets[(i + 1) % n]
            node_start_distances[i] = total_length

            start = fillet.tangent_out
            end = next_fillet.tangent_in
            line_length = (end - start).length
            if line_length > 1e-6:
                elements.append(CenterlineElement(
                    element_type="line",
                    start=start,
                    end=end,
                    length=line_length,
                ))
                total_length += line_length

            if not next_fillet.is_collinear and next_fillet.radius > 1e-6:
                arc_len = arc_length(next_fillet.radius, abs(next_fillet.sweep_angle))
                elements.append(CenterlineElement(
                    element_type="arc",
                    start=next_fillet.tangent_in,
                    end=next_fillet.tangent_out,
                    center=next_fillet.center,
                    radius=next_fillet.radius,
                    start_angle=next_fillet.start_angle,
                    sweep_angle=next_fillet.sweep_angle,
                    length=arc_len,
                ))
                total_length += arc_len

        if total_length < 1e-6 or not elements:
            return None, None

        start_idx = state.track.start_node_index % n
        base_dist = node_start_distances[start_idx]
        target_dist = (base_dist + state.track.start_offset) % total_length

        remaining = max(0.0, target_dist)
        for elem in elements:
            if remaining <= elem.length + 1e-9:
                t = remaining / max(elem.length, 1e-10)
                if elem.element_type == "line":
                    pos = elem.start + (elem.end - elem.start) * t
                    direction = (elem.end - elem.start).normalized()
                else:
                    angle = elem.start_angle + elem.sweep_angle * t
                    pos = elem.center + Vec2d(math.cos(angle), math.sin(angle)) * elem.radius
                    if elem.sweep_angle >= 0:
                        direction = Vec2d(-math.sin(angle), math.cos(angle))
                    else:
                        direction = Vec2d(math.sin(angle), -math.cos(angle))
                return pos, direction
            remaining -= elem.length

        # Fallback to end of last element
        last = elements[-1]
        if last.element_type == "line":
            return last.end, (last.end - last.start).normalized()
        angle = last.start_angle + last.sweep_angle
        pos = last.center + Vec2d(math.cos(angle), math.sin(angle)) * last.radius
        if last.sweep_angle >= 0:
            direction = Vec2d(-math.sin(angle), math.cos(angle))
        else:
            direction = Vec2d(math.sin(angle), -math.cos(angle))
        return pos, direction

    def _render_validation(self, state: EditorState, validation: ValidationResult):
        """Render validation issues."""
        for issue in validation.issues:
            color = self.WARNING_COLOR if issue.severity == "warning" else self.ERROR_COLOR
            if issue.position:
                screen_pos = state.world_to_screen(
                    issue.position.x, issue.position.y, self.screen_height
                )

                # Draw error indicator
                pygame.draw.circle(self.screen, color, screen_pos, 20, 3)
                if issue.severity != "warning":
                    pygame.draw.line(
                        self.screen, color,
                        (screen_pos[0] - 8, screen_pos[1] - 8),
                        (screen_pos[0] + 8, screen_pos[1] + 8), 3
                    )
                    pygame.draw.line(
                        self.screen, color,
                        (screen_pos[0] + 8, screen_pos[1] - 8),
                        (screen_pos[0] - 8, screen_pos[1] + 8), 3
                    )

            elif issue.node_index is not None:
                node = state.track.nodes[issue.node_index]
                screen_pos = state.world_to_screen(node.x, node.y, self.screen_height)
                pygame.draw.circle(self.screen, color, screen_pos, 18, 3)

    def _render_hud(self, state: EditorState, validation: Optional[ValidationResult]):
        """Render heads-up display."""
        y = 10

        # Mode indicator
        mode_text = f"Mode: {state.mode.value.upper()}"
        if state.mode == EditorMode.PREVIEW:
            mode_text += " (ESC to exit)"
        text = self._font_medium.render(mode_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (10, y))
        y += 25

        # Node count
        text = self._font_small.render(
            f"Nodes: {len(state.track.nodes)}", True, self.TEXT_COLOR
        )
        self.screen.blit(text, (10, y))
        y += 20

        # Track width
        text = self._font_small.render(
            f"Width: {state.track.width:.0f}", True, self.TEXT_COLOR
        )
        self.screen.blit(text, (10, y))
        y += 20

        # Zoom level
        text = self._font_small.render(
            f"Zoom: {state.view.zoom:.2f}x", True, self.TEXT_COLOR
        )
        self.screen.blit(text, (10, y))
        y += 20

        # Grid snap
        if state.view.grid_snap:
            text = self._font_small.render(
                f"Grid: {state.view.grid_size:.0f}", True, self.TEXT_COLOR
            )
            self.screen.blit(text, (10, y))
            y += 20

        # Selected node info
        if state.selected_node is not None and state.selected_node < len(state.track.nodes):
            node = state.track.nodes[state.selected_node]
            text = self._font_small.render(
                f"Selected: Node {state.selected_node} ({node.x:.0f}, {node.y:.0f})",
                True, self.NODE_SELECTED
            )
            self.screen.blit(text, (10, y))
            y += 20

        # Start position info
        text = self._font_small.render(
            f"Start: Node {state.track.start_node_index} | Offset {state.track.start_offset:.0f}",
            True, self.TEXT_COLOR
        )
        self.screen.blit(text, (10, y))
        y += 20

        # Validation status
        if validation:
            error_count = sum(
                1 for issue in validation.issues if issue.severity == "error"
            )
            warning_count = sum(
                1 for issue in validation.issues if issue.severity == "warning"
            )

            if error_count == 0:
                if warning_count > 0:
                    text = self._font_small.render(
                        f"Warnings: {warning_count}", True, self.WARNING_COLOR
                    )
                else:
                    text = self._font_small.render("Track Valid", True, (100, 255, 100))
            else:
                status = f"Errors: {error_count}"
                if warning_count:
                    status += f" | Warnings: {warning_count}"
                text = self._font_small.render(status, True, self.ERROR_COLOR)

            self.screen.blit(text, (10, y))
            y += 20

            # Show first issue
            if validation.issues:
                issue = validation.issues[0]
                color = self.WARNING_COLOR if issue.severity == "warning" else self.ERROR_COLOR
                text = self._font_small.render(issue.message, True, color)
                self.screen.blit(text, (10, y))

        # Dirty indicator
        if state.dirty:
            text = self._font_small.render("*Unsaved", True, (255, 200, 100))
            self.screen.blit(text, (self.screen_width - 80, 10))

        # File path
        if state.file_path:
            text = self._font_small.render(state.file_path, True, (150, 150, 150))
            self.screen.blit(text, (self.screen_width - text.get_width() - 10, 30))

    def _render_help(self, state: EditorState):
        """Render help text at bottom of screen."""
        if state.mode == EditorMode.PREVIEW:
            help_text = "WASD/Arrows: Drive | R: Reset | L: Lidar | ESC/P: Exit Preview"
        else:
            help_text = "Click+Drag: Add | Click empty: Deselect | Drag radius handle: Radius | Del: Delete | G: Grid | C: Checkpoints | H: Curvature | P: Preview | S: Set Start | [/]: Start Offset | Ctrl+S: Save | Ctrl+O: Load"

        lines = self._wrap_help_text(help_text, self.screen_width - 20)
        line_height = self._font_small.get_linesize()
        total_height = line_height * len(lines)
        y = self.screen_height - 10 - total_height
        y = max(10, y)

        for line in lines:
            text = self._font_small.render(line, True, (150, 150, 150))
            self.screen.blit(text, (10, y))
            y += line_height

    def _wrap_help_text(self, text: str, max_width: int) -> List[str]:
        """Wrap help text so it stays within the screen."""
        if self._font_small is None:
            return [text]

        tokens = text.split(" ")
        lines: List[str] = []
        current: List[str] = []

        for token in tokens:
            candidate = " ".join(current + [token]) if current else token
            if self._font_small.size(candidate)[0] <= max_width:
                current.append(token)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [token]

        if current:
            lines.append(" ".join(current))

        return lines

    def _draw_line(
        self,
        state: EditorState,
        p1: Vec2d,
        p2: Vec2d,
        color: Tuple[int, int, int],
        width: int = 1,
    ):
        """Draw a line in world coordinates."""
        screen_p1 = state.world_to_screen(p1.x, p1.y, self.screen_height)
        screen_p2 = state.world_to_screen(p2.x, p2.y, self.screen_height)
        pygame.draw.line(self.screen, color, screen_p1, screen_p2, width)

    def _draw_arc(
        self,
        state: EditorState,
        center: Vec2d,
        radius: float,
        start_angle: float,
        sweep_angle: float,
        color: Tuple[int, int, int],
        width: int = 1,
        num_segments: int = 16,
    ):
        """Draw an arc in world coordinates."""
        if abs(radius) < 1e-6 or abs(sweep_angle) < 1e-6:
            return

        points = discretize_arc(center, radius, start_angle, sweep_angle, num_segments)
        screen_points = [
            state.world_to_screen(p.x, p.y, self.screen_height)
            for p in points
        ]

        if len(screen_points) >= 2:
            pygame.draw.lines(self.screen, color, False, screen_points, width)

    def close(self):
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
