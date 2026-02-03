"""Editor state management with undo/redo support."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import copy
from pymunk import Vec2d


class EditorMode(Enum):
    """Editor modes."""
    EDIT = "edit"
    PREVIEW = "preview"


@dataclass
class NodeData:
    """Data for a single track node."""
    x: float
    y: float
    radius: float = 50.0

    def to_vec2d(self) -> Vec2d:
        """Convert to Vec2d."""
        return Vec2d(self.x, self.y)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to (x, y, radius) tuple."""
        return (self.x, self.y, self.radius)


@dataclass
class ViewState:
    """Camera/view state for the editor."""
    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0
    grid_snap: bool = False
    grid_size: float = 25.0


@dataclass
class TrackState:
    """State of the track being edited."""
    nodes: List[NodeData] = field(default_factory=list)
    width: float = 100.0
    start_node_index: int = 0
    start_offset: float = 0.0


@dataclass
class EditorSnapshot:
    """A snapshot of the track state for undo/redo."""
    track: TrackState


class EditorState:
    """Main editor state with undo/redo history."""

    MAX_HISTORY = 50

    def __init__(self):
        """Initialize editor state."""
        self.mode: EditorMode = EditorMode.EDIT
        self.track: TrackState = TrackState()
        self.view: ViewState = ViewState()

        # Selection state
        self.selected_node: Optional[int] = None
        self.hovered_node: Optional[int] = None
        self.hovered_edge: Optional[int] = None  # Index of edge start node
        self.dragging: bool = False
        self.drag_start: Optional[Vec2d] = None

        # Radius adjustment
        self.adjusting_radius: bool = False
        self.radius_node: Optional[int] = None

        # Undo/redo history
        self._undo_stack: List[EditorSnapshot] = []
        self._redo_stack: List[EditorSnapshot] = []

        # Dirty flag (unsaved changes)
        self.dirty: bool = False

        # Current file path
        self.file_path: Optional[str] = None

        # Overlay toggles
        self.show_walls: bool = True
        self.show_checkpoints: bool = True
        self.show_curvature: bool = False
        self.show_validation: bool = True

    def _create_snapshot(self) -> EditorSnapshot:
        """Create a snapshot of current state."""
        return EditorSnapshot(
            track=copy.deepcopy(self.track)
        )

    def _restore_snapshot(self, snapshot: EditorSnapshot):
        """Restore state from a snapshot."""
        self.track = copy.deepcopy(snapshot.track)

    def save_undo_state(self):
        """Save current state to undo history."""
        self._undo_stack.append(self._create_snapshot())
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        # Clear redo stack when new action is taken
        self._redo_stack.clear()
        self.dirty = True

    def undo(self) -> bool:
        """Undo the last action.

        Returns:
            True if undo was performed.
        """
        if not self._undo_stack:
            return False

        # Save current state to redo stack
        self._redo_stack.append(self._create_snapshot())

        # Restore previous state
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        self.dirty = True

        # Clear selection
        self.selected_node = None
        self.dragging = False

        return True

    def redo(self) -> bool:
        """Redo the last undone action.

        Returns:
            True if redo was performed.
        """
        if not self._redo_stack:
            return False

        # Save current state to undo stack
        self._undo_stack.append(self._create_snapshot())

        # Restore next state
        snapshot = self._redo_stack.pop()
        self._restore_snapshot(snapshot)
        self.dirty = True

        # Clear selection
        self.selected_node = None
        self.dragging = False

        return True

    def add_node(self, x: float, y: float, radius: float = 50.0):
        """Add a node at the given position."""
        self.save_undo_state()
        self.track.nodes.append(NodeData(x=x, y=y, radius=radius))
        self.selected_node = len(self.track.nodes) - 1

    def insert_node(self, edge_index: int, x: float, y: float, radius: float = 50.0):
        """Insert a node on an edge.

        Args:
            edge_index: Index of the node where the edge starts.
            x: X position of the new node.
            y: Y position of the new node.
            radius: Fillet radius of the new node.
        """
        self.save_undo_state()
        # Insert after edge_index
        insert_pos = edge_index + 1
        self.track.nodes.insert(insert_pos, NodeData(x=x, y=y, radius=radius))
        self.selected_node = insert_pos

    def delete_node(self, index: int):
        """Delete a node by index."""
        if index < 0 or index >= len(self.track.nodes):
            return
        if len(self.track.nodes) <= 3:
            return  # Need at least 3 nodes for a closed track

        self.save_undo_state()
        self.track.nodes.pop(index)

        # Adjust start_node_index if needed
        if self.track.start_node_index >= len(self.track.nodes):
            self.track.start_node_index = 0
        elif self.track.start_node_index > index:
            self.track.start_node_index -= 1

        # Clear selection
        self.selected_node = None

    def move_node(self, index: int, x: float, y: float):
        """Move a node to a new position."""
        if index < 0 or index >= len(self.track.nodes):
            return
        self.track.nodes[index].x = x
        self.track.nodes[index].y = y
        self.dirty = True

    def set_node_radius(self, index: int, radius: float):
        """Set the fillet radius of a node."""
        if index < 0 or index >= len(self.track.nodes):
            return
        self.track.nodes[index].radius = max(0.0, radius)
        self.dirty = True

    def get_nodes_as_tuples(self) -> List[Tuple[float, float, float]]:
        """Get nodes as list of (x, y, radius) tuples."""
        return [node.to_tuple() for node in self.track.nodes]

    def screen_to_world(self, screen_x: float, screen_y: float, screen_height: float) -> Vec2d:
        """Convert screen coordinates to world coordinates.

        Args:
            screen_x: Screen X coordinate.
            screen_y: Screen Y coordinate.
            screen_height: Height of the screen.

        Returns:
            World position as Vec2d.
        """
        # Flip Y for Pymunk coordinate system
        world_x = (screen_x - self.view.offset_x) / self.view.zoom
        world_y = (screen_height - screen_y - self.view.offset_y) / self.view.zoom
        return Vec2d(world_x, world_y)

    def world_to_screen(self, world_x: float, world_y: float, screen_height: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.

        Args:
            world_x: World X coordinate.
            world_y: World Y coordinate.
            screen_height: Height of the screen.

        Returns:
            Screen (x, y) tuple.
        """
        screen_x = world_x * self.view.zoom + self.view.offset_x
        screen_y = screen_height - (world_y * self.view.zoom + self.view.offset_y)
        return (int(screen_x), int(screen_y))

    def snap_to_grid(self, pos: Vec2d) -> Vec2d:
        """Snap a position to the grid if enabled.

        Args:
            pos: Position to snap.

        Returns:
            Snapped position.
        """
        if not self.view.grid_snap:
            return pos

        grid = self.view.grid_size
        return Vec2d(
            round(pos.x / grid) * grid,
            round(pos.y / grid) * grid,
        )

    def snap_to_axis(self, pos: Vec2d, anchor: Vec2d, threshold: float = 20.0) -> Vec2d:
        """Snap position to horizontal and/or vertical axis from anchor.

        Snaps to each axis independently if within threshold distance.
        Can snap to both axes simultaneously (resulting in anchor point).

        Args:
            pos: Position to snap.
            anchor: Anchor point.
            threshold: Distance threshold for snapping (in world units, scaled by zoom).

        Returns:
            Position snapped to nearby axes.
        """
        scaled_threshold = threshold / self.view.zoom
        dx = abs(pos.x - anchor.x)
        dy = abs(pos.y - anchor.y)

        result_x = pos.x
        result_y = pos.y

        # Snap to vertical axis (same x as anchor) if close
        if dx < scaled_threshold:
            result_x = anchor.x

        # Snap to horizontal axis (same y as anchor) if close
        if dy < scaled_threshold:
            result_y = anchor.y

        return Vec2d(result_x, result_y)

    def zoom_at(self, screen_x: float, screen_y: float, factor: float, screen_height: float):
        """Zoom centered on a screen position.

        Args:
            screen_x: Screen X coordinate to zoom toward.
            screen_y: Screen Y coordinate to zoom toward.
            factor: Zoom factor (>1 to zoom in, <1 to zoom out).
            screen_height: Height of the screen.
        """
        # Get world position before zoom
        world_pos = self.screen_to_world(screen_x, screen_y, screen_height)

        # Apply zoom
        old_zoom = self.view.zoom
        self.view.zoom = max(0.1, min(10.0, self.view.zoom * factor))

        # Adjust offset to keep world_pos at the same screen position
        new_screen_x = world_pos.x * self.view.zoom + self.view.offset_x
        new_screen_y = screen_height - (world_pos.y * self.view.zoom + self.view.offset_y)

        self.view.offset_x += screen_x - new_screen_x
        self.view.offset_y += (screen_height - screen_y) - (screen_height - new_screen_y)

    def pan(self, dx: float, dy: float):
        """Pan the view.

        Args:
            dx: Screen X delta.
            dy: Screen Y delta.
        """
        self.view.offset_x += dx
        self.view.offset_y -= dy  # Flip Y for world coordinates

    def find_node_at(self, world_pos: Vec2d, radius: float = 15.0) -> Optional[int]:
        """Find a node near the given world position.

        Args:
            world_pos: World position to check.
            radius: Detection radius (in world units, scaled by zoom).

        Returns:
            Index of the nearest node within radius, or None.
        """
        scaled_radius = radius / self.view.zoom
        for i, node in enumerate(self.track.nodes):
            dist = (Vec2d(node.x, node.y) - world_pos).length
            if dist <= scaled_radius:
                return i
        return None

    def find_edge_at(self, world_pos: Vec2d, threshold: float = 10.0) -> Optional[int]:
        """Find an edge near the given world position.

        Args:
            world_pos: World position to check.
            threshold: Detection threshold (in world units, scaled by zoom).

        Returns:
            Index of the edge start node, or None.
        """
        if len(self.track.nodes) < 2:
            return None

        scaled_threshold = threshold / self.view.zoom
        n = len(self.track.nodes)

        for i in range(n):
            p1 = Vec2d(self.track.nodes[i].x, self.track.nodes[i].y)
            p2 = Vec2d(self.track.nodes[(i + 1) % n].x, self.track.nodes[(i + 1) % n].y)

            # Point-to-segment distance
            v = p2 - p1
            w = world_pos - p1
            c1 = w.dot(v)
            c2 = v.dot(v)

            if c2 < 1e-10:
                continue

            if c1 <= 0:
                dist = (world_pos - p1).length
            elif c1 >= c2:
                dist = (world_pos - p2).length
            else:
                t = c1 / c2
                projection = p1 + v * t
                dist = (world_pos - projection).length

            if dist <= scaled_threshold:
                return i

        return None

    def clear(self):
        """Clear all nodes and reset state."""
        self.save_undo_state()
        self.track = TrackState()
        self.selected_node = None
        self.dragging = False

    def create_default_track(self, screen_width: float, screen_height: float):
        """Create a default track to start editing.

        Args:
            screen_width: Screen width.
            screen_height: Screen height.
        """
        self.save_undo_state()

        # Create a simple rectangle track centered on screen
        cx = screen_width / 2
        cy = screen_height / 2
        half_w = 200
        half_h = 150
        radius = 60

        self.track.nodes = [
            NodeData(x=cx - half_w, y=cy - half_h, radius=radius),
            NodeData(x=cx + half_w, y=cy - half_h, radius=radius),
            NodeData(x=cx + half_w, y=cy + half_h, radius=radius),
            NodeData(x=cx - half_w, y=cy + half_h, radius=radius),
        ]
        self.track.width = 100.0
        self.selected_node = None
