"""Visual node-based track creator.

A visual editor for designing custom tracks using draggable nodes with
configurable turn radii. Nodes connect in series forming a closed circuit;
each node's corner is filleted with a circular arc.

Usage:
    py scripts/create_track.py
    py scripts/create_track.py --load configs/custom_tracks/my_track.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame

from racing_sim.editor.editor_state import EditorState, EditorMode, NodeData
from racing_sim.editor.editor_renderer import EditorRenderer
from racing_sim.editor.validation import validate_track
from racing_sim.editor.node_track import NodeTrack
from racing_sim.editor.geometry import compute_fillet
from racing_sim.config.config import EnvConfig, NodeConfig, NodeTrackConfig, TrackConfig
from racing_sim.config.defaults import load_default_env_config
from racing_sim.physics.car import Car
from racing_sim.sensors.lidar import Lidar
from racing_sim.utils.checkpoints import smoothed_checkpoint_lines
import pymunk
from pymunk import Vec2d


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual track editor for creating custom racing tracks"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load an existing track file",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Editor window width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Editor window height (default: 768)",
    )
    return parser.parse_args()


class TrackEditor:
    """Main track editor application."""

    def __init__(self, screen_width: int = 1024, screen_height: int = 768):
        """Initialize the track editor.

        Args:
            screen_width: Width of editor window.
            screen_height: Height of editor window.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = self._default_env_config()

        # Initialize state and renderer
        self.state = EditorState()
        self.renderer = EditorRenderer(screen_width, screen_height)

        # Preview mode components
        self.preview_space: Optional[pymunk.Space] = None
        self.preview_track: Optional[NodeTrack] = None
        self.preview_car: Optional[Car] = None
        self.preview_lidar: Optional[Lidar] = None

        # Clock for timing
        self.clock: Optional[pygame.time.Clock] = None

        # Running flag
        self.running = True

        # Mouse state for dragging
        self._drag_start_pos: Optional[tuple] = None
        self._drag_node_start: Optional[tuple] = None
        self._pending_new_node: bool = False
        self._pending_new_node_start: Optional[tuple] = None
        self._pending_new_node_world: Optional[tuple] = None
        self._radius_drag_start: Optional[float] = None
        self._preview_view_state: Optional[tuple] = None

    def _default_env_config(self) -> EnvConfig:
        """Load the canonical default environment config."""
        return load_default_env_config()

    def run(self, load_path: Optional[str] = None):
        """Run the editor main loop.

        Args:
            load_path: Optional path to load track from.
        """
        self.renderer.init()
        pygame.key.set_repeat(300, 40)
        self.clock = pygame.time.Clock()

        if load_path:
            self._load_track(load_path)
        else:
            # Create a default track to start with
            self.state.create_default_track(self.screen_width, self.screen_height)

        while self.running:
            self._handle_events()

            if self.state.mode == EditorMode.EDIT:
                self._update_edit_mode()
            else:
                self._update_preview_mode()

            self.clock.tick(60)

        self.renderer.close()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.VIDEORESIZE:
                old_w, old_h = self.screen_width, self.screen_height
                center_world = self.state.screen_to_world(old_w / 2, old_h / 2, old_h)
                self.screen_width, self.screen_height = event.w, event.h
                self.renderer.resize(event.w, event.h)

                # Keep the world center fixed at the new screen center.
                new_cx, new_cy = self.screen_width / 2, self.screen_height / 2
                self.state.view.offset_x = new_cx - center_world.x * self.state.view.zoom
                self.state.view.offset_y = (self.screen_height - new_cy) - center_world.y * self.state.view.zoom

            if self.state.mode == EditorMode.EDIT:
                self._handle_edit_event(event)
            else:
                self._handle_preview_event(event)

    def _handle_edit_event(self, event: pygame.event):
        """Handle events in edit mode."""
        if event.type == pygame.KEYDOWN:
            self._handle_edit_keydown(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_edit_mousedown(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._handle_edit_mouseup(event)
        elif event.type == pygame.MOUSEMOTION:
            self._handle_edit_mousemove(event)
        elif event.type == pygame.MOUSEWHEEL:
            self._handle_mousewheel(event)

    def _handle_edit_keydown(self, event: pygame.event):
        """Handle key press in edit mode."""
        mods = pygame.key.get_mods()

        # Ctrl combinations
        if mods & pygame.KMOD_CTRL:
            if event.key == pygame.K_s:
                self._save_track()
            elif event.key == pygame.K_o:
                self._open_track()
            elif event.key == pygame.K_z:
                self.state.undo()
            elif event.key == pygame.K_y:
                self.state.redo()
            elif event.key == pygame.K_n:
                self._new_track()
            return

        # Single key shortcuts
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
            if self.state.selected_node is not None:
                self.state.delete_node(self.state.selected_node)
        elif event.key == pygame.K_g:
            self.state.view.grid_snap = not self.state.view.grid_snap
        elif event.key == pygame.K_c:
            self.state.show_checkpoints = not self.state.show_checkpoints
        elif event.key == pygame.K_h:
            self.state.show_curvature = not self.state.show_curvature
        elif event.key == pygame.K_v:
            self.state.show_validation = not self.state.show_validation
        elif event.key == pygame.K_w:
            self.state.show_walls = not self.state.show_walls
        elif event.key == pygame.K_s:
            # Set start node to selected node
            if self.state.selected_node is not None:
                self.state.save_undo_state()
                self.state.track.start_node_index = self.state.selected_node
                self.state.track.start_offset = 0.0
        elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
            self._enter_preview_mode()
        elif event.key == pygame.K_LEFTBRACKET or event.key == pygame.K_RIGHTBRACKET:
            # Adjust start offset along centerline
            step = 10.0
            if mods & pygame.KMOD_SHIFT:
                step = 50.0
            if event.key == pygame.K_LEFTBRACKET:
                step = -step
            self.state.save_undo_state()
            self.state.track.start_offset += step
        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
            # Increase track width
            self.state.save_undo_state()
            self.state.track.width = min(200, self.state.track.width + 10)
        elif event.key == pygame.K_MINUS:
            # Decrease track width
            self.state.save_undo_state()
            self.state.track.width = max(40, self.state.track.width - 10)
        elif event.key == pygame.K_UP:
            # Increase selected node radius
            if self.state.selected_node is not None:
                self.state.save_undo_state()
                node = self.state.track.nodes[self.state.selected_node]
                node.radius = min(200, node.radius + 10)
        elif event.key == pygame.K_DOWN:
            # Decrease selected node radius
            if self.state.selected_node is not None:
                self.state.save_undo_state()
                node = self.state.track.nodes[self.state.selected_node]
                node.radius = max(0, node.radius - 10)

    def _handle_edit_mousedown(self, event: pygame.event):
        """Handle mouse button down in edit mode."""
        mouse_pos = pygame.mouse.get_pos()
        world_pos = self.state.screen_to_world(mouse_pos[0], mouse_pos[1], self.screen_height)

        if event.button == 1:  # Left click
            radius_node = self._find_radius_handle_at(world_pos)
            if radius_node is not None:
                self.state.selected_node = radius_node
                self.state.adjusting_radius = True
                self.state.radius_node = radius_node
                self._radius_drag_start = self.state.track.nodes[radius_node].radius
                return

            # Check if clicking on a node
            node_idx = self.state.find_node_at(world_pos)
            if node_idx is not None:
                self.state.selected_node = node_idx
                self.state.dragging = True
                self._drag_start_pos = mouse_pos
                node = self.state.track.nodes[node_idx]
                self._drag_node_start = (node.x, node.y)
            else:
                # Check if clicking on an edge (to insert)
                edge_idx = self.state.find_edge_at(world_pos)
                if edge_idx is not None:
                    # Insert node on edge
                    snapped = self.state.snap_to_grid(world_pos)
                    self.state.insert_node(edge_idx, snapped.x, snapped.y)
                    # Start dragging the newly inserted node immediately
                    self.state.dragging = True
                    self._drag_start_pos = mouse_pos
                    self.state.selected_node = edge_idx + 1
                    node = self.state.track.nodes[self.state.selected_node]
                    self._drag_node_start = (node.x, node.y)
                else:
                    # Begin pending new node (drag to place)
                    self._pending_new_node = True
                    self._pending_new_node_start = mouse_pos
                    self._pending_new_node_world = (world_pos.x, world_pos.y)

        elif event.button == 2:  # Middle click - start pan
            self._drag_start_pos = mouse_pos

        elif event.button == 3:  # Right click - delete node
            node_idx = self.state.find_node_at(world_pos)
            if node_idx is not None:
                self.state.delete_node(node_idx)

    def _handle_edit_mouseup(self, event: pygame.event):
        """Handle mouse button up in edit mode."""
        if event.button == 1:  # Left click
            if self.state.dragging and self._drag_start_pos:
                # Save undo state only if we actually moved
                mouse_pos = pygame.mouse.get_pos()
                if self._drag_start_pos != mouse_pos and self._drag_node_start:
                    # Temporarily restore original position for undo
                    node = self.state.track.nodes[self.state.selected_node]
                    current_pos = (node.x, node.y)
                    node.x, node.y = self._drag_node_start
                    self.state.save_undo_state()
                    node.x, node.y = current_pos

            self.state.dragging = False
            self._drag_start_pos = None
            self._drag_node_start = None

            if self.state.adjusting_radius and self.state.radius_node is not None:
                node = self.state.track.nodes[self.state.radius_node]
                if (
                    self._radius_drag_start is not None
                    and abs(node.radius - self._radius_drag_start) > 1e-6
                ):
                    current_radius = node.radius
                    node.radius = self._radius_drag_start
                    self.state.save_undo_state()
                    node.radius = current_radius
                self.state.adjusting_radius = False
                self.state.radius_node = None
                self._radius_drag_start = None
                return

            # Handle pending new node placement or deselect
            if self._pending_new_node and self._pending_new_node_start:
                mouse_pos = pygame.mouse.get_pos()
                dx = mouse_pos[0] - self._pending_new_node_start[0]
                dy = mouse_pos[1] - self._pending_new_node_start[1]
                drag_dist_sq = dx * dx + dy * dy

                if drag_dist_sq >= 25:  # 5px threshold to treat as a drag
                    world_pos = self.state.screen_to_world(
                        mouse_pos[0], mouse_pos[1], self.screen_height
                    )
                    snapped = self.state.snap_to_grid(world_pos)
                    self.state.add_node(snapped.x, snapped.y)
                else:
                    # Clicked empty space without dragging: deselect
                    self.state.selected_node = None

                self._pending_new_node = False
                self._pending_new_node_start = None
                self._pending_new_node_world = None

        elif event.button == 2:  # Middle click
            self._drag_start_pos = None

    def _handle_edit_mousemove(self, event: pygame.event):
        """Handle mouse move in edit mode."""
        mouse_pos = pygame.mouse.get_pos()
        world_pos = self.state.screen_to_world(mouse_pos[0], mouse_pos[1], self.screen_height)

        if self.state.adjusting_radius and self.state.radius_node is not None:
            self._update_radius_drag(world_pos)
            return

        # Update hover state
        self.state.hovered_node = self.state.find_node_at(world_pos)
        if self.state.hovered_node is None:
            self.state.hovered_edge = self.state.find_edge_at(world_pos)
        else:
            self.state.hovered_edge = None

        # Handle node dragging
        if self.state.dragging and self.state.selected_node is not None:
            new_pos = world_pos

            # Snap to grid if enabled
            new_pos = self.state.snap_to_grid(new_pos)

            # Shift key: snap to axis
            if pygame.key.get_mods() & pygame.KMOD_SHIFT and self._drag_node_start:
                anchor = self.state.screen_to_world(
                    self._drag_start_pos[0], self._drag_start_pos[1], self.screen_height
                )
                new_pos = self.state.snap_to_axis(new_pos, anchor)

            self.state.move_node(self.state.selected_node, new_pos.x, new_pos.y)

        # Handle panning (middle mouse button)
        elif self._drag_start_pos and pygame.mouse.get_pressed()[1]:
            dx = mouse_pos[0] - self._drag_start_pos[0]
            dy = mouse_pos[1] - self._drag_start_pos[1]
            self.state.pan(dx, dy)
            self._drag_start_pos = mouse_pos

    def _find_radius_handle_at(self, world_pos: Vec2d, threshold_px: float = 12.0) -> Optional[int]:
        """Return the selected node index if the radius handle is clicked."""
        if self.state.selected_node is None:
            return None

        handle_pos = self._get_radius_handle_world(self.state.selected_node)
        if handle_pos is None:
            return None

        threshold_world = threshold_px / max(self.state.view.zoom, 1e-6)
        if (world_pos - handle_pos).length <= threshold_world:
            return self.state.selected_node

        return None

    def _get_radius_handle_world(self, node_index: int) -> Optional[Vec2d]:
        """Compute world position of the radius handle for a node."""
        nodes = self.state.track.nodes
        if node_index < 0 or node_index >= len(nodes):
            return None
        if len(nodes) < 3:
            return None

        node = nodes[node_index]
        if node.radius <= 1e-6:
            return None

        prev = nodes[(node_index - 1) % len(nodes)]
        next_node = nodes[(node_index + 1) % len(nodes)]
        fillet = compute_fillet(
            Vec2d(prev.x, prev.y),
            Vec2d(node.x, node.y),
            Vec2d(next_node.x, next_node.y),
            node.radius,
        )
        if fillet.is_collinear or fillet.radius <= 1e-6:
            return None

        direction = Vec2d(node.x, node.y) - fillet.center
        if direction.length < 1e-6:
            return None
        direction = direction.normalized()
        return Vec2d(node.x, node.y) + direction * node.radius

    def _update_radius_drag(self, world_pos: Vec2d) -> None:
        """Update node radius while dragging the radius handle."""
        node_index = self.state.radius_node
        nodes = self.state.track.nodes
        if node_index is None or node_index < 0 or node_index >= len(nodes):
            return
        if len(nodes) < 3:
            return

        prev = nodes[(node_index - 1) % len(nodes)]
        node = nodes[node_index]
        next_node = nodes[(node_index + 1) % len(nodes)]

        fillet = compute_fillet(
            Vec2d(prev.x, prev.y),
            Vec2d(node.x, node.y),
            Vec2d(next_node.x, next_node.y),
            max(node.radius, 1.0),
        )
        if fillet.is_collinear:
            return

        node_pos = Vec2d(node.x, node.y)
        new_radius = (world_pos - node_pos).length
        self.state.set_node_radius(node_index, new_radius)

    def _handle_mousewheel(self, event: pygame.event):
        """Handle mouse wheel for zooming."""
        mouse_pos = pygame.mouse.get_pos()
        if event.y > 0:
            self.state.zoom_at(mouse_pos[0], mouse_pos[1], 1.1, self.screen_height)
        elif event.y < 0:
            self.state.zoom_at(mouse_pos[0], mouse_pos[1], 0.9, self.screen_height)

    def _handle_preview_event(self, event: pygame.event):
        """Handle events in preview mode."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_p:
                self._exit_preview_mode()
            elif event.key == pygame.K_r:
                self._reset_preview()
            elif event.key == pygame.K_l:
                # Toggle lidar visualization (handled by renderer)
                pass

    def _update_edit_mode(self):
        """Update and render in edit mode."""
        # Validate track
        nodes = self.state.get_nodes_as_tuples()
        validation = None
        if len(nodes) >= 3:
            validation = validate_track(nodes, self.state.track.width)

        # Render
        self.renderer.render(self.state, validation)

    def _update_preview_mode(self):
        """Update and render in preview mode."""
        if not self.preview_car or not self.preview_track:
            return

        # Get keyboard input
        keys = pygame.key.get_pressed()
        steering = 0.0
        throttle = 0.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 1.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0

        # Apply controls
        self.preview_car.apply_control(steering, throttle)
        self.preview_car.update_friction(self.config.physics_dt)

        # Step physics
        self.preview_space.step(self.config.physics_dt)

        # Scan lidar
        lidar_obs = self.preview_lidar.scan(self.preview_car)

        # Check collision
        if self.preview_car.collided:
            self._reset_preview()

        # Keep the preview camera centered on the car.
        self._sync_preview_camera()

        # Render preview
        self._render_preview(lidar_obs)

    def _render_preview(self, lidar_obs):
        """Render the preview mode."""
        screen = self.renderer.screen
        screen.fill((30, 30, 30))

        # Draw track walls
        for wall in self.preview_track.outer_walls + self.preview_track.inner_walls:
            p1 = self.state.world_to_screen(wall.a.x, wall.a.y, self.screen_height)
            p2 = self.state.world_to_screen(wall.b.x, wall.b.y, self.screen_height)
            pygame.draw.line(screen, (100, 100, 100), p1, p2, 3)

        # Draw checkpoints
        checkpoints = smoothed_checkpoint_lines(self.preview_track.checkpoints)
        for inner, outer in checkpoints:
            p1 = self.state.world_to_screen(inner.x, inner.y, self.screen_height)
            p2 = self.state.world_to_screen(outer.x, outer.y, self.screen_height)
            pygame.draw.line(screen, (50, 80, 50), p1, p2, 1)

        # Draw lidar rays
        rays = self.preview_lidar.get_debug_rays()
        for start, end, distance in rays:
            p1 = self.state.world_to_screen(start.x, start.y, self.screen_height)
            p2 = self.state.world_to_screen(end.x, end.y, self.screen_height)

            # Color based on distance
            t = distance
            r = int(255 * (1 - t))
            g = int(255 * t)
            color = (r, g, 0)

            pygame.draw.line(screen, color, p1, p2, 2)
            if distance < 1.0:
                pygame.draw.circle(screen, (255, 255, 0), p2, 4)

        # Draw car
        corners = self.preview_car.get_corners()
        screen_corners = [
            self.state.world_to_screen(c.x, c.y, self.screen_height)
            for c in corners
        ]
        pygame.draw.polygon(screen, (100, 150, 255), screen_corners)

        # Draw HUD
        font = self.renderer._font_medium
        text = font.render("PREVIEW MODE - ESC to exit", True, (200, 200, 200))
        screen.blit(text, (10, 10))

        text = font.render(f"Speed: {self.preview_car.speed:.1f}", True, (200, 200, 200))
        screen.blit(text, (10, 40))

        pygame.display.flip()

    def _enter_preview_mode(self):
        """Enter preview mode with physics simulation."""
        nodes = self.state.get_nodes_as_tuples()
        if len(nodes) < 3:
            return

        # Validate first
        validation = validate_track(nodes, self.state.track.width)
        if not validation.valid:
            return

        # Create physics space
        self.preview_space = pymunk.Space()
        self.preview_space.gravity = (0, 0)

        # Create track
        num_checkpoints = 64
        if (
            self.config.track.track_type == "custom"
            and self.config.track.custom is not None
        ):
            num_checkpoints = self.config.track.custom.num_checkpoints
        self.preview_track = NodeTrack(
            self.preview_space,
            nodes=nodes,
            width=self.state.track.width,
            num_checkpoints=num_checkpoints,
            start_node_index=self.state.track.start_node_index,
            start_offset=self.state.track.start_offset,
            build_bitmap=False,
        )

        # Create car
        car_config = self.config.car
        self.preview_car = Car(
            self.preview_space,
            car_config,
            position=(self.preview_track.start_position.x, self.preview_track.start_position.y),
            angle=self.preview_track.start_angle,
        )

        # Create lidar
        lidar_config = self.config.lidar
        self.preview_lidar = Lidar(self.preview_space, lidar_config)

        # Cache current view before switching to preview camera.
        self._preview_view_state = (
            self.state.view.offset_x,
            self.state.view.offset_y,
            self.state.view.zoom,
        )
        self._sync_preview_camera()
        self.state.mode = EditorMode.PREVIEW

    def _exit_preview_mode(self):
        """Exit preview mode."""
        self.preview_space = None
        self.preview_track = None
        self.preview_car = None
        self.preview_lidar = None
        if self._preview_view_state is not None:
            self.state.view.offset_x, self.state.view.offset_y, self.state.view.zoom = (
                self._preview_view_state
            )
            self._preview_view_state = None
        self.state.mode = EditorMode.EDIT

    def _reset_preview(self):
        """Reset car position in preview mode."""
        if self.preview_car and self.preview_track:
            self.preview_car.reset(
                position=(self.preview_track.start_position.x, self.preview_track.start_position.y),
                angle=self.preview_track.start_angle,
            )

    def _sync_preview_camera(self):
        """Center the preview camera on the car position."""
        if not self.preview_car:
            return
        zoom = max(self.state.view.zoom, 1e-6)
        self.state.view.offset_x = self.screen_width / 2 - self.preview_car.position.x * zoom
        self.state.view.offset_y = self.screen_height / 2 - self.preview_car.position.y * zoom

    def _save_track(self):
        """Save track to file."""
        try:
            # Use tkinter file dialog
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()

            # Default to custom_tracks directory
            initial_dir = Path(__file__).parent.parent / "configs" / "custom_tracks"
            initial_dir.mkdir(parents=True, exist_ok=True)

            file_path = filedialog.asksaveasfilename(
                initialdir=str(initial_dir),
                title="Save Track",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
                defaultextension=".yaml",
            )
            root.destroy()

            if not file_path:
                return

            self._save_track_to_path(file_path)

        except ImportError:
            # Fallback: save to default location
            file_path = Path(__file__).parent.parent / "configs" / "custom_tracks" / "custom_track.yaml"
            self._save_track_to_path(str(file_path))
            print(f"Saved to {file_path}")

    def _save_track_to_path(self, file_path: str):
        """Save track to specific path."""
        nodes = [
            NodeConfig(x=node.x, y=node.y, radius=node.radius)
            for node in self.state.track.nodes
        ]

        config = self.config if self.config is not None else self._default_env_config()
        num_checkpoints = 64
        if (
            config.track.track_type == "custom"
            and config.track.custom is not None
        ):
            num_checkpoints = config.track.custom.num_checkpoints

        config.track = TrackConfig(
            track_type="custom",
            custom=NodeTrackConfig(
                nodes=nodes,
                width=self.state.track.width,
                start_node_index=self.state.track.start_node_index,
                start_offset=self.state.track.start_offset,
                num_checkpoints=num_checkpoints,
            ),
        )

        config.to_yaml(file_path)
        self.state.file_path = file_path
        self.state.dirty = False
        print(f"Track saved to {file_path}")

    def _open_track(self):
        """Open track from file."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()

            initial_dir = Path(__file__).parent.parent / "configs" / "custom_tracks"

            file_path = filedialog.askopenfilename(
                initialdir=str(initial_dir),
                title="Open Track",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            )
            root.destroy()

            if not file_path:
                return

            self._load_track(file_path)

        except ImportError:
            print("tkinter not available for file dialog")

    def _load_track(self, file_path: str):
        """Load track from file."""
        try:
            config = EnvConfig.from_yaml(file_path)

            if config.track.track_type != "custom" or config.track.custom is None:
                print(f"File is not a custom track: {file_path}")
                return

            self.state.save_undo_state()
            self.state.track.nodes = [
                NodeData(x=n.x, y=n.y, radius=n.radius)
                for n in config.track.custom.nodes
            ]
            self.state.track.width = config.track.custom.width
            self.state.track.start_node_index = config.track.custom.start_node_index
            self.state.track.start_offset = config.track.custom.start_offset

            self.config = config
            self.state.file_path = file_path
            self.state.dirty = False
            self.state.selected_node = None

            print(f"Loaded track from {file_path}")

        except Exception as e:
            print(f"Error loading track: {e}")

    def _new_track(self):
        """Create a new track."""
        self.state.clear()
        self.state.create_default_track(self.screen_width, self.screen_height)
        self.state.file_path = None
        self.state.dirty = False
        self.config = self._default_env_config()


def main():
    """Main entry point."""
    args = parse_args()

    print("Track Editor")
    print("=" * 40)
    print("Controls:")
    print("  Click: Add node / Select node")
    print("  Drag: Move selected node")
    print("  Shift+Drag: Snap to axis")
    print("  Right-click: Delete node")
    print("  Middle-click drag: Pan view")
    print("  Scroll: Zoom")
    print("  G: Toggle grid snap")
    print("  C: Toggle checkpoints")
    print("  H: Toggle curvature heatmap")
    print("  V: Toggle validation overlay")
    print("  W: Toggle walls")
    print("  S: Set start node (selected)")
    print("  [ / ]: Adjust start offset (Shift = faster)")
    print("  P/Space: Preview mode")
    print("  Up/Down: Adjust selected node radius")
    print("  Drag radius handle: Adjust selected node radius")
    print("  +/-: Adjust track width")
    print("  Del: Delete selected node")
    print("  Ctrl+S: Save")
    print("  Ctrl+O: Open")
    print("  Ctrl+Z: Undo")
    print("  Ctrl+Y: Redo")
    print("  ESC: Quit")
    print()

    editor = TrackEditor(args.width, args.height)
    editor.run(args.load)


if __name__ == "__main__":
    main()
