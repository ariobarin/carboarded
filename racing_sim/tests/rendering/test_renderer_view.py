from pymunk import Vec2d

from racing_sim.config.config import RenderConfig
from racing_sim.rendering.renderer import Renderer


class NodeTrack:
    """Minimal custom track stub for renderer view tests."""

    def __init__(self, segments):
        self._segments = segments

    def get_all_wall_segments(self):
        return self._segments


def test_custom_track_fits_screen_bounds():
    config = RenderConfig(screen_width=800, screen_height=600)
    renderer = Renderer(config, render_mode="rgb_array")

    min_x, max_x = -1000.0, 1000.0
    min_y, max_y = -500.0, 500.0
    segments = [
        (Vec2d(min_x, min_y), Vec2d(max_x, min_y)),
        (Vec2d(max_x, min_y), Vec2d(max_x, max_y)),
        (Vec2d(max_x, max_y), Vec2d(min_x, max_y)),
        (Vec2d(min_x, max_y), Vec2d(min_x, min_y)),
    ]
    track = NodeTrack(segments)

    renderer._refresh_view(track)

    corners = [
        Vec2d(min_x, min_y),
        Vec2d(min_x, max_y),
        Vec2d(max_x, min_y),
        Vec2d(max_x, max_y),
    ]
    for corner in corners:
        x, y = renderer._to_screen(corner)
        assert 0 <= x <= config.screen_width
        assert 0 <= y <= config.screen_height
