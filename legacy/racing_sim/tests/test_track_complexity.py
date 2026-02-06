import pymunk
from pymunk import Vec2d

from racing_sim.config.config import TrackConfig
from racing_sim.physics.track import Track


def test_wavy_track_expands_radius_at_peak():
    space = pymunk.Space()
    config = TrackConfig(
        width=10.0,
        outer_radius_x=100.0,
        outer_radius_y=80.0,
        center_x=0.0,
        center_y=0.0,
        waviness=0.2,
        waves=1,
        wave_phase=0.0,
    )
    track = Track(space, config)

    # At pi/2, scale should be > 1.0; this point should be on track.
    point = Vec2d(0.0, 90.0)
    assert track.is_on_track(point)


def test_wavy_checkpoints_exceed_base_radius():
    space = pymunk.Space()
    config = TrackConfig(
        width=10.0,
        outer_radius_x=100.0,
        outer_radius_y=80.0,
        center_x=0.0,
        center_y=0.0,
        waviness=0.2,
        waves=1,
        wave_phase=0.0,
    )
    track = Track(space, config)

    max_outer_y = max(point.y for _, point in track.checkpoints)
    assert max_outer_y > config.outer_radius_y
