import importlib

import pymunk


def import_create_track_module():
    return importlib.import_module("scripts.create_track")


def test_preview_mode_uses_start_offset():
    create_track = import_create_track_module()
    editor = create_track.TrackEditor(800, 600)

    editor.state.track.nodes = [
        create_track.NodeData(x=0.0, y=0.0, radius=0.0),
        create_track.NodeData(x=10.0, y=0.0, radius=0.0),
        create_track.NodeData(x=10.0, y=10.0, radius=0.0),
        create_track.NodeData(x=0.0, y=10.0, radius=0.0),
    ]
    editor.state.track.width = 4.0
    editor.state.track.start_node_index = 0
    editor.state.track.start_offset = 5.0

    editor._enter_preview_mode()
    assert editor.preview_track is not None

    nodes = editor.state.get_nodes_as_tuples()
    expected_track = create_track.NodeTrack(
        pymunk.Space(),
        nodes=nodes,
        width=editor.state.track.width,
        start_node_index=editor.state.track.start_node_index,
        start_offset=editor.state.track.start_offset,
    )

    actual = editor.preview_track.start_position
    expected = expected_track.start_position
    assert (actual - expected).length < 1e-6


def test_preview_recenters_view_on_start_position():
    create_track = import_create_track_module()
    editor = create_track.TrackEditor(800, 600)

    editor.state.track.nodes = [
        create_track.NodeData(x=-500.0, y=-300.0, radius=0.0),
        create_track.NodeData(x=-400.0, y=-300.0, radius=0.0),
        create_track.NodeData(x=-400.0, y=-200.0, radius=0.0),
        create_track.NodeData(x=-500.0, y=-200.0, radius=0.0),
    ]
    editor.state.track.width = 4.0
    editor.state.track.start_node_index = 0
    editor.state.track.start_offset = 0.0

    editor._enter_preview_mode()
    assert editor.preview_track is not None

    start = editor.preview_track.start_position
    screen_x, screen_y = editor.state.world_to_screen(
        start.x, start.y, editor.screen_height
    )
    assert abs(screen_x - editor.screen_width / 2) <= 1
    assert abs(screen_y - editor.screen_height / 2) <= 1


def test_preview_camera_follows_car_position():
    create_track = import_create_track_module()
    editor = create_track.TrackEditor(800, 600)

    editor.state.track.nodes = [
        create_track.NodeData(x=0.0, y=0.0, radius=0.0),
        create_track.NodeData(x=200.0, y=0.0, radius=0.0),
        create_track.NodeData(x=200.0, y=200.0, radius=0.0),
        create_track.NodeData(x=0.0, y=200.0, radius=0.0),
    ]
    editor.state.track.width = 4.0

    editor._enter_preview_mode()
    assert editor.preview_car is not None

    editor.preview_car.body.position = pymunk.Vec2d(1500.0, -800.0)
    editor._sync_preview_camera()

    screen_x, screen_y = editor.state.world_to_screen(
        editor.preview_car.position.x,
        editor.preview_car.position.y,
        editor.screen_height,
    )
    assert abs(screen_x - editor.screen_width / 2) <= 1
    assert abs(screen_y - editor.screen_height / 2) <= 1


def test_preview_restores_view_after_exit():
    create_track = import_create_track_module()
    editor = create_track.TrackEditor(800, 600)

    editor.state.view.offset_x = 123.0
    editor.state.view.offset_y = -456.0
    editor.state.view.zoom = 1.5

    editor.state.track.nodes = [
        create_track.NodeData(x=1000.0, y=800.0, radius=0.0),
        create_track.NodeData(x=1100.0, y=800.0, radius=0.0),
        create_track.NodeData(x=1100.0, y=900.0, radius=0.0),
        create_track.NodeData(x=1000.0, y=900.0, radius=0.0),
    ]
    editor.state.track.width = 4.0

    editor._enter_preview_mode()
    assert (editor.state.view.offset_x, editor.state.view.offset_y) != (123.0, -456.0)

    editor._exit_preview_mode()
    assert editor.state.view.offset_x == 123.0
    assert editor.state.view.offset_y == -456.0
    assert editor.state.view.zoom == 1.5


def test_preview_skips_bitmap_build_for_speed():
    create_track = import_create_track_module()
    editor = create_track.TrackEditor(800, 600)

    editor.state.track.nodes = [
        create_track.NodeData(x=0.0, y=0.0, radius=0.0),
        create_track.NodeData(x=200.0, y=0.0, radius=0.0),
        create_track.NodeData(x=200.0, y=200.0, radius=0.0),
        create_track.NodeData(x=0.0, y=200.0, radius=0.0),
    ]
    editor.state.track.width = 4.0

    editor._enter_preview_mode()
    assert editor.preview_track is not None
    assert editor.preview_track._track_bitmap is None
