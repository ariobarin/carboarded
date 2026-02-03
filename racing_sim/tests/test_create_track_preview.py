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
