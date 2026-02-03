import pymunk

from racing_sim.editor.node_track import NodeTrack


def _pos_for_checkpoint(track: NodeTrack, idx: int, offset: float = 0.01):
    spacing = track._centerline_length / len(track.checkpoints)
    distance = (idx + offset) * spacing
    position, _ = track._sample_centerline_at_distance(distance)
    return position


def test_progress_counts_skipped_checkpoints_with_wrap():
    nodes = [
        (0.0, 0.0, 0.0),
        (200.0, 0.0, 0.0),
        (200.0, 200.0, 0.0),
        (0.0, 200.0, 0.0),
    ]
    space = pymunk.Space()
    track = NodeTrack(space, nodes, width=40.0, num_checkpoints=16)

    position = _pos_for_checkpoint(track, 6)
    current, passed = track.get_progress(position, last_checkpoint=0, max_skip=10)
    assert current == 6
    assert passed == 6

    position = _pos_for_checkpoint(track, 2)
    current, passed = track.get_progress(position, last_checkpoint=14, max_skip=10)
    assert current == 2
    assert passed == 4

    position = _pos_for_checkpoint(track, 11)
    current, passed = track.get_progress(position, last_checkpoint=0, max_skip=10)
    assert current == 11
    assert passed == 0
