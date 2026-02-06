import pymunk

from racing_sim.editor.node_track import NodeTrack


def test_node_track_checkpoint_index_uses_centerline_distance():
    nodes = [
        (0, 0, 0),
        (200, 0, 0),
        (200, 200, 0),
        (100, 200, 0),
        (100, 50, 0),
        (0, 50, 0),
    ]
    space = pymunk.Space()
    track = NodeTrack(space, nodes=nodes, width=40, num_checkpoints=16)
    assert track._centerline_length > 0.0

    distance = 28.0
    position, _ = track._sample_centerline_at_distance(distance)
    spacing = track._centerline_length / len(track.checkpoints)
    assert distance < spacing

    # Naive midpoint-based lookup would pick a later checkpoint here.
    naive_idx = min(
        range(len(track.checkpoints)),
        key=lambda i: (position - (track.checkpoints[i][0] + track.checkpoints[i][1]) * 0.5).length,
    )
    assert naive_idx != 0

    assert track.get_checkpoint_index(position) == 0


def test_node_track_progress_prefers_forward_centerline():
    nodes = [
        (0, 0, 0),
        (200, 0, 0),
        (200, 200, 0),
        (100, 200, 0),
        (100, 50, 0),
        (0, 50, 0),
    ]
    space = pymunk.Space()
    track = NodeTrack(space, nodes=nodes, width=40, num_checkpoints=16)
    num_checkpoints = len(track.checkpoints)
    assert track._centerline_length > 0.0

    spacing = track._centerline_length / num_checkpoints
    min_x = int(track._bitmap_min_x)
    max_x = int(track._bitmap_max_x)
    min_y = int(track._bitmap_min_y)
    max_y = int(track._bitmap_max_y)

    found = None
    for last_checkpoint in range(num_checkpoints):
        ref = last_checkpoint * spacing
        for x in range(min_x, max_x + 1, 2):
            for y in range(min_y, max_y + 1, 2):
                pos = pymunk.Vec2d(x, y)
                if not track.is_on_track_fast(pos):
                    continue
                along_no, _ = track._project_to_centerline_distance(pos)
                along_ref, _ = track._project_to_centerline_distance(
                    pos, reference_distance=ref
                )
                idx_no = int((along_no / track._centerline_length) * num_checkpoints) % num_checkpoints
                idx_ref = int((along_ref / track._centerline_length) * num_checkpoints) % num_checkpoints
                if idx_no == idx_ref:
                    continue
                fd_no = (along_no - ref) % track._centerline_length
                fd_ref = (along_ref - ref) % track._centerline_length
                if fd_ref < fd_no:
                    found = (pos, last_checkpoint, idx_no, idx_ref)
                    break
            if found:
                break
        if found:
            break

    assert found is not None

    pos, last_checkpoint, idx_no, idx_ref = found
    current, _ = track.get_progress(pos, last_checkpoint)

    assert idx_ref != idx_no
    assert current == idx_ref
