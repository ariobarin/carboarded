from racing_sim.utils.off_track import compute_off_track_state


def test_on_track_resets_counter_and_no_penalty():
    steps, penalty, terminated = compute_off_track_state(
        on_track=True,
        collided=False,
        prev_off_track_steps=3,
        off_track_penalty=-5.0,
        max_off_track_steps=10,
    )

    assert steps == 0
    assert penalty == 0.0
    assert terminated is False


def test_off_track_increments_and_penalizes():
    steps, penalty, terminated = compute_off_track_state(
        on_track=False,
        collided=False,
        prev_off_track_steps=1,
        off_track_penalty=-5.0,
        max_off_track_steps=10,
    )

    assert steps == 2
    assert penalty == -10.0  # penalty scales with steps off track
    assert terminated is False


def test_collision_counts_as_off_track_when_allowed():
    steps, penalty, terminated = compute_off_track_state(
        on_track=True,
        collided=True,
        prev_off_track_steps=0,
        off_track_penalty=-3.0,
        max_off_track_steps=5,
    )

    assert steps == 1
    assert penalty == -3.0
    assert terminated is False


def test_off_track_terminates_after_limit():
    steps, penalty, terminated = compute_off_track_state(
        on_track=False,
        collided=False,
        prev_off_track_steps=4,
        off_track_penalty=-1.0,
        max_off_track_steps=5,
    )

    assert steps == 5
    assert penalty == -5.0
    assert terminated is True
