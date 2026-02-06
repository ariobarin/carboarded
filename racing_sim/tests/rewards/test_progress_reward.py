import math

import pytest

from racing_sim.utils.progress import progress_delta, progress_delta_cyclic


def test_progress_delta_forward():
    assert progress_delta(0.0, 0.5) == 0.5


def test_progress_delta_wraparound():
    delta = progress_delta(6.2, 0.1)
    assert delta == pytest.approx(0.1 - 6.2 + math.tau, rel=1e-6)


def test_progress_delta_backward():
    delta = progress_delta(0.5, 0.1)
    assert delta < 0


def test_progress_delta_cyclic_wraparound():
    assert progress_delta_cyclic is not None
    delta = progress_delta_cyclic(9.5, 0.5, 10.0)
    assert delta == pytest.approx(1.0)
