import math

import importlib.util
from pathlib import Path
import pytest


def load_progress_delta():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "racing_sim" / "utils" / "progress.py"
    spec = importlib.util.spec_from_file_location("racing_sim_progress", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.progress_delta


progress_delta = load_progress_delta()


def test_progress_delta_forward():
    assert progress_delta(0.0, 0.5) == 0.5


def test_progress_delta_wraparound():
    delta = progress_delta(6.2, 0.1)
    assert delta == pytest.approx(0.1 - 6.2 + math.tau, rel=1e-6)


def test_progress_delta_backward():
    delta = progress_delta(0.5, 0.1)
    assert delta < 0
