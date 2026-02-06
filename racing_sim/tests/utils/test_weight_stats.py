import math

import torch

from racing_sim.utils.weight_stats import summarize_tensors


def test_summarize_tensors_basic_stats():
    tensors = {
        "layer.weight": torch.tensor([1.0, -1.0, 0.0, 2.0]),
        "layer.bias": torch.tensor([0.5, -0.5]),
    }

    rows = summarize_tensors(tensors)
    row_w = next(r for r in rows if r["name"] == "layer.weight")
    row_b = next(r for r in rows if r["name"] == "layer.bias")

    assert row_w["numel"] == 4
    assert math.isclose(row_w["mean"], 0.5, rel_tol=1e-6)
    assert math.isclose(row_w["std"], 1.1180339, rel_tol=1e-6)
    assert math.isclose(row_w["max_abs"], 2.0, rel_tol=1e-6)
    assert math.isclose(row_w["zeros_frac"], 0.25, rel_tol=1e-6)
    assert row_w["is_bias"] is False

    assert row_b["numel"] == 2
    assert math.isclose(row_b["mean"], 0.0, rel_tol=1e-6)
    assert row_b["is_bias"] is True
