"""Utility helpers for racing_sim."""

__all__ = ["detect_algo_from_model", "load_model"]


def __getattr__(name):
    if name in ("detect_algo_from_model", "load_model", "infer_obs_type"):
        from racing_sim.utils import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
