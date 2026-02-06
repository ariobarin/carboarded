"""Custom callbacks for RL training."""

__all__ = [
    "GradStatsCallback",
    "LogStdClampCallback",
    "RolloutStatsCallback",
    "ShrinkPerturbCallback",
]


def __getattr__(name):
    if name == "ShrinkPerturbCallback":
        from racing_sim.callbacks.plasticity import ShrinkPerturbCallback
        return ShrinkPerturbCallback
    if name == "GradStatsCallback":
        from racing_sim.callbacks.grad_stats import GradStatsCallback
        return GradStatsCallback
    if name == "LogStdClampCallback":
        from racing_sim.callbacks.log_std_clamp import LogStdClampCallback
        return LogStdClampCallback
    if name == "RolloutStatsCallback":
        from racing_sim.callbacks.rollout_stats import RolloutStatsCallback
        return RolloutStatsCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
