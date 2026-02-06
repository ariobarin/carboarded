"""2D autonomous racing car simulation with Lidar and RL."""

__version__ = "0.1.0"
__all__ = ["RacingEnv"]


def __getattr__(name: str):
    if name == "RacingEnv":
        from racing_sim.envs.racing_env import RacingEnv
        return RacingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
