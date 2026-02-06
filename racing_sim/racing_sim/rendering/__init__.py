"""Rendering components."""

__all__ = ["Renderer"]


def __getattr__(name):
    if name == "Renderer":
        from racing_sim.rendering.renderer import Renderer
        return Renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
