"""Track editor components for visual track creation."""

__all__ = [
    "compute_fillet",
    "offset_line",
    "offset_arc",
    "discretize_arc",
    "point_to_segment_distance",
    "point_to_arc_distance",
    "line_line_intersection",
    "segments_intersect",
    "NodeTrack",
    "validate_track",
    "ValidationResult",
    "EditorState",
    "EditorMode",
    "EditorRenderer",
]


def __getattr__(name):
    """Lazy import for editor components."""
    if name in (
        "compute_fillet",
        "offset_line",
        "offset_arc",
        "discretize_arc",
        "point_to_segment_distance",
        "point_to_arc_distance",
        "line_line_intersection",
        "segments_intersect",
        "CenterlineElement",
        "FilletResult",
    ):
        from racing_sim.editor import geometry
        return getattr(geometry, name)

    if name == "NodeTrack":
        from racing_sim.editor.node_track import NodeTrack
        return NodeTrack

    if name in ("validate_track", "ValidationResult"):
        from racing_sim.editor import validation
        return getattr(validation, name)

    if name in ("EditorState", "EditorMode"):
        from racing_sim.editor import editor_state
        return getattr(editor_state, name)

    if name == "EditorRenderer":
        from racing_sim.editor.editor_renderer import EditorRenderer
        return EditorRenderer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
