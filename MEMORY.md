# MEMORY.md

Persistent notes for cross-session continuity.

## 2026-02-06: Default observation switched from lidar to grid

All active YAML configs, the `from_yaml()` fallback in `config.py`, and documentation
now default to `obs_type: grid` (36x36 CNN). Lidar (9-ray MLP) is still supported
but must be set explicitly. Good Models configs are unchanged (historical snapshots
trained with lidar). Redundant `track3_grid*.yaml` configs were deleted.
