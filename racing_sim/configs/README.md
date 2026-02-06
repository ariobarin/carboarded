# Configuration Guide

## Config Files

| File | Purpose |
|------|---------|
| `default.yaml` | Baseline environment config (simple custom track). Start here. |
| `training_presets.yaml` | Algorithm hyperparameter defaults (PPO and SAC). |
| `custom_tracks/square_test.yaml` | Square test track for the node-based editor. |
| `custom_tracks/simple.yaml` | Simple oval custom track. |
| `custom_tracks/supersimple.yaml` | Minimal custom track for quick testing. |
| `custom_tracks/track1.yaml` | Custom track 1 (node-based). |
| `custom_tracks/track2.yaml` | Custom track 2 (node-based). |
| `custom_tracks/track3.yaml` | Custom track 3 (node-based). |
| `custom_tracks/track4.yaml` | Custom track 4 (node-based). |

## How Config Loading Works

1. YAML files are loaded into Python dataclasses (`EnvConfig`, `CarConfig`, `TrackConfig`, etc.) via `EnvConfig.from_yaml(path)`.
2. CLI arguments in `train.py` override YAML values. For example, `--learning-rate 0.001` overrides the YAML's learning rate.
3. If no config is specified, `EnvConfig.default()` loads `default.yaml`.

## Creating a New Config

1. Copy `default.yaml` to a new file.
2. Modify the parameters you want to change. All fields have sensible defaults, so you only need to include values you want to override.
3. For custom tracks, set `track.track_type: "custom"` and provide node definitions under `track.custom`. See the files in `custom_tracks/` for examples.

## Key Parameters

- **`obs_type`**: `"grid"` (CNN, 36x36 occupancy grid, default) or `"lidar"` (MLP, 9-ray)
- **`progress_reward_scale`**: 0.5-0.75 recommended. Primary convergence driver. Values >= 0.8 cause instability.
- **`collision_penalty`**: Applied once on wall hit when `terminate_on_collision: true`.
- **`random_start`**: Start car at a random checkpoint each episode (helps SAC, hurts PPO).
- **`track.waviness`** / **`track.waves`**: Control track difficulty for elliptical tracks.

See `racing_sim/config/config.py` for the full list of parameters and their defaults.
