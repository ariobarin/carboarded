# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `racing_sim/`:

```bash
# Setup
uv venv && uv pip install -e .
uv pip install -e ".[dev]"    # includes pytest

# Tests
py -m pytest tests/                              # all tests
py -m pytest tests/physics/test_car_physics.py   # single file
py -m pytest tests/ -k "test_checkpoint"         # by name pattern
py -m pytest tests/ --co -q                      # list tests without running

# Training
py scripts/train.py --algo ppo --total-timesteps 500000 --config configs/default.yaml
py scripts/train.py --algo sac --total-timesteps 200000 --config configs/default.yaml

# Playback / manual driving
py scripts/play.py --model models/<run>/best_model.zip --deterministic
py scripts/play.py --config configs/default.yaml   # keyboard control (arrows/WASD)

# Headless validation
py scripts/validate.py --model MODEL --config CONFIG --episodes 1 --deterministic

# Track editor
py scripts/create_track.py
py scripts/create_track.py --load configs/custom_tracks/square_test.yaml

# Physics tools
py scripts/physics_probe.py   # log speed/accel to CSV
py scripts/physics_sweep.py   # compare configs across scenarios
py scripts/physics_tuner.py   # interactive parameter tuning
```

CLI flags override YAML config values. `--config` defaults to `configs/default.yaml`. Training presets (algo hyperparameters) live in `configs/training_presets.yaml`.

## Architecture

### Simulation loop (per step)
Agent sends `[steering, throttle]` -> `Car.apply_control()` + `Car.update_friction()` -> `pymunk.Space.step()` -> `Lidar.scan()` (always) + optionally `compute_grid()` -> `_calculate_reward()` -> return `(obs, reward, terminated, truncated, info)`.

### Two observation modes
- **Lidar** (`obs_type: lidar`): 9-ray forward arc, normalized [0,1] floats. Works with MLP policies.
- **Grid** (`obs_type: grid`): 36x36 binary occupancy grid via perspective projection. Shape `(36,36,1)` uint8. Requires CNN policy (NatureCNN). Slightly outperforms lidar on hard tracks.

### Track types
- **Elliptical** (`track_type: elliptical`): Parametric oval with optional waviness. Implemented in `physics/track.py` (`Track` class).
- **Custom** (`track_type: custom`): Node-based closed circuit with fillet arcs. Implemented in `editor/node_track.py` (`NodeTrack` class). Both expose the same interface: `checkpoints`, `get_progress()`, `progress_coordinate()`, `is_on_track()`, `get_spawn_position()`.

### Config system
`EnvConfig` dataclass (`config/config.py`) nests `CarConfig`, `LidarConfig`, `TrackConfig`, `GridConfig`, `RenderConfig`. Loaded from YAML via `EnvConfig.from_yaml()`. The canonical default is `configs/default.yaml`, resolved at runtime through `config/defaults.py`. `EnvConfig.default()` loads from YAML (not hardcoded). CLI arguments override YAML values.

### Reward structure
Checkpoint (per checkpoint crossed) + progress shaping (continuous, scaled by `progress_reward_scale`) + speed bonus (* speed_ratio) + collision penalty (terminates episode). Progress reward uses `progress_delta_cyclic()` for wrap-around arithmetic.

### Physics
Pymunk 2D space, top-down (no gravity). Car uses lateral friction model with speed-dependent steering, air drag, acceleration boost, and throttle smoothing. Collision detection via `pre_solve` callback sets `car.touching_wall` per step; `begin` callback sets `car.collided` (sticky). `sensor_only` mode detects but skips physics response.

### Key module boundaries
- `physics/car.py`: Body/shape creation, control application, friction model, collision handling
- `physics/track.py` / `editor/node_track.py`: Wall geometry, checkpoint generation, progress tracking, bitmap for `is_on_track()`
- `sensors/lidar.py`: Pymunk raycasting against walls
- `sensors/grid.py`: Perspective-projected occupancy grid (vectorized numpy + per-cell `is_on_track()`)
- `utils/progress.py`: Cyclic delta math for progress reward
- `utils/reward.py`: Slowdown penalty helper (currently disabled)
- `policies/`: Custom SB3 policy networks (LayerNorm, Dropout variants)
- `scripts/train.py`: Builds SB3 model (PPO/SAC), sets up eval callbacks, loads presets from `training_presets.yaml`, applies CLI overrides

## Project conventions

- Use `uv` for pip, `py` for python.
- No emoji in code or documentation.
- Tests use pytest. Organized into subdirs: `cli/`, `config/`, `editor/`, `env/`, `physics/`, `policies/`, `rendering/`, `rewards/`, `smoke/`, `tracks/`, `utils/`.
- Experiment protocol: change one hyperparameter at a time, validate before saving results. See `STANDARDS.md`.
- `progress_reward_scale` must stay in 0.5-0.75 range; values >= 0.8 cause instability.
- PPO runs eventually collapse; save checkpoints frequently and keep the peak.
- SAC: use `--ent-coef auto`; fixed entropy prevents learning. Do not fine-tune SAC across tracks.
- `legacy/` contains a frozen physics snapshot for reproducing historical results; current codebase uses updated physics.
