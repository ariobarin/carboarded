# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the `racing_sim/` directory unless noted otherwise.

```bash
# Install
uv venv && uv pip install -e .

# If uv can't find the venv:
uv pip install <package> --python "C:\Desktop\Repos\fine-ill-do-it-myself\racing_sim\.venv\Scripts\python.exe"

# Train (PPO default)
py scripts/train.py --preset fast --total-timesteps 200000
py scripts/train.py --algo sac --preset balanced --total-timesteps 300000
py scripts/train.py --config configs/fast_iter_v3_complex_wavy_v1.yaml --learning-rate 0.003 --ent-coef 0.03

# Train with random starting positions (improves exploration/robustness)
py scripts/train.py --algo sac --preset fast --total-timesteps 80000 --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --random-start --ent-coef auto --gradient-steps 4 --learning-starts 0

# Fast iteration (skip eval/checkpoints/TensorBoard)
py scripts/train.py --preset fast --no-eval --no-checkpoint --no-tensorboard

# Play/visualize
py scripts/play.py                                    # keyboard control (arrows/WASD, ESC to quit)
py scripts/play.py --model models/path/best_model.zip --deterministic --episodes 10

# TensorBoard
tensorboard --logdir logs

# Tests
pytest
pytest tests/test_env_config_defaults.py              # single test file
pytest --cov=racing_sim

# Environment sanity check
py -c "from gymnasium.utils.env_checker import check_env; from racing_sim import RacingEnv; check_env(RacingEnv())"
```

## Architecture

2D racing RL environment built on Gymnasium + Pymunk + Stable-Baselines3.

**Data flow per step:** RacingEnv.step(action) -> Car applies steering/throttle -> Pymunk physics step -> Lidar raycasts -> compute reward -> return (obs, reward, done, info)

**Core components** (all under `racing_sim/racing_sim/`):
- **envs/racing_env.py** -- Gymnasium env orchestrating everything. Obs: N normalized lidar distances [0,1]. Action: [steering, throttle].
- **physics/car.py** -- Pymunk rigid body with lateral friction (prevents sliding). Speed-dependent steering (30% at standstill, 100% at speed >= 100).
- **physics/track.py** -- Elliptical track with static wall segments (64 each for inner/outer). Supports waviness parameters for difficulty variation. 64 checkpoint lines for progress tracking.
- **sensors/lidar.py** -- Pymunk raycast sensor. Configurable ray count/angles/max distance. Filters car's own shape via ShapeFilter.
- **rendering/renderer.py** -- PyGame visualization with lidar debug overlay, HUD stats.
- **config/config.py** -- Dataclass-based config (EnvConfig, CarConfig, LidarConfig, TrackConfig, RenderConfig). Load from YAML via `EnvConfig.from_yaml(path)`.
- **utils/reward.py** -- `compute_slowdown_penalty()` helper.
- **utils/progress.py** -- `progress_delta()` angular progress tracker.

**Collision types:** Car=1 (0b0001), Wall=2 (0b0010). Detection via `space.on_collision()` (pymunk 7.x API).

**Training pipeline** (`scripts/train.py`): Parses CLI args, loads config YAML, creates vectorized envs, configures SB3 PPO/SAC with preset hyperparameters, runs training with eval callbacks. Outputs to timestamped `logs/` and `models/` directories.

**Analysis pipeline** (`analysis/`): `evaluate_agents.py` batch-evaluates models defined in YAML agent specs. `tune_rule_based.py` grid-searches a rule-based baseline. `make_plots.py` and `plot_eval.py` generate comparison charts.

## Reward Structure

| Component | Value | Notes |
|-----------|-------|-------|
| Checkpoint | +1.0 | Crossing next checkpoint in sequence |
| Progress bonus | +scale * normalized_progress | Per step; scale 0.5-0.7 (main convergence driver) |
| Speed bonus | +0.1 * (speed/max_speed) | Per step |
| Slowdown penalty | -(scale * speed_ratio * closeness) | Near walls at speed |
| Collision | -10.0 | Terminates episode |
| Time penalty | -0.1 | Per step (hurts convergence -- avoid increasing) |

## Empirical Training Findings

These findings from Phase 1 experiments should guide hyperparameter choices:

- **Progress reward shaping (0.5-0.7) is the primary convergence accelerator.** Values above 0.8 cause instability.
- **Entropy scales with track difficulty:** simple=0.02, wavy V1 (waves=3)=0.03, wavy V2 (waves=5)=0.04.
- **SAC requires `--ent-coef auto`** with default target entropy. Fixed entropy values don't learn.
- **SAC gradient_steps scales inversely with difficulty:** 8 for simple tracks, 4 for wavy tracks.
- **SAC curriculum learning (fine-tuning pretrained models on harder tracks) destroys policy.**
- **Time penalty, higher collision penalties, gamma/GAE tweaks, and clip_range changes all failed to improve results.**
- **Random starting positions (`--random-start`) improves exploration** but makes training harder. On Wavy V2 (80K steps): SAC peaked at 52.16 (vs PPO's 22.06), but both show mid-training instability. Best model callback captures peak performance. Recommend longer training (150K+) with random start for robust policies.

## Config Files

YAML configs under `configs/` define track geometry, reward settings, and sensor parameters. Key ones:
- `default.yaml` -- baseline reference (9 rays, 400px lidar)
- `fast_iter_v3_complex_progress_0p5.yaml` -- simple ellipse, progress reward 0.5
- `fast_iter_v3_complex_wavy_v1.yaml` -- wavy track (waves=3, waviness=0.06)
- `fast_iter_v3_complex_wavy_v2_progress_0p7.yaml` -- harder wavy track (waves=5, waviness=0.08)
- `fast_iter_v3_complex_sac_bootstrap.yaml` -- SAC-optimized variant

## Conventions

- `py` for Python, `uv` for pip (Windows dev environment)
- Configs in YAML loaded via dataclasses; avoid hardcoding env parameters
- `__init__.py` uses `__getattr__` for lazy imports (avoid eager import overhead)
- Training presets (fast/balanced/quality) defined as dicts in `scripts/train.py`
- Saved models go in `Good Models/` with a README.md per folder documenting config, command, and results
