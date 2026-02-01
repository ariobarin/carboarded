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

# Best PPO training command - Lidar (247.26 reward on Wavy V2)
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --learning-rate 0.001 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5

# Best PPO training command - CNN (249.43 reward on Wavy V2 -- ALL-TIME BEST)
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42

# Train with random starting positions (SAC only -- PPO fails with random starts)
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --random-start --ent-coef auto --gradient-steps 4 --learning-starts 0

# Fast iteration (skip eval/checkpoints/TensorBoard)
py scripts/train.py --preset fast --no-eval --no-checkpoint --no-tensorboard

# Play/visualize
py scripts/play.py                                    # keyboard control (arrows/WASD, ESC to quit)
py scripts/play.py --model models/path/best_model.zip --deterministic --episodes 10
py scripts/play.py --show-grid --model models/path/best_model.zip  # CNN grid visualization
py scripts/play.py --show-grid --fps 30               # limit to 30 FPS (default: 60)
# In-game keys: G=toggle grid, V=toggle grid debug mode (world-space circles), L=toggle lidar, R=reset

# Validate (headless, 100 episodes)
py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 100 --deterministic

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
- **sensors/grid.py** -- Homographic grid sensor for CNN observation. Projects a 36x36 grid ahead of the car using perspective transform; each cell is 1 (on track) or 0 (off track) via bitmap lookup.
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
| Progress bonus | +scale * normalized_progress | Per step; scale 0.5-0.75 (main convergence driver) |
| Speed bonus | +0.05 * (speed/max_speed) | Per step |
| Collision | -20.0 | Terminates episode |
| Time penalty | -0.0 | Disabled (hurts convergence) |
| Slowdown penalty | disabled (0.0) | Disabled (hurts convergence) |

## Config Files

6 proven configs in `configs/`:
- `fast_iter_v3_complex_progress_0p5.yaml` -- simple ellipse, progress reward 0.5
- `fast_iter_v3_complex_wavy_v1.yaml` -- wavy track (waves=3, waviness=0.06)
- `fast_iter_v3_complex_wavy_v2_progress_0p7.yaml` -- harder wavy track (waves=5, waviness=0.08)
- `wavy_v2_progress_0p75.yaml` -- Wavy V2 optimized (progress 0.75, best PPO lidar config)
- `wavy_v2_progress_0p75_3k_steps.yaml` -- Same as above with max_episode_steps=3000
- `wavy_v2_cnn.yaml` -- Wavy V2 with CNN grid observation (36x36 homographic grid, obs_type=grid)

Deprecated configs are in `configs/deprecated/` with deprecation headers.

## Hyperparameter Reference

| Parameter | Simple Track | Wavy V1 | Wavy V2 |
|-----------|-------------|---------|---------|
| PPO ent_coef | 0.02 | 0.03 | 0.02 (was 0.04) |
| PPO progress_scale | 0.5 | 0.5 | 0.7-0.75 |
| SAC ent_coef | auto | auto | auto |
| SAC gradient_steps | 8 | 4 | 4 |
| SAC batch_size | 256 | 256 | 256 |
| learning_rate | 0.003 | 0.003 | 0.001 (was 0.003) |
| CNN learning_rate | - | - | 0.0003 (lower than lidar) |

## Phase 1 Final Results (Validated)

| Algorithm | Track | Best Reward | Steps | Model |
|-----------|-------|-------------|-------|-------|
| PPO | Simple | **252.49** | 80k | Good Models/Fast Iter V3 Complex Progress0p5.../ |
| PPO | Wavy V1 | **237.57** | 80k | Good Models/Fast Iter V3 Complex Wavy V1.../ |
| PPO | Wavy V2 | **247.26** | 220k | Good Models/PPO Wavy V2 Ent0p02 LR0p001 500k.../ |
| SAC | Wavy V1 | 209.09 | 40k | Good Models/SAC Wavy V1 GradSteps4.../ |
| SAC | Wavy V2 | 183.70 | 90k | Good Models/SAC Wavy V2 Random Start.../ |

*Validated with `validate.py --episodes 100 --deterministic` on 2026-01-30. See `Learnings/Phase One Summary.md` for details.*
*Superseded models (PPO Wavy V2 0.7, 226.49, 243.71, 342.80/3k) archived to `Good Models/_archived/`.*

## Phase 2 Results (CNN Grid Observation)

| Algorithm | Track | Obs Type | Best Reward | Steps | Model |
|-----------|-------|----------|-------------|-------|-------|
| PPO+CNN | Wavy V2 | Grid (36x36) | **249.43** | 220k | Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/ |

*CNN with homographic grid exceeds lidar baseline (247.26) by 1%. See `Learnings/CNN Grid Research.md` for details.*

## Empirical Training Findings

- **Progress reward shaping (0.5-0.75) is the primary convergence accelerator.** Values >= 0.8 cause instability.
- **Entropy scales with track difficulty (PPO):** simple=0.02, wavy V1=0.03, wavy V2=0.02 (lower than expected -- V2 needs less exploration at optimal LR).
- **LR=0.001 is optimal for Wavy V2.** LR=0.003 collapses at 120k, LR=0.001 peaks at 280k. LR=0.0003 is more stable but peaks 20+ points lower.
- **All PPO runs oscillate/collapse.** This is fundamental, not fixable by LR/entropy/batch size. Best model strategy: save snapshots via eval callback, keep the peak.
- **Seed variance is ~12-14 points.** Run multiple seeds and keep the best model for reliable peak performance.
- **SAC requires `--ent-coef auto`** with default target entropy. Fixed entropy values don't learn.
- **SAC gradient_steps scales inversely with difficulty:** 8 for simple tracks, 4 for wavy tracks.
- **SAC curriculum learning (fine-tuning pretrained models on harder tracks) destroys policy.**
- **Random starts help SAC (183.7) but destroy PPO (17.7).** SAC's off-policy replay buffer handles variance.
- **Time penalty, slowdown penalty, collision penalty changes, gamma/GAE tweaks, and clip_range changes all failed.**
- **Combine only one change at a time.** Multi-parameter changes catastrophically interfere.

### CNN-Specific Findings (Phase 2)

- **CNN with homographic grid matches/exceeds lidar performance** (249.43 vs 247.26 on Wavy V2).
- **CNN requires lower LR than lidar:** LR=0.0003 (not 0.001). LR=0.001 causes approx_kl spikes (3.0+) and instability.
- **CNN learns slower than lidar:** First positive reward at ~66k steps vs ~10k for lidar. Catches up by 220k.
- **VecTransposeImage wrapper required for CNN eval env** to match training observation format.
- **Grid configuration:** 36x36, near=30, far=200, FOV=60 degrees, camera pitch=45 degrees.

See `Learnings/CNN Grid Research.md` for the full CNN experiment log.

### Rendering Optimizations

- **Track bitmap cache:** Precomputed occupancy bitmap enables batch `is_on_track()` lookups (0.11ms vs ~2ms per grid).
- **Fast grid overlay:** Corner overlay uses `pygame.surfarray` (1 blit vs 2,592 draw calls). Press V for world-space debug view.
- **Font caching:** Fonts cached at init, not created per frame.
- **DOUBLEBUF:** Display mode uses double buffering for smoother rendering.
- **pygame-ce:** Switched from pygame to pygame-ce for ~14% performance improvement.

See `Learnings/What Didnt Work.md` for the full anti-pattern guide.
See `Learnings/Phase One Summary.md` for the consolidated experiment findings.
See `STANDARDS.md` for experiment protocols and conventions.

## Conventions

- `py` for Python, `uv` for pip (Windows dev environment)
- Configs in YAML loaded via dataclasses; avoid hardcoding env parameters
- `__init__.py` uses `__getattr__` for lazy imports (avoid eager import overhead)
- Training presets (fast/balanced/quality) defined as dicts in `scripts/train.py`
- Saved models go in `Good Models/` with a README.md per folder documenting config, command, and results
- Archived/superseded models go in `Good Models/_archived/`
- Concise imperative commit messages (e.g., `envs: fix checkpoint order`)
- **Default seed in train.py is 42** -- always pass explicit `--seed` for different runs
