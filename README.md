# Racing Sim RL

A 2D autonomous racing environment for training reinforcement learning agents. Built with Gymnasium, Pymunk physics, and Stable-Baselines3.

## Highlights

- **252.60 best reward** on the hardest track (Wavy V2) using PPO with a CNN grid observation and L2 regularization
- Two observation types: 9-ray lidar and 36x36 homographic grid (CNN)
- Two algorithms: PPO (best performance) and SAC (supports random starts)
- Three track difficulties with configurable waviness
- Real-time visualization with lidar/grid debug overlays

## Getting Started

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

```bash
cd racing_sim
uv venv && uv pip install -e .
```

**Train a model** (Physics v2 baseline, ~5-10 min on a modern CPU):

```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --config configs/physics_v2.yaml \
  --learning-rate 0.003 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

**Watch it drive:**

```bash
# Keyboard control (arrows/WASD, ESC to quit)
py scripts/play.py

# Run a trained model
py scripts/play.py --model "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/best_model.zip" \
  --config "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/config.yaml" --deterministic --episodes 5

# In-game keys: G=toggle grid, V=grid debug view, L=toggle lidar, R=reset
```

**Physics tuning:**

```bash
# Interactive tuning on a simple ellipse
py scripts/physics_tuner.py --config configs/physics_v2.yaml

# Log speed/accel/position to CSV (headless)
py scripts/physics_probe.py --config configs/physics_v2.yaml --scenario straight_full_throttle --steps 600

# Sweep multiple configs + scenarios to a summary CSV
py scripts/physics_sweep.py --configs configs/physics_v2.yaml \
  configs/deprecated/physics_v2_candidates/physics_v2_f1_grip.yaml \
  configs/deprecated/physics_v2_candidates/physics_v2_training_safe.yaml \
  --scenario all --steps 600
```

**Validate headless:**

```bash
py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 1 --deterministic
```

## Results

All models validated with 100 deterministic episodes.

| Phase | Algorithm | Track | Obs Type | Best Reward | Steps |
|-------|-----------|-------|----------|-------------|-------|
| 3 | PPO+CNN+L2 | Wavy V2 | Grid 36x36 | **252.60** | 120k |
| 2 | PPO+CNN | Wavy V2 | Grid 36x36 | 249.43 | 220k |
| 1 | PPO | Simple | Lidar 9-ray | 252.49 | 80k |
| 1 | PPO | Wavy V2 | Lidar 9-ray | 247.26 | 220k |
| 1 | PPO | Wavy V1 | Lidar | 237.57 | 80k |
| 1 | SAC | Wavy V1 | Lidar | 209.09 | 40k |
| 1 | SAC | Wavy V2 | Lidar | 183.70 | 90k |

Trained model weights are in `Good Models/` -- see [Good Models/README.md](Good%20Models/README.md) for the full index.
Legacy results use pre-physics_v2 configs; each model folder includes a `config.yaml` snapshot for reproduction.

## Observation Types

**Lidar** -- 9 rays cast from the car in a forward arc. Each returns a normalized distance [0,1] to the nearest wall. Simple, fast, and effective. Works with standard MLP policies.

**Grid (CNN)** -- A 36x36 binary grid projected ahead of the car using a perspective (homographic) transform. Each cell is 1 (on track) or 0 (off track). Nearby cells are dense; distant cells are sparse, mimicking natural perspective. Processed by a CNN feature extractor. Slightly outperforms lidar on the hardest track.

## Track Variants

| Track | Config | Difficulty | Description |
|-------|--------|------------|-------------|
| Physics v2 (Simple) | `physics_v2.yaml` | Easy | New physics baseline |
| Legacy Simple | `configs/deprecated/legacy_2026_02/fast_iter_v3_complex_progress_0p5.yaml` | Easy | Smooth ellipse (legacy) |
| Legacy Wavy V1 | `configs/deprecated/legacy_2026_02/fast_iter_v3_complex_wavy_v1.yaml` | Medium | 3 waves, waviness=0.06 |
| Legacy Wavy V2 | `configs/deprecated/legacy_2026_02/wavy_v2_progress_0p75.yaml` | Hard | 5 waves, waviness=0.08 |
| Legacy Wavy V2 CNN | `configs/deprecated/legacy_2026_02/wavy_v2_cnn.yaml` | Hard | Same track, grid observation |

All configs live in `racing_sim/configs/`. Legacy configs are in `racing_sim/configs/deprecated/legacy_2026_02/`.
Experimental physics v2 candidates (UNVALIDATED) are in `racing_sim/configs/deprecated/physics_v2_candidates/`.

## Project Structure

```
racing_sim/
  racing_sim/           # Python package
    envs/               # Gymnasium environment
    physics/            # Car dynamics + track generation (Pymunk)
    sensors/            # Lidar raycasting + grid projection
    rendering/          # PyGame visualization
    config/             # Dataclass configs, YAML loading
    utils/              # Reward, progress helpers
  scripts/              # train.py, play.py, validate.py
  configs/              # YAML environment configs
  tests/                # pytest suite
Good Models/            # Trained model weights + per-model READMEs
Learnings/              # Experiment summaries and research docs
CLAUDE.md               # Developer reference (commands, architecture, hyperparameters)
STANDARDS.md            # Experiment protocols and conventions
```

## Documentation Guide

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Commands, architecture, hyperparameter tables, all empirical findings |
| [STANDARDS.md](STANDARDS.md) | Experiment protocols, model saving conventions, config management |
| [Learnings/README.md](Learnings/README.md) | Index of all experiment documents |
| [Learnings/Glossary.md](Learnings/Glossary.md) | RL/ML term definitions for newcomers |
| [Good Models/README.md](Good%20Models/README.md) | Trained model index with results |

## Key Findings

Three phases of experiments produced several practical insights:

1. **Progress reward shaping (0.5-0.75) is the primary convergence driver.** Without it, agents struggle to learn directional movement. Values above 0.8 cause instability.

2. **All PPO runs eventually collapse.** Performance peaks then drops, often permanently. This is fundamental to PPO on this task, not a bug. The strategy is to save checkpoints frequently and keep the peak.

3. **L2 regularization enables higher peaks.** Adding `weight_decay=0.0001` to the optimizer is the only method (out of four tested) that improved peak performance. It does not prevent collapse, but the peaks are higher before it happens.

4. **CNN grid observation slightly outperforms lidar** on the hardest track (252.60 vs 247.26) but requires lower learning rates (0.0003 vs 0.001) and learns slower initially.

5. **SAC and PPO have complementary strengths.** SAC handles random starting positions and is more sample-efficient on simple tracks. PPO reaches higher peaks on difficult tracks. SAC curriculum learning (fine-tuning across tracks) destroys the policy.

6. **Seed variance accounts for 12-14 points.** Always run multiple seeds and keep the best model for reliable results.

See `Learnings/` for detailed experiment logs and `Learnings/What Didnt Work.md` for the full anti-pattern guide.

## Future Work

- **Robustness:** Multi-seed validation, sensitivity analysis, failure mode documentation
- **Harder tracks:** Waviness > 0.08, more waves, narrower width
- **Domain randomization:** Lidar noise, friction variation, action delay for sim-to-real transfer
- **Deployment:** ONNX export, model quantization for edge inference
- **Sim-to-real bridge:** Homography-warped camera feed on physical car with steering/throttle control
- **Collapse mitigation:** Explore population-based training, periodic policy distillation, or ensemble methods
