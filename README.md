# Racing Sim RL

A 2D autonomous racing environment for training reinforcement learning agents. Built with Gymnasium, Pymunk physics, and Stable-Baselines3.

<!-- TODO: Add a GIF/screenshot of a trained agent driving -->

## Quick Start

```bash
cd racing_sim
uv venv && uv pip install -e .

# Train a PPO agent (~500k steps on the simple custom track)
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --config configs/custom_tracks/simple.yaml \
  --save-freq 100000 --eval-freq 20000 --eval-episodes 1

# Multi-track PPO (simple + track1-4, cycles tracks per episode)
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --config-list configs/custom_tracks/simple.yaml,configs/custom_tracks/track1.yaml,configs/custom_tracks/track2.yaml,configs/custom_tracks/track3.yaml,configs/custom_tracks/track4.yaml \
  --save-freq 100000 --eval-freq 20000 --eval-episodes 1

# Watch it drive (or drive yourself with arrows/WASD)
py scripts/play.py --model models/<run>/best_model.zip --deterministic
```

**Prerequisites:** Python 3.12+, [uv](https://github.com/astral-sh/uv)

## Results (Current Physics)

| Model | Track | Reward | Steps (best) |
|-------|-------|--------|--------------|
| PPO Simple Custom Track - 24.92 Reward | simple | 24.92 | 180k |
| PPO Supersimple Custom Track - 14.00 Reward | supersimple | 14.00 | 140k |
| PPO Square Test Custom Track - 48.85 Reward | square_test | 48.85 | 280k |
| PPO Track1 Custom Track - 24.07 Reward | track1 | 24.07 | 200k |
| PPO Track2 Custom Track - 15.47 Reward | track2 | 15.47 | 140k |
| PPO Track3 Custom Track - 21.03 Reward | track3 | 21.03 | 220k |
| PPO Track4 Custom Track - 15.38 Reward | track4 | 15.38 | 100k |

Each model folder in `Good Models/` includes a config snapshot and training command for reproduction. Model weights are not committed to git (train your own using the provided configs).

## Legacy Results (Legacy Physics)

Legacy scores and configs are documented in `legacy/Good Models/` and `legacy/README.md`.

## Legacy Reproduction

The historical scores and runs documented in `legacy/Good Models/` were produced with the legacy physics implementation. The current `racing_sim/` codebase uses updated physics and will not reproduce those scores even with the same configs.

This repo includes a clearly labeled legacy snapshot under `legacy/`. To reproduce historical results, follow `legacy/README.md` and run training from `legacy/racing_sim`.

## How It Works

Each step: the agent sends `[steering, throttle]` to the car, Pymunk simulates physics (lateral friction, speed-dependent steering), sensors observe the result, and a reward is computed.

**Observation types (default: grid):**
- **Grid (CNN)** -- A 36x36 binary occupancy grid projected ahead of the car using a perspective transform. Nearby cells are dense, distant cells are sparse. Processed by a CNN feature extractor (NatureCNN). Default for all configs.
- **Lidar** -- 9 rays in a forward arc, each returning a normalized wall distance [0,1]. Simple, fast, works with MLP policies. Used by legacy/Good Models snapshots.

**Reward structure:**

| Component | Value | Notes |
|-----------|-------|-------|
| Checkpoint | +1.0 | Crossing next checkpoint in sequence |
| Progress bonus | +scale * normalized_progress | Per step; scale 0.5-0.75 (primary convergence driver) |
| Speed bonus | +0.05 * (speed/max_speed) | Per step |
| Collision | -20.0 | Terminates episode |

## Configuration

All configs live in `racing_sim/configs/`:
- `default.yaml` -- baseline environment config (start here)
- `training_presets.yaml` -- algorithm hyperparameter defaults
- `custom_tracks/*.yaml` -- example custom tracks for the node-based editor

Configs are YAML files loaded into dataclasses. CLI arguments override YAML values. See `racing_sim/configs/README.md` for details.

## Track Editor

Create custom tracks with the visual node-based editor:

```bash
py scripts/create_track.py                            # launch editor
py scripts/create_track.py --load configs/custom_tracks/square_test.yaml  # edit existing

# Editor controls: Click=add/select, Drag=move, Right-click=delete
# G=grid snap, P=preview, Ctrl+S=save, Ctrl+Z/Y=undo/redo
```

Train and play on custom tracks:

```bash
py scripts/train.py --config configs/custom_tracks/square_test.yaml --total-timesteps 100000
py scripts/play.py --config configs/custom_tracks/square_test.yaml
```

## Project Structure

```
legacy/
  racing_sim/           # Legacy physics snapshot for reproduction
  Good Models/          # Legacy model READMEs + config snapshots
racing_sim/
  racing_sim/           # Python package
    envs/               # Gymnasium environment
    physics/            # Car dynamics + track generation (Pymunk)
    sensors/            # Lidar raycasting + grid projection
    editor/             # Node-based track editor
    rendering/          # PyGame visualization
    config/             # Dataclass configs, YAML loading
    policies/           # Custom policy networks (dropout)
    utils/              # Reward, progress helpers
  scripts/              # train.py, play.py, validate.py, physics tools
  configs/              # YAML environment configs
  tests/                # pytest suite
Good Models/            # Per-model READMEs + config snapshots
Learnings/              # Experiment summaries and research docs
```

## Experiment Findings

Several practical insights from the legacy physics experiments:

1. **Progress reward shaping (0.5-0.75) is the primary convergence driver.** Without it, agents struggle to learn directional movement. Values above 0.8 cause instability.

2. **All PPO runs eventually collapse.** Performance peaks then drops, often permanently. This is fundamental to PPO on this task, not a bug. The strategy is to save checkpoints frequently and keep the peak.

3. **L2 regularization enables higher peaks.** Adding `weight_decay=0.0001` to the optimizer is the only method (out of four tested) that improved peak performance. It does not prevent collapse, but the peaks are higher before it happens.

4. **CNN grid observation slightly outperforms lidar** on the hardest track (252.60 vs 247.26) but requires lower learning rates (0.0003 vs 0.001) and learns slower initially.

5. **SAC and PPO have complementary strengths.** SAC handles random starting positions and is more sample-efficient on simple tracks. PPO reaches higher peaks on difficult tracks. SAC curriculum learning (fine-tuning across tracks) destroys the policy.

6. **Seed variance accounts for 12-14 points.** Always run multiple seeds and keep the best model for reliable results.

See `Learnings/` for detailed experiment logs and `Learnings/What Didnt Work.md` for the full anti-pattern guide.

## Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Developer reference: commands, architecture, config reference |
| [STANDARDS.md](STANDARDS.md) | Experiment protocols, model saving conventions |
| [Learnings/README.md](Learnings/README.md) | Index of all experiment documents |
| [Learnings/Glossary.md](Learnings/Glossary.md) | RL/ML term definitions |
| [Good Models/README.md](Good%20Models/README.md) | Trained model index (current physics) |
| [legacy/Good Models/README.md](legacy/Good%20Models/README.md) | Trained model index (legacy physics) |

## Scripts Reference

| Script | Description |
|--------|-------------|
| `train.py` | Train PPO or SAC agents with configurable hyperparameters |
| `play.py` | Visualize a trained model or drive manually (arrows/WASD) |
| `validate.py` | Headless evaluation over N episodes |
| `create_track.py` | Visual node-based track editor |
| `physics_tuner.py` | Interactive physics parameter tuning |
| `physics_probe.py` | Log speed/accel/position to CSV for a given scenario |
| `physics_sweep.py` | Compare multiple configs across physics scenarios |

## Future Work

- **Harder tracks:** Waviness > 0.08, more waves, narrower width
- **Domain randomization:** Lidar noise, friction variation, action delay for sim-to-real transfer
- **Deployment:** ONNX export, model quantization for edge inference
- **Sim-to-real bridge:** Homography-warped camera feed on a physical car
- **Collapse mitigation:** Population-based training, periodic policy distillation, or ensemble methods
