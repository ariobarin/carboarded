# Racing Sim

2D autonomous racing car simulation using Gymnasium, Pymunk physics, PyGame rendering, and Stable Baselines 3 for training PPO/SAC agents.

## Features

- **Realistic car physics** with lateral friction (tire grip) preventing sideways sliding
- **9-ray lidar sensor** for obstacle detection (configurable angles and range)
- **Wavy oval tracks** with inner/outer walls and checkpoint-based progress tracking
- **Gymnasium-compliant** environment for RL training
- **PPO and SAC** training with TensorBoard logging

## Installation

```bash
cd racing_sim
uv venv
uv pip install -e .
uv pip install tqdm rich  # for progress bar
```

Or with pip:
```bash
pip install -e .
pip install tqdm rich
```

## Usage

### Train an Agent

```bash
# Train with PPO on physics_v2 baseline
py scripts/train.py --algo ppo --preset fast --total-timesteps 50000 \
  --config configs/physics_v2.yaml

# Train with SAC on legacy Wavy V2 (deprecated config)
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config configs/deprecated/legacy_2026_02/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --learning-rate 0.003 --ent-coef auto --gradient-steps 4

# More options
py scripts/train.py --help
```

Training logs are saved to `logs/` and models to `models/`.

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

### Play / Visualize

```bash
# Manual keyboard control
py scripts/play.py

# Visualize a trained model
py scripts/play.py --model models/ppo_final.zip --config configs/physics_v2.yaml

# Run multiple episodes deterministically
py scripts/play.py --model models/ppo_final.zip --episodes 10 --deterministic
```

**Controls:**
- Arrow keys or WASD: Steer and accelerate
- ESC: Quit

### Physics Tuning

```bash
# Interactive tuner on a simple ellipse track
py scripts/physics_tuner.py --config configs/physics_v2.yaml

# Log speed/accel/position to CSV (headless)
py scripts/physics_probe.py --config configs/physics_v2.yaml --scenario straight_full_throttle --steps 600

# Sweep multiple configs + scenarios to a summary CSV
py scripts/physics_sweep.py --configs configs/physics_v2.yaml \
  configs/deprecated/physics_v2_candidates/physics_v2_f1_grip.yaml \
  configs/deprecated/physics_v2_candidates/physics_v2_training_safe.yaml \
  --scenario all --steps 600
```

### Validate (Headless)

```bash
py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 1 --deterministic
```

## Environment Details

### Observation Space
- `Box(0, 1, shape=(9,))` - 9 normalized lidar distances (0 = hit close, 1 = no hit)

### Action Space
- `Box([-1, 0], [1, 1])` - [steering, throttle]
  - Steering: -1 (left) to 1 (right)
  - Throttle: 0 to 1

### Reward Function
| Component | Value | Description |
|-----------|-------|-------------|
| Checkpoint | +1.0 | Per checkpoint passed |
| Progress bonus | +scale * progress | Per step (scale 0.5-0.75) |
| Speed bonus | +0.05 * (speed/max_speed) | Encourages going fast |
| Collision | -20.0 | Wall hit (terminates episode) |

## Configuration

Physics v2 baseline config in `configs/`:
- `physics_v2.yaml` -- simple ellipse, new physics defaults

Legacy configs are in `configs/deprecated/legacy_2026_02/` (pre-physics_v2):
- `fast_iter_v3_complex_progress_0p5.yaml`
- `fast_iter_v3_complex_wavy_v1.yaml`
- `fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- `wavy_v2_progress_0p75.yaml`
- `wavy_v2_progress_0p75_3k_steps.yaml`
- `wavy_v2_cnn.yaml`

Experimental physics v2 candidates (UNVALIDATED) are in:
- `configs/deprecated/physics_v2_candidates/`

Use with:
```bash
py scripts/train.py --config configs/physics_v2.yaml
```

## Project Structure

```
racing_sim/
├── racing_sim/
│   ├── config/config.py      # Dataclass configurations
│   ├── envs/racing_env.py    # Gymnasium environment
│   ├── physics/
│   │   ├── car.py            # Car with lateral friction
│   │   └── track.py          # Oval track geometry
│   ├── sensors/lidar.py      # Raycast lidar sensor
│   └── rendering/renderer.py # PyGame renderer
├── scripts/
│   ├── train.py              # SB3 training script
│   ├── play.py               # Visualization script
│   └── validate.py           # Headless validation
└── configs/                  # physics_v2.yaml + deprecated/
```

## Dependencies

- gymnasium
- pymunk
- pygame
- stable-baselines3
- numpy
- pyyaml
- tensorboard
