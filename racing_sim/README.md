# Racing Sim

2D autonomous racing car simulation using Gymnasium, Pymunk physics, PyGame rendering, and Stable Baselines 3 for training PPO/SAC agents.

## Features

- **Realistic car physics** with lateral friction (tire grip) preventing sideways sliding
- **5-ray lidar sensor** at [-60, -30, 0, 30, 60] degrees for obstacle detection
- **Oval track** with inner/outer walls and checkpoint-based progress tracking
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
# Train with PPO (fast preset, default)
py scripts/train.py

# Train with SAC
py scripts/train.py --algo sac

# Longer run
py scripts/train.py --preset quality --total-timesteps 500000

# Fast iteration (no eval/checkpoints/tensorboard)
py scripts/train.py --preset fast --no-eval --no-checkpoint --no-tensorboard

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
py scripts/play.py --model models/ppo_20260121_223632/ppo_final

# Run multiple episodes
py scripts/play.py --model models/ppo_final --episodes 10
```

**Controls:**
- Arrow keys or WASD: Steer and accelerate
- ESC: Quit

## Environment Details

### Observation Space
- `Box(0, 1, shape=(5,))` - 5 normalized lidar distances (0 = hit close, 1 = no hit)

### Action Space
- `Box([-1, 0], [1, 1])` - [steering, throttle]
  - Steering: -1 (left) to 1 (right)
  - Throttle: 0 to 1

### Reward Function
| Component | Value | Description |
|-----------|-------|-------------|
| Checkpoint | +1.0 | Per checkpoint passed |
| Speed bonus | +0.1 * (speed/max_speed) | Encourages going fast |
| Collision | -10.0 | Wall hit (terminates episode) |
| Time penalty | -0.1 | Per step (encourages efficiency) |

## Configuration

Edit `configs/default.yaml` (used by default) or create a custom config:

```yaml
car:
  max_speed: 1000.0
  engine_power: 500.0
  lateral_friction: 0.9

lidar:
  num_rays: 5
  ray_angles: [-60.0, -30.0, 0.0, 30.0, 60.0]
  max_distance: 200.0

max_episode_steps: 1000
```

Use with:
```bash
py scripts/train.py --config configs/my_config.yaml
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
│   └── play.py               # Visualization script
└── configs/default.yaml      # Default parameters
```

## Dependencies

- gymnasium
- pymunk
- pygame
- stable-baselines3
- numpy
- pyyaml
- tensorboard
