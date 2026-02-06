# Racing Sim

2D autonomous racing car simulation using Gymnasium, Pymunk physics, PyGame rendering, and Stable Baselines 3 for training PPO/SAC agents.

For full documentation, see the [project README](../README.md).

## Installation

```bash
cd racing_sim
uv venv && uv pip install -e .
```

## Quick Usage

```bash
# Train
py scripts/train.py --algo ppo --total-timesteps 200000 --config configs/default.yaml

# Multi-track PPO (cycles tracks per episode)
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --config-list configs/custom_tracks/simple.yaml,configs/custom_tracks/track1.yaml,configs/custom_tracks/track2.yaml,configs/custom_tracks/track3.yaml,configs/custom_tracks/track4.yaml

# Play (keyboard or model)
py scripts/play.py
py scripts/play.py --model models/path/best_model.zip --deterministic

# Validate (headless)
py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 1 --deterministic

# Tests
pytest
```

## Environment Details

### Observation Space
- **Lidar mode:** `Box(0, 1, shape=(num_rays,))` - normalized lidar distances
- **Grid mode:** `Box(0, 255, shape=(36, 36, 1))` - binary occupancy grid

### Action Space
- `Box([-1, 0], [1, 1])` - [steering, throttle]

### Reward Function
| Component | Value | Description |
|-----------|-------|-------------|
| Checkpoint | +1.0 | Per checkpoint passed |
| Progress bonus | +scale * progress | Per step (scale 0.5-0.75) |
| Speed bonus | +0.05 * (speed/max_speed) | Encourages going fast |
| Collision | -20.0 | Wall hit (terminates episode) |

## Configuration

See `configs/README.md` for the full config guide.

## Dependencies

- gymnasium
- pymunk
- pygame-ce
- stable-baselines3
- numpy
- pyyaml
- tensorboard
