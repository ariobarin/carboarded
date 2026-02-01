# Racing RL Project

2D autonomous racing simulation using Gymnasium, Pymunk physics, and Stable-Baselines3 for training PPO/SAC agents.

## Results

| Phase | Algorithm | Track | Obs Type | Best Reward | Steps |
|-------|-----------|-------|----------|-------------|-------|
| 2 | PPO+CNN | Wavy V2 | Grid (36x36) | **249.43** | 220k |
| 1 | PPO | Wavy V2 | Lidar (9-ray) | **247.26** | 220k |
| 1 | PPO | Wavy V1 | Lidar | **237.57** | 80k |
| 1 | PPO | Simple | Lidar | **252.49** | 80k |
| 1 | SAC | Wavy V1 | Lidar | 209.09 | 40k |
| 1 | SAC | Wavy V2 | Lidar | 183.70 | 90k |

All models validated with 100 episodes in deterministic mode. See `CLAUDE.md` for full details.

## Quickstart

All commands run from the `racing_sim/` directory.

```bash
# Install
uv venv && uv pip install -e .

# Train PPO (lidar, best config)
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --learning-rate 0.001 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5

# Train PPO (CNN, all-time best)
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42

# Play with keyboard (arrows/WASD, ESC to quit)
py scripts/play.py

# Play a trained model
py scripts/play.py --model "../Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/best_model.zip" \
  --config configs/wavy_v2_cnn.yaml --deterministic --episodes 5

# Validate (headless, 100 episodes)
py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 100 --deterministic

# TensorBoard
tensorboard --logdir logs
```

## Track Variants

- **Simple ellipse:** `configs/fast_iter_v3_complex_progress_0p5.yaml`
- **Wavy V1** (waves=3, waviness=0.06): `configs/fast_iter_v3_complex_wavy_v1.yaml`
- **Wavy V2** (waves=5, waviness=0.08): `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Wavy V2 optimized** (progress 0.75): `configs/wavy_v2_progress_0p75.yaml`
- **Wavy V2 CNN** (36x36 grid): `configs/wavy_v2_cnn.yaml`

## Project Organization

- `racing_sim/` -- Main Python package, training scripts, and configs
- `Good Models/` -- Proven trained models with READMEs
- `Learnings/` -- Experiment summaries and anti-pattern guide (see `Learnings/README.md` for index)
- `CLAUDE.md` -- Agent quick reference (commands, architecture, hyperparameters)
- `STANDARDS.md` -- Experiment protocols and conventions

## Future Work

- **Robustness:** Multi-seed validation, sensitivity analysis, failure mode documentation
- **Harder tracks:** waviness > 0.08, more waves, narrower width
- **Domain randomization:** Lidar noise, friction variation, action delay for sim-to-real transfer
- **Deployment:** ONNX export, model quantization for edge inference
- **Sim-to-real bridge:** Homography-warped camera feed on physical car with steering/throttle control
