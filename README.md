# Racing RL Project

## Three-Phase Roadmap
**Phase 1 (current):** Build a simple, learnable simulation and prove PPO/SAC can train reliably with LIDAR-style inputs. This phase focuses on fast iteration, clean reward shaping, and stable baselines.

**Phase 2 (planned):** Increase realism with a vision-based policy. Replace raw LIDAR inputs with a processed camera feed (track lines in a bird's-eye view) and train a convolutional model. Phase 2 is treated as the near-final model architecture we'll carry forward.

**Phase 3 (planned):** Transfer to a real race car with camera + IMU. The camera feed will be homography-warped to bird's-eye view, with steering and throttle control mapped to the physical vehicle. Phase 3 focuses on tuning and sim-to-real adaptation.

## Phase 1 Status
- PPO and SAC are training and evaluating in the simulation.
- We now have a more complex �wavy� track for richer decision-making, while keeping the ellipse track for fast sanity checks.

## Quickstart (from repo root)
```bash
# train (PPO / SAC)
py racing_sim/scripts/train.py --algo ppo --preset fast --config racing_sim/configs/fast_iter_v3_complex_long.yaml --total-timesteps 200000
py racing_sim/scripts/train.py --algo sac --preset balanced --config racing_sim/configs/fast_iter_v3_complex_long.yaml --total-timesteps 300000 --ent-coef 0.2

# play a trained model
py racing_sim/scripts/play.py --algo ppo --model models/ppo_fast_YYYYMMDD_HHMMSS/best/best_model.zip --config racing_sim/configs/fast_iter_v3_complex_long.yaml --episodes 3 --deterministic

# evaluate a set of agents and make plots
py racing_sim/analysis/evaluate_agents.py --agents racing_sim/analysis/agents_fast_iter_v3_complex.yaml --config racing_sim/configs/fast_iter_v3_complex_long.yaml --tag phase1_part2
py racing_sim/analysis/make_plots.py --tag phase1_part2
```
Notes:
- Running from repo root writes outputs to `models/` and `logs/` in the root.
- Use `tensorboard --logdir logs` to visualize training curves.

## Track Variants (Phase 1)
The simulator supports optional �waviness� to make the ellipse more challenging. Add these fields under `track` in any config:
- `waviness`: float (0.0 keeps a perfect ellipse)
- `waves`: integer (number of bumps around the track)
- `wave_phase`: phase offset in radians

Example config: `racing_sim/configs/fast_iter_v3_complex.yaml`

## Sensors (LIDAR)
Current configs use **9 rays** and **400px max distance** for better lookahead. Older checkpoints trained with 5 rays require:
`racing_sim/configs/legacy_fast_iter_v3_complex_long_5ray.yaml`

## Versioning Across Phases
We will keep phase-specific configs and models so we can always re-run:
- Phase 1: LIDAR + simple physics (current)
- Phase 2: Vision (CNN) with processed track lines
- Phase 3: Real-world deployment with camera + IMU

Use clear naming for configs and models (e.g., `phase1_*`, `phase2_*`, `phase3_*`) so experiments remain reproducible.

