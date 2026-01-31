# Racing RL Project

## Three-Phase Roadmap
**Phase 1 (COMPLETE):** Build a simple, learnable simulation and prove PPO/SAC can train reliably with LIDAR-style inputs. PPO achieved 226.49 reward on the hardest track (Wavy V2). SAC achieved 183.7 with random starts.

**Phase 2 (planned):** Increase realism with a vision-based policy. Replace raw LIDAR inputs with a processed camera feed (track lines in a bird's-eye view) and train a convolutional model.

**Phase 3 (planned):** Transfer to a real race car with camera + IMU. The camera feed will be homography-warped to bird's-eye view, with steering and throttle control mapped to the physical vehicle.

## Phase 1 Results
- **Best PPO:** 226.49 on Wavy V2 (100% validation, 0.00 std dev across 100 episodes)
- **Best SAC:** 183.7 on Wavy V2 with random starts
- See `PHASE_1_COMPLETE.md` for details and `Learnings/Phase One Summary.md` for experiment history

## Quickstart (from racing_sim/ directory)
```bash
# Train PPO on best config
py scripts/train.py --algo ppo --preset fast --total-timesteps 50000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5

# Train SAC on Wavy V2
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc

# Play the best model
py scripts/play.py --algo ppo \
  --model "../Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/best_model.zip" \
  --config configs/wavy_v2_progress_0p75.yaml --episodes 5 --deterministic

# Validate (headless, 100 episodes)
py scripts/validate.py \
  --model "../Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/best_model.zip" \
  --config configs/wavy_v2_progress_0p75.yaml --episodes 100 --deterministic

# TensorBoard
tensorboard --logdir logs
```

## Track Variants (Phase 1)
The simulator supports optional waviness to make the ellipse more challenging:
- **Simple ellipse:** `configs/fast_iter_v3_complex_progress_0p5.yaml`
- **Wavy V1** (waves=3, waviness=0.06): `configs/fast_iter_v3_complex_wavy_v1.yaml`
- **Wavy V2** (waves=5, waviness=0.08): `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Wavy V2 optimized** (progress 0.75): `configs/wavy_v2_progress_0p75.yaml`

## Sensors (LIDAR)
Current configs use **9 rays** and **400px max distance** for better lookahead.

## Project Organization
- `racing_sim/` -- Main Python package and training scripts
- `Good Models/` -- Proven trained models with READMEs (6 models)
- `Good Models/_archived/` -- Superseded models kept for reference
- `Learnings/` -- Experiment summaries and anti-pattern guide
- `archive/` -- Deprecated root-level documents
- `CLAUDE.md` -- Agent quick reference
- `STANDARDS.md` -- Conventions for future work
- `RESEARCH_ROADMAP.md` -- Multi-phase research plan
