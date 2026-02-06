# PPO Track3 Custom Track - 26.23 Reward

**Status:** PROVEN
**Date:** 2026-02-06
**Algorithm:** PPO
**Track:** Track3 custom track (`custom_tracks/track3.yaml`)
**Observation:** Grid (36x36 binary occupancy, NatureCNN)
**Physics:** Current (`racing_sim/`)

## Performance
| Metric | Value |
|--------|-------|
| Best eval reward | 26.07 at ~210k steps |
| Validation (3 ep, deterministic) | 26.23 |
| Training steps | 300k (1 env, linear LR schedule) |

## Training
**Config:** `config.yaml` (standard: progress=0.5, speed=0.01)
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 300000 \
  --config configs/custom_tracks/track3.yaml \
  --n-envs 1 --lr-schedule linear --l2-reg 0.0001 \
  --eval-freq 10000 --eval-episodes 3 --save-freq 50000 --no-progress --grad-log-freq 10000
```

## Usage
```bash
cd racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO Track3 Custom Track - 26.23 Reward/best_model.zip" \
  --config "../Good Models/PPO Track3 Custom Track - 26.23 Reward/config.yaml" --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO Track3 Custom Track - 26.23 Reward/best_model.zip" \
  --config "../Good Models/PPO Track3 Custom Track - 26.23 Reward/config.yaml" --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint
- `config.yaml` -- Snapshot of the training config

## Notes
- Supersedes "PPO Track3 Custom Track - 23.13 Reward" (+13% reward).
- First grid/CNN model to beat lidar on track3 (26.23 vs 23.13 lidar v2).
- Standard reward config (not reward_exp). Track3 uses standard config unlike other tracks.
- Grid obs with NatureCNN policy, 1 env (4 envs causes PPO collapse on this track).
- Linear LR decay to 10%, L2 regularization (weight_decay=0.0001).
- Training is volatile -- best eval at 210k, performance oscillates significantly.
- Model weights are not committed to git.
