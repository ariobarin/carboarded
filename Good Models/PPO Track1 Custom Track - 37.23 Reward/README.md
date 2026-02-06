# PPO Track1 Custom Track - 37.23 Reward

**Status:** PROVEN
**Date:** 2026-02-06
**Algorithm:** PPO
**Track:** Track1 custom track (`custom_tracks/track1_reward_exp.yaml`)
**Observation:** Grid (36x36 binary occupancy, NatureCNN)
**Physics:** Current (`racing_sim/`)

## Performance
| Metric | Value |
|--------|-------|
| Best eval reward | 37.23 at ~470k steps |
| Validation (3 ep, deterministic) | 37.23 |
| Training steps | 500k (1 env, linear LR schedule) |

## Training
**Config:** `config.yaml` (reward-shaped: progress=0.65, speed=0.03)
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --config configs/custom_tracks/track1_reward_exp.yaml \
  --n-envs 1 --lr-schedule linear --l2-reg 0.0001 \
  --eval-freq 10000 --eval-episodes 3 --save-freq 50000 --no-progress --grad-log-freq 10000
```

## Usage
```bash
cd racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO Track1 Custom Track - 37.23 Reward/best_model.zip" \
  --config "../Good Models/PPO Track1 Custom Track - 37.23 Reward/config.yaml" --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO Track1 Custom Track - 37.23 Reward/best_model.zip" \
  --config "../Good Models/PPO Track1 Custom Track - 37.23 Reward/config.yaml" --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint
- `config.yaml` -- Snapshot of the training config (reward-shaped variant)

## Notes
- Supersedes "PPO Track1 Custom Track - 32.80 Reward" (+14% reward).
- Grid/CNN beats lidar by +48% on this track (37.23 vs 25.14 lidar v2).
- Reward shaping: progress_reward_scale=0.65, speed_bonus_scale=0.03.
- Grid obs with NatureCNN policy, 1 env (4 envs causes PPO collapse on this track).
- Linear LR decay to 10%, L2 regularization (weight_decay=0.0001).
- Model weights are not committed to git.
