# PPO Supersimple Custom Track - 47.31 Reward

**Status:** PROVEN
**Date:** 2026-02-05
**Algorithm:** PPO
**Track:** Supersimple custom track (`custom_tracks/supersimple.yaml`)
**Physics:** Current (`racing_sim/`)

## Performance
| Metric | Value |
|--------|-------|
| Best eval reward | 47.31 at ~890k steps |
| Validation (3 ep, deterministic) | 47.31 |
| Training steps | 1M (4 envs, linear LR schedule) |

## Training
**Config:** `config.yaml`
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 1000000 \
  --config configs/custom_tracks/supersimple.yaml \
  --n-envs 4 --lr-schedule linear --eval-freq 10000 --eval-episodes 3 --no-progress
```

## Usage
```bash
cd racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO Supersimple Custom Track - 47.31 Reward/best_model.zip" \
  --config "../Good Models/PPO Supersimple Custom Track - 47.31 Reward/config.yaml" --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO Supersimple Custom Track - 47.31 Reward/best_model.zip" \
  --config "../Good Models/PPO Supersimple Custom Track - 47.31 Reward/config.yaml" --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint
- `config.yaml` -- Snapshot of the training config

## Notes
- Supersedes "PPO Supersimple Custom Track - 14.00 Reward" (+238% reward).
- 4 parallel envs with cohort spawn and linear LR decay to 10%.
- Model weights are not committed to git.
