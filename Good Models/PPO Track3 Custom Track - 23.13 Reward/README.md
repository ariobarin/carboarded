# PPO Track3 Custom Track - 23.13 Reward

**Status:** PROVEN
**Date:** 2026-02-05
**Algorithm:** PPO
**Track:** Track3 custom track (`custom_tracks/track3.yaml`)
**Physics:** Current (`racing_sim/`)

## Performance
| Metric | Value |
|--------|-------|
| Best eval reward | 23.13 at ~440k steps |
| Validation (3 ep, deterministic) | 23.13 |
| Training steps | 1M (1 env, linear LR schedule) |

## Training
**Config:** `config.yaml`
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 1000000 \
  --config configs/custom_tracks/track3.yaml \
  --n-envs 1 --lr-schedule linear --eval-freq 10000 --eval-episodes 3 --no-progress
```

## Usage
```bash
cd racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO Track3 Custom Track - 23.13 Reward/best_model.zip" \
  --config "../Good Models/PPO Track3 Custom Track - 23.13 Reward/config.yaml" --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO Track3 Custom Track - 23.13 Reward/best_model.zip" \
  --config "../Good Models/PPO Track3 Custom Track - 23.13 Reward/config.yaml" --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint
- `config.yaml` -- Snapshot of the training config

## Notes
- Supersedes "PPO Track3 Custom Track - 21.03 Reward" (+10% reward).
- 1 env with cohort spawn required for this track (4 envs caused PPO collapse).
- Linear LR decay to 10% of initial.
- Model weights are not committed to git.
