# PPO Track2 Custom Track - 37.00 Reward

**Status:** PROVEN
**Date:** 2026-02-05
**Algorithm:** PPO
**Track:** Track2 custom track (`custom_tracks/track2.yaml`)
**Physics:** Current (`racing_sim/`)

## Performance
| Metric | Value |
|--------|-------|
| Best eval reward | 36.83 at 480k steps |
| Validation (3 ep, deterministic) | 37.00 |
| Training steps | 1M (1 env, ent_coef=0.03) |

## Training
**Config:** `config.yaml`
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 1000000 \
  --config configs/custom_tracks/track2.yaml \
  --n-envs 1 --ent-coef 0.03 --eval-freq 10000 --eval-episodes 3 --no-progress
```

## Usage
```bash
cd racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO Track2 Custom Track - 37.00 Reward/best_model.zip" \
  --config "../Good Models/PPO Track2 Custom Track - 37.00 Reward/config.yaml" --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO Track2 Custom Track - 37.00 Reward/best_model.zip" \
  --config "../Good Models/PPO Track2 Custom Track - 37.00 Reward/config.yaml" --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint
- `config.yaml` -- Snapshot of the training config

## Notes
- Supersedes "PPO Track2 Custom Track - 15.47 Reward" (+139% reward).
- Higher entropy (0.03 vs 0.02 default) was key; linear LR schedule caused collapse on this track.
- 1 env with cohort spawn required (4 envs collapsed).
- Classic PPO collapse after peak; best checkpoint saved at 480k.
- Model weights are not committed to git.
