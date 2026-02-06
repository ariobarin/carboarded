# PPO Track1 Custom Track - 32.80 Reward

**Status:** PROVEN
**Date:** 2026-02-05
**Algorithm:** PPO
**Track:** Track1 custom track (`custom_tracks/track1.yaml`)
**Physics:** Current (`racing_sim/`)

## Performance
| Metric | Value |
|--------|-------|
| Best eval reward | 32.80 |
| Validation (3 ep, deterministic) | 32.80 |
| Training steps | 1M (1 env, linear LR schedule) |

## Training
**Config:** `config.yaml` (reward-shaped: progress=0.65, speed=0.03)
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 1000000 \
  --config configs/custom_tracks/track1_reward_exp.yaml \
  --n-envs 1 --lr-schedule linear --eval-freq 10000 --eval-episodes 3 --no-progress
```

## Usage
```bash
cd racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO Track1 Custom Track - 32.80 Reward/best_model.zip" \
  --config "../Good Models/PPO Track1 Custom Track - 32.80 Reward/config.yaml" --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO Track1 Custom Track - 32.80 Reward/best_model.zip" \
  --config "../Good Models/PPO Track1 Custom Track - 32.80 Reward/config.yaml" --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint
- `config.yaml` -- Snapshot of the training config (reward-shaped variant)

## Notes
- Supersedes "PPO Track1 Custom Track - 25.14 Reward" (+30% reward via reward shaping).
- Reward shaping: progress_reward_scale=0.65 (vs 0.5), speed_bonus_scale=0.03 (vs 0.01).
- Validated on standard config (track1.yaml) -- 32.80 is genuine improvement, not inflated by reward scaling.
- 1 env with cohort spawn required for this track (4 envs caused PPO collapse).
- Linear LR decay to 10% of initial.
- Model weights are not committed to git.
