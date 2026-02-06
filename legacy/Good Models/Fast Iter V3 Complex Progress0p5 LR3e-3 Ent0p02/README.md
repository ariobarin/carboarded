# PPO Simple Track - Best (246 at 80k)

**Physics:** Legacy (`legacy/racing_sim`).

**Status:** PROVEN
**Date:** 2026-01-27
**Algorithm:** PPO
**Track:** Simple ellipse (no waviness)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 246 at 80k steps |
| First >200 | 217 at 30k steps |
| Episode length | 2000 (full laps) |

## Training
**Config:** `config.yaml` (legacy snapshot)
**Command:**
```bash
cd legacy/racing_sim
py scripts/train.py --algo ppo --total-timesteps 80000 \
  --config "../Good Models/Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02/config.yaml" \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.02
```

## Usage
```bash
cd legacy/racing_sim
py scripts/play.py --algo ppo \
  --model "../Good Models/Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02/best_model.zip" \
  --config "../Good Models/Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02/config.yaml" \
  --episodes 5 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint (246 reward)
- `ppo_final.zip` -- Final model at 80k steps

## Notes
- Stable full-length episodes through 80k steps
- Simple track serves as fast sanity check for PPO hyperparameters
- LR=0.003 and ent_coef=0.02 are the proven defaults for simple tracks

