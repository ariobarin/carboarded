# PPO Wavy V1 - Best (223 at 80k)

**Status:** PROVEN
**Date:** 2026-01-27
**Algorithm:** PPO
**Track:** Wavy V1 (waves=3, waviness=0.06)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 223 at 80k steps |
| First >200 | 216 at 30k steps |
| Episode length | 2000 (full laps) |

## Training
**Config:** `config.yaml` (legacy snapshot)
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --preset fast --total-timesteps 80000 \
  --config "../Good Models/Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03/config.yaml" \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.03
```

## Usage
```bash
cd racing_sim
py scripts/play.py --algo ppo \
  --model "../Good Models/Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03/best_model.zip" \
  --config "../Good Models/Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03/config.yaml" \
  --episodes 5 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint (223 reward)
- `ppo_final.zip` -- Final model at 80k steps

## Notes
- Stable full-length episodes through 80k steps
- Wavy V1 requires slightly higher entropy (0.03) than simple track (0.02)
- Converges to >200 by ~30k steps
