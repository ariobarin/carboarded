# SAC Wavy V2 GradSteps4 LR3e-3 AutoEnt

**Status:** ARCHIVED
**Reason:** Superseded by SAC Wavy V2 Random Start (183.7 at 90k vs this model's 154 at 100k)
**Use instead:** `Good Models/SAC Wavy V2 Random Start 183.7 at 90k/`

**Date:** 2026-01-29
**Algorithm:** SAC
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 154 at 100k steps |
| Eval at 60k | 148 |
| Stability | Typical SAC instability (dips at 70k-90k) |

## Training
**Config:** `../Good Models/_archived/SAC Wavy V2 GradSteps4 LR3e-3 AutoEnt/config.yaml`
**Command:**
```bash
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config ../Good Models/_archived/SAC Wavy V2 GradSteps4 LR3e-3 AutoEnt/config.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

## Files
- `best_model.zip` -- Best eval snapshot (154 reward)
- `sac_final.zip` -- Final snapshot (154 reward)

## Notes
- First SAC model to achieve >100 reward on Wavy V2 (historical milestone)
- gradient_steps=4 confirmed as sweet spot for hard tracks
- Superseded by random start variant which reached 183.7
