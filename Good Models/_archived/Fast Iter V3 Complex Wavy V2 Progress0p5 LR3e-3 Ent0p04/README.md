# Fast Iter V3 Complex Wavy V2 Progress0p5 LR3e-3 Ent0p04

**Status:** ARCHIVED
**Reason:** Superseded by Progress0p7 variant (225 at 80k) and Progress0p75 variant (226.49 at 30k)
**Use instead:** `Good Models/Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04/`

**Date:** 2026-01-27
**Algorithm:** PPO
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | Unknown (not recorded) |
| Config progress scale | 0.5 (suboptimal for this track) |

## Training
**Config:** `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 80000 \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.04
```

## Files
- `ppo_final.zip` -- Final model checkpoint
- `best_model.zip` -- Best eval checkpoint

## Notes
- Slower convergence than Progress0p7 variant due to lower progress reward scale
- Track requires progress_reward_scale >= 0.7 for good results
