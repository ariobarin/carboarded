# PPO Wavy V2 - From Scratch (225 at 80k)

**Status:** PROVEN
**Date:** 2026-01-27
**Algorithm:** PPO
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 225 at 80k steps |
| First >200 | 216 at 50k steps |
| Episode length | 2000 (full laps) |

## Training
**Config:** `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --preset fast --total-timesteps 80000 \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.04
```

## Usage
```bash
cd racing_sim
py scripts/play.py --algo ppo \
  --model "../Good Models/Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04/best_model.zip" \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --episodes 5 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint (225 reward)
- `ppo_final.zip` -- Final model at 80k steps

## Notes
- Recommended from-scratch training config for Wavy V2
- Wavy V2 requires highest entropy (0.04) to prevent late-stage collapse
- progress_reward_scale=0.7 is the minimum for good Wavy V2 convergence
- Superseded in peak reward by the fine-tuned 0.75 variant (226.49)
