# SAC Wavy V1 - Best (209 at 40k)

**Status:** PROVEN
**Date:** 2026-01-29
**Algorithm:** SAC
**Track:** Wavy V1 (waves=3, waviness=0.06)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 209 at 40k/50k steps |
| Final reward | 191 at 100k steps |
| Stability | Typical SAC dips at 60k/90k, recovers |

## Eval History
| Steps | Reward |
|-------|--------|
| 40k | 209 |
| 50k | 209 |
| 80k | 200 |
| 100k | 191 |

## Training
**Config:** `config.yaml` (legacy snapshot)
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config "../Good Models/SAC Wavy V1 GradSteps4 LR3e-3 AutoEnt/config.yaml" \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

## Usage
```bash
cd racing_sim
py scripts/play.py --algo sac \
  --model "../Good Models/SAC Wavy V1 GradSteps4 LR3e-3 AutoEnt/best_model.zip" \
  --config "../Good Models/SAC Wavy V1 GradSteps4 LR3e-3 AutoEnt/config.yaml" \
  --episodes 5 --deterministic
```

## Files
- `best_model.zip` -- Best eval snapshot (209 reward)
- `sac_final.zip` -- Final snapshot (191 reward)

## Notes
- First SAC model to match PPO performance on Wavy V1 (209 vs PPO's 223)
- gradient_steps=4 is the sweet spot for wavy tracks
- gradient_steps=8 fails on wavy tracks (too aggressive)
- SAC requires --ent-coef auto (fixed values don't learn)
