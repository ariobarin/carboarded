# SAC Wavy V2 BatchSize512 Peak209

**Status:** ARCHIVED
**Reason:** Highly unstable (169->14->193->131). Not recommended for any use.
**Use instead:** `Good Models/SAC Wavy V2 Random Start 183.7 at 90k/`

**Date:** 2026-01-29
**Algorithm:** SAC
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 209 at 90k steps |
| Final reward | 131 at 100k steps |
| Stability | UNSTABLE -- large variance (169->14->193->131) |

## Eval History
| Steps | Reward |
|-------|--------|
| 10k | -16.5 |
| 20k | 8.7 |
| 30k | 145 |
| 40k | 169 |
| 50k | 14.7 (crash) |
| 60k | 159 |
| 70k | 154 |
| 80k | 193 |
| 90k | 209 (peak) |
| 100k | 131 (collapse) |

## Training
**Config:** `../Good Models/_archived/SAC Wavy V2 BatchSize512 Peak209/config.yaml`
**Command:**
```bash
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config ../Good Models/_archived/SAC Wavy V2 BatchSize512 Peak209/config.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 512 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

## Files
- `best_model.zip` -- Best eval snapshot (209 reward at 90k)

## Notes
- batch_size=512 enables higher peaks but introduces severe variance
- Collapsed from 209 to 131 in final 10k steps
- Demonstrates that batch_size alone cannot fix SAC instability on hard tracks
