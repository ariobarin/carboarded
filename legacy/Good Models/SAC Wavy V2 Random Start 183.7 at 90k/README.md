# SAC Wavy V2 Random Start - Best SAC Overall (183.7 at 90k)

**Physics:** Legacy (`legacy/racing_sim`).

**Status:** PROVEN
**Date:** 2026-01-30
**Algorithm:** SAC
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 183.7 at 90k steps |
| Stability | Very unstable (typical SAC dip-and-recover) |
| Episode length | 2000 at high-reward checkpoints |

## Eval History
| Steps | Reward | Episode Length |
|-------|--------|----------------|
| 10k | 5.34 | 656 |
| 20k | 24.08 | 2000 |
| 30k | 9.72 | 1426 |
| 40k | 120.39 | 2000 |
| 50k | 148.00 | 2000 |
| 60k | 7.62 | 307 (crash) |
| 70k | 176.66 | 2000 |
| 80k | 92.85 | 2000 |
| 90k | 183.70 | 2000 (peak) |

## Training
**Config:** `config.yaml` (legacy snapshot)
**Command:**
```bash
cd legacy/racing_sim
py scripts/train.py --algo sac --total-timesteps 100000 \
  --config "../Good Models/SAC Wavy V2 Random Start 183.7 at 90k/config.yaml" \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --random-start --n-envs 4 --vec-env subproc
```

## Usage
```bash
cd legacy/racing_sim
py scripts/play.py --algo sac \
  --model "../Good Models/SAC Wavy V2 Random Start 183.7 at 90k/best_model.zip" \
  --config "../Good Models/SAC Wavy V2 Random Start 183.7 at 90k/config.yaml" \
  --episodes 5 --deterministic
```

## Files
- `best_model.zip` -- Best policy (183.7 reward)
- `vecnormalize.pkl` -- Normalization stats

## Notes
- Best SAC result on any hard track
- Random starts work for SAC but destroy PPO (PPO got 17.7 with random starts)
- SAC's off-policy learning handles varied starting conditions better than PPO
- Still 42 points below PPO's best (226.49), confirming PPO is stronger on Wavy V2
- Longer training (150k+) may push past 200

