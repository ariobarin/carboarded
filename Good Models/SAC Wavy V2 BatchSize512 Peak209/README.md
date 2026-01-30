# SAC Wavy V2 BatchSize512 - Peak 209

**Date:** January 29, 2026
**Algorithm:** SAC
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Results

| Steps | Eval Reward |
|-------|-------------|
| 10k | -16.5 |
| 20k | 8.7 |
| 30k | 145 |
| 40k | 169 |
| 50k | 14.7 (dip) |
| 60k | 159 |
| 70k | 154 |
| 80k | 193 |
| 90k | **209** (best) |
| 100k | 131 (final) |

**Best:** 209 at 90k steps (36% above baseline of 154)
**Stability:** Unstable - large variance (169->14->193->131)

## Configuration

- **Config:** `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Learning Rate:** 0.003
- **Batch Size:** 512 (key change from default 256)
- **Buffer Size:** 200000
- **Gradient Steps:** 4
- **Entropy:** auto
- **Learning Starts:** 0
- **Envs:** 4 (subproc)
- **Total Steps:** 100k

## Command

```bash
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 512 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

## Play

```bash
py scripts/play.py --algo sac \
  --model "Good Models/SAC Wavy V2 BatchSize512 Peak209/best_model.zip" \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --episodes 5 --deterministic
```

## Notes

- Highest SAC peak on Wavy V2 (beats baseline 154 and prior best 159)
- Highly unstable - NOT recommended for production
- batch_size=512 appears to enable higher peaks but introduces variance
- Best model saved at 90k before collapse to 131
