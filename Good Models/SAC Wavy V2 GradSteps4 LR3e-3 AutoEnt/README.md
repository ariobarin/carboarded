# SAC Wavy V2 (gradient_steps=4, LR 3e-3, Auto Entropy)

Date saved: 2026-01-29

## What this is
SAC model trained on highly wavy track (waves=5, waviness=0.08) with conservative gradient steps.

**This is the first SAC model to achieve >100 reward on Wavy V2!**

## Model files
- best_model.zip (best eval snapshot - 154 reward)
- sac_final.zip (final snapshot - 154 reward)

## Config
- racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml

## Verified Results
- **154 at 100k** (vs PPO's 225)
- 148 at 60k
- Typical SAC instability (dips at 70k-90k) but recovers
- Previous SAC best on Wavy V2: only 40.8 (this is 4x better!)

## Play command
Best checkpoint:
```bash
py racing_sim/scripts/play.py --algo sac \
  --model "Good Models/SAC Wavy V2 GradSteps4 LR3e-3 AutoEnt/best_model.zip" \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --episodes 5 --deterministic
```

## Training command
```bash
py racing_sim/scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

## Key Insight
gradient_steps=4 is critical for hard tracks!
- gradient_steps=8 completely fails on Wavy V2 (max 40.8)
- gradient_steps=4 achieves 154 (4x improvement!)
- Still below PPO (225) - may need longer training or gradient_steps=2
