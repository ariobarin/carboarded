# SAC Wavy V1 (gradient_steps=4, LR 3e-3, Auto Entropy)

Date saved: 2026-01-29

## What this is
SAC model trained on moderately wavy track (waves=3, waviness=0.06) with conservative gradient steps.

**This is the first SAC model to MATCH PPO performance on Wavy V1!**

## Model files
- best_model.zip (best eval snapshot - 209 reward)
- sac_final.zip (final snapshot - 191 reward)

## Config
- racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml

## Verified Results
- **209 at 40k/50k** (matches PPO's 220!)
- 200 at 80k
- 191 at 100k
- Typical SAC instability (dips at 60k/90k) but recovers

## Play command
Best checkpoint:
```bash
py racing_sim/scripts/play.py --algo sac \
  --model "Good Models/SAC Wavy V1 GradSteps4 LR3e-3 AutoEnt/best_model.zip" \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml \
  --episodes 5 --deterministic
```

## Training command
```bash
py racing_sim/scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

## Key Insight
gradient_steps=4 is the sweet spot for wavy tracks! 
- gradient_steps=8 fails on wavy tracks (too aggressive)
- gradient_steps=4 achieves PPO-level performance
