# Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04

Date saved: 2026-01-27
Verified: 2026-01-29

## What this is
PPO model trained on highly wavy oval track (waves=5, waviness=0.08) with stronger progress shaping (0.7) for faster convergence.

**This is the recommended model for the Wavy V2 track.**

## Model files
- ppo_final.zip
- best_model.zip

## Config
- racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml

## Verified Results
- **216 at 50k** (first >=200)
- **225 at 80k** (stable)
- Full 2000-step episodes maintained

## Play command
Final model:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04/ppo_final.zip" --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --episodes 5 --deterministic
```

Best checkpoint:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04/best_model.zip" --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --episodes 5 --deterministic
```

## Training command
```bash
py racing_sim/scripts/train.py --algo ppo --preset fast --total-timesteps 80000 --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.04
```

## Notes
- Faster convergence than progress_reward_scale=0.5 on Wavy V2
- Reaches >=200 mean reward by ~50k (vs ~70k with 0.5)
- Stable through 80k
- Track: Very wavy ellipse (waviness=0.08, waves=5)
- Requires highest entropy (0.04) to prevent late-stage collapse
