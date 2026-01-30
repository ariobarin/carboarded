# Fast Iter V3 Complex Wavy V2 Progress0p5 LR3e-3 Ent0p04

Date saved: 2026-01-27
Updated: 2026-01-29

## What this is
PPO model trained on highly wavy oval track (waves=5, waviness=0.08) with progress reward shaping (0.5) and tuned PPO hyperparameters.

**Note:** This model uses progress_reward_scale=0.5. The Progress0p7 version is recommended for faster convergence on this track.

## Model files
- ppo_final.zip
- best_model.zip

## Config
- racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml
  (Use this config - it has the same track settings but better reward shaping)

## Play command
Final model:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Wavy V2 Progress0p5 LR3e-3 Ent0p04/ppo_final.zip" --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --episodes 5 --deterministic
```

Best checkpoint:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Wavy V2 Progress0p5 LR3e-3 Ent0p04/best_model.zip" --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --episodes 5 --deterministic
```

## Training command (original)
```bash
py racing_sim/scripts/train.py --algo ppo --preset fast --total-timesteps 80000 --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.04
```

## Notes
- Track: Very wavy ellipse (waviness=0.08, waves=5)
- Slower convergence than Progress0p7 variant
- Requires highest entropy (0.04) to prevent collapse
- Consider using the Progress0p7 model for better results
