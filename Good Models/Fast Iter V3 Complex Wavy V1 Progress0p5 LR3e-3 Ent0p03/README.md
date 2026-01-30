# Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03

Date saved: 2026-01-27
Verified: 2026-01-29

## What this is
PPO model trained on moderately wavy oval track (waves=3, waviness=0.06) with progress reward shaping and tuned PPO hyperparameters.

## Model files
- ppo_final.zip
- best_model.zip

## Config
- racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml

## Verified Results
- **216 at 30k** (first >=200)
- **223 at 80k** (stable)
- Full 2000-step episodes maintained

## Play command
Final model:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03/ppo_final.zip" --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml --episodes 5 --deterministic
```

Best checkpoint:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03/best_model.zip" --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml --episodes 5 --deterministic
```

## Training command
```bash
py racing_sim/scripts/train.py --algo ppo --preset fast --total-timesteps 80000 --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.03
```

## Notes
- Converges to >=200 mean reward by ~30k steps on wavy track
- Stable full-length episodes through 80k
- Track: Wavy ellipse (waviness=0.06, waves=3)
- Requires slightly higher entropy (0.03) than simple track
