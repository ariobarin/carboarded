# Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02

Date saved: 2026-01-27
Verified: 2026-01-29

## What this is
PPO model trained on simple ellipse track with progress reward shaping (0.5) and tuned PPO hyperparameters for fast, stable convergence.

## Model files
- ppo_final.zip
- best_model.zip

## Config
- racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml

## Verified Results
- **217 at 30k** (first >=200)
- **246 at 80k** (stable)
- Full 2000-step episodes maintained

## Play command
Final model:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02/ppo_final.zip" --config racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml --episodes 5 --deterministic
```

Best checkpoint:
```bash
py racing_sim/scripts/play.py --algo ppo --model "Good Models/Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02/best_model.zip" --config racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml --episodes 5 --deterministic
```

## Training command
```bash
py racing_sim/scripts/train.py --algo ppo --preset fast --total-timesteps 80000 --config racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.02
```

## Notes
- Converges to >=200 mean reward by ~30k steps
- Stable full-length episodes through 80k
- Track: Simple ellipse (waviness=0, waves=0)
