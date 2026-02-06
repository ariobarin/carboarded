# Custom Track Baselines (2026-02-05)

Baseline PPO runs on all custom track configs using the **simple.yaml protocol** (updated across the custom track configs and default.yaml).

## Protocol
- Algorithm: PPO (defaults from `training_presets.yaml`)
- Obs type: Lidar (9-ray)
- Reward config: `simple.yaml` protocol (progress_reward_scale=0.5, speed_bonus_scale=0.01, collision_penalty=-2.0, no terminate_on_collision)
- Total timesteps budget: 500k per run, with early stopping when eval rewards plateaued at low values
- Eval: `--eval-freq 20000`, `--eval-episodes 1` (deterministic), and final validation with `validate.py --episodes 1 --deterministic`
- Command template:
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --config "../Good Models/<model>/config.yaml" --no-progress
```

## Initial Baselines (v1)
| Track | Run dir | Best eval (step) | Validation (1 ep) | Steps run | Notes |
|------|---------|------------------|-------------------|-----------|-------|
| simple | `models/ppo_20260205_012918` | 24.92 @ 180k | 24.92 | 220k | Stopped early: plateaued eval (<25) |
| supersimple | `models/ppo_20260205_013425` | 14.00 @ 140k | 14.00 | 331k | Stopped early: eval mostly 0-14 |
| square_test | `models/ppo_20260205_013927` | 48.85 @ 280k | 48.85 | 319k | Best performer in this batch |
| track1 | `models/ppo_20260205_014508` | 24.07 @ 200k | 24.07 | 200k | Stopped early: unstable eval |
| track2 | `models/ppo_20260205_014834` | 15.47 @ 140k | 15.47 | 155k | Stopped early: non-learning behavior |
| track3 | `models/ppo_20260205_015131` | 21.03 @ 220k | 21.03 | 227k | Stopped early: flat eval |
| track4 | `models/ppo_20260205_015532` | 15.38 @ 100k | 15.38 | 160k | Stopped early: low eval |

## Improved Baselines (v2)

Longer training with tuned hyperparameters. Protocol changes:
- 1M timesteps (no early stopping)
- `--eval-freq 10000`, `--eval-episodes 3`
- Validated with `validate.py --episodes 3 --deterministic`

Two configurations found to matter:
- **Easy tracks (simple, supersimple, square_test, track4):** 4 envs + linear LR schedule worked well
- **Hard tracks (track1, track2, track3):** 1 env required (4 envs killed PPO diversity via cohort spawn). Track2 also needed higher entropy (0.03 vs 0.02 default).

| Track | Run dir | Best eval (step) | Validation (3 ep) | Config notes | vs v1 |
|------|---------|------------------|-------------------|--------------|-------|
| simple | `models/ppo_20260205_100832` | 30.59 @ 440k | 30.59 | 4 envs, linear LR | +23% |
| supersimple | `models/ppo_20260205_102018` | 47.31 @ ~890k | 47.31 | 4 envs, linear LR | +238% |
| square_test | `models/ppo_20260205_102939` | 63.92 @ ~790k | 63.92 | 4 envs, linear LR | +31% |
| track1 | `models/ppo_20260205_112505` | 25.14 @ ~630k | 25.14 | 1 env, linear LR | +4% |
| track2 | `models/ppo_20260205_121856` | 36.83 @ 480k | 37.00 | 1 env, ent_coef=0.03 | +139% |
| track3 | `models/ppo_20260205_115716` | 23.13 @ ~440k | 23.13 | 1 env, linear LR | +10% |
| track4 | `models/ppo_20260205_111149` | 30.59 @ 640k | 30.59 | 4 envs, linear LR | +99% |

## Key Findings
- **4 envs + cohort spawn collapses on harder tracks.** With 4 envs, all start from the same checkpoint per rollout, reducing within-batch diversity. Simpler tracks are robust to this, harder tracks collapse to zero.
- **Linear LR schedule helps most tracks** by preventing late-stage oscillation, but can hurt track2 where the policy needs sustained exploration.
- **Higher entropy (0.03) rescued track2** from persistent collapse.
- **1M steps was necessary** for most tracks; v1 baselines stopped too early.
- Track2 exhibits classic PPO collapse pattern: peaks then drops sharply. Always save best checkpoint.

## Notes
- All config snapshots and READMEs are in `Good Models/` (current physics).
- Model weights (`best_model.zip`) are present locally but excluded from git by `.gitignore`.
- Legacy results and configs live in `legacy/Good Models/`.
