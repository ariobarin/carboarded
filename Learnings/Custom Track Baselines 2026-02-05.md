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

## PPO-Adjacent Experiments (v3)

GRPO-inspired experiments on the two worst-performing tracks (track3=23.13, track1=25.14).
All runs: 1M steps, 1 env, linear LR, eval-freq 10000, eval-episodes 3, validated with 3 deterministic episodes.

### Experiment 1: Advantage Normalization Off
Disabled SB3's per-minibatch advantage normalization (`normalize_advantage=False`) to preserve absolute advantage magnitude.

| Track | Run dir | Validation (3 ep) | Baseline | Delta |
|-------|---------|-------------------|----------|-------|
| track3 | `models/ppo_20260205_191005` | 1.15 | 23.13 | -95% |
| track1 | `models/ppo_20260205_191009` | 25.09 | 25.14 | ~0% |

Result: Catastrophic on track3, neutral on track1. Advantage normalization is clearly necessary for hard tracks -- raw advantages have high variance that destabilizes learning.

### Experiment 2: More Epochs + Tighter Clipping
`n_epochs=10` (vs 5), `clip_range=0.1` (vs 0.2), `target_kl=0.02` (vs 0.05). More passes through rollout data with tighter policy constraints.

| Track | Run dir | Validation (3 ep) | Baseline | Delta |
|-------|---------|-------------------|----------|-------|
| track3 | `models/ppo_20260205_191013` | 18.90 | 23.13 | -18% |
| track1 | `models/ppo_20260205_191018` | 21.87 | 25.14 | -13% |

Result: Worse on both tracks. Tighter clipping likely prevents the policy from making large enough updates to escape local optima on hard tracks.

### Experiment 3: Larger Rollouts
`n_steps=2048` (vs 1024). Longer rollouts for more diverse experience per update.

| Track | Run dir | Validation (3 ep) | Baseline | Delta |
|-------|---------|-------------------|----------|-------|
| track3 | `models/ppo_20260205_191023` | 20.94 | 23.13 | -9% |
| track1 | `models/ppo_20260205_191028` | **28.41** | 25.14 | **+13%** |

Result: Mixed. Hurt track3 slightly but improved track1 meaningfully. Larger rollouts may help tracks where the car needs to see more of the circuit per update to learn coherent strategies.

### Experiment 4: Reward Shaping
Higher reward scales: `progress_reward_scale=0.65` (vs 0.5), `speed_bonus_scale=0.03` (vs 0.01). Trained on modified configs (`track{1,3}_reward_exp.yaml`).

| Track | Run dir | Validation (3 ep, reward config) | Validation (3 ep, standard config) | Baseline | Delta (std) |
|-------|---------|----------------------------------|-------------------------------------|----------|-------------|
| track3 | `models/ppo_20260205_191033` | 18.57 | -- | 23.13 | -20% |
| track1 | `models/ppo_20260205_191037` | 32.80 | **32.80** | 25.14 | **+30%** |

Result: Track1 reward-shaped model achieves 32.80 even when evaluated against the standard reward config -- a genuine +30% improvement in driving quality. Track3 still resists improvement.

### Summary

| Experiment | track3 | track1 |
|------------|--------|--------|
| v2 baseline | 23.13 | 25.14 |
| No adv norm | 1.15 (-95%) | 25.09 (~0%) |
| Epochs+clip | 18.90 (-18%) | 21.87 (-13%) |
| n_steps=2048 | 20.94 (-9%) | **28.41 (+13%)** |
| Reward shaping | 18.57 (-20%) | **32.80 (+30%)** |

**New track1 best: 32.80** (reward shaping, `models/ppo_20260205_191037`). Runner-up: 28.41 (larger rollouts).

Track3 remains the hardest track. None of the GRPO-inspired modifications improved it -- all four experiments scored worse than the v2 baseline. Track3 likely needs architectural changes (CNN obs, curriculum, or track geometry edits) rather than PPO hyperparameter tuning.

### Experiment Findings
- Advantage normalization is essential; disabling it causes collapse on hard tracks.
- Tighter clipping + more epochs hurts hard tracks (prevents escaping local optima).
- Larger rollouts (2048 vs 1024) help track1 but not track3.
- Richer reward signal (higher progress + speed bonus) produced the biggest gain on track1 (+30%) and the improvement is real (validated on standard config).
- Track3 is resistant to all PPO hyperparameter changes tested.

## Notes
- All config snapshots and READMEs are in `Good Models/` (current physics).
- Model weights (`best_model.zip`) are present locally but excluded from git by `.gitignore`.
- Legacy results and configs live in `legacy/Good Models/`.
