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

## Reward Shaping Sweep + Track3 Experiments (v4)

Reward shaping (progress=0.65, speed=0.03) applied to all remaining tracks. Also tested combined modifications and higher entropy on track3.
All runs: 1M steps, linear LR, eval-freq 10000, eval-episodes 3. Easy tracks used 4 envs; hard tracks used 1 env.
**Note:** v4 runs were executed in parallel (CPU overloaded). Lidar runs completed; grid obs runs (B1/B2/D3) did NOT complete and need to be rerun.

### Batch A: Combine Winners (reward shaping + n_steps=2048)

| Track | Run dir | Validation (3 ep, std config) | Baseline | Delta |
|-------|---------|-------------------------------|----------|-------|
| track3 | `models/ppo_20260205_205145` | 0.15 | 23.13 | -99% (collapsed) |
| track1 | `models/ppo_20260205_205149` | 32.75 | 32.80 | ~0% |

Result: Combining reward shaping + larger rollouts provides no benefit over reward shaping alone on track1, and catastrophically collapses on track3.

### Batch C: Reward Shaping Sweep (remaining tracks)

All validated against **standard** configs (not reward_exp) for fair comparison with v2 baselines.

| Track | Run dir | Envs | Validation (3 ep, std config) | Baseline | Delta |
|-------|---------|------|-------------------------------|----------|-------|
| simple | `models/ppo_20260205_205202` | 4 | **32.68** | 30.59 | **+7%** |
| supersimple | `models/ppo_20260205_205214` | 4 | 47.26 | 47.31 | ~0% |
| square_test | `models/ppo_20260205_205218` | 4 | 64.02 | 63.92 | ~0% |
| track2 | `models/ppo_20260205_205236` | 1 (ent=0.03) | 27.36 | 37.00 | **-26%** |
| track4 | `models/ppo_20260205_205241` | 4 | 30.63 | 30.59 | ~0% |

Result: Reward shaping gives a modest +7% on simple, is neutral on supersimple/square_test/track4, and actively harmful on track2 (-26%). Track2's collapse with reward shaping may stem from interaction with higher entropy (0.03) needed for that track.

### Batch D: Higher Entropy for track3 (ent_coef=0.03)

| Track | Config | Run dir | Validation (3 ep, std config) | Baseline | Delta |
|-------|--------|---------|-------------------------------|----------|-------|
| track3 | standard | `models/ppo_20260205_205246` | 16.66 | 23.13 | **-28%** |
| track3 | reward_exp | `models/ppo_20260205_205256` | 16.61 | 23.13 | **-28%** |

Result: Higher entropy that rescued track2 actively hurts track3. The two tracks have fundamentally different exploration needs.

### Batch B: Grid/CNN Obs on track3 (INCOMPLETE)

Grid runs were killed due to CPU overload from parallel execution (~80 fps vs ~300 for lidar). Partial results at time of interruption:

| Track | Config | Run dir | Last eval (incomplete) | Steps reached |
|-------|--------|---------|------------------------|---------------|
| track3 | grid | `models/ppo_20260205_205153` | 12.24 @ 380k | ~380k/1M |
| track3 | grid+reward_exp | `models/ppo_20260205_205157` | 8.83 @ 260k | ~260k/1M |
| track3 | grid+reward_exp+ent=0.03 | `models/ppo_20260205_205301` | -1.21 @ 260k | ~260k/1M |

These runs need to be rerun sequentially to get valid results.

### v4 Summary

| Experiment | Result |
|------------|--------|
| Reward shaping on simple | **+7%** (32.68 vs 30.59) |
| Reward shaping on supersimple | ~0% (47.26 vs 47.31) |
| Reward shaping on square_test | ~0% (64.02 vs 63.92) |
| Reward shaping on track2 | **-26%** (27.36 vs 37.00) |
| Reward shaping on track4 | ~0% (30.63 vs 30.59) |
| Combined (rwd+2048) on track1 | ~0% (32.75 vs 32.80) |
| Combined (rwd+2048) on track3 | -99% collapsed |
| Higher entropy on track3 | -28% (16.66 vs 23.13) |
| Higher entropy + rwd on track3 | -28% (16.61 vs 23.13) |
| Grid obs on track3 | INCOMPLETE -- rerun needed |

**Key findings:**
- Reward shaping is track-specific: helps track1 (+30% in v3) and simple (+7%), neutral on easy tracks, harmful on track2.
- Combining multiple modifications does not stack -- reward shaping + larger rollouts = same as reward shaping alone (track1) or collapse (track3).
- Higher entropy (0.03) helps track2 but hurts track3. These tracks have opposite exploration needs.
- Track3 remains resistant to every PPO modification tested across v3 and v4. Grid obs is the last untested approach but runs are incomplete.
- No new records set in v4. Track1's 32.80 from v3 remains the only improvement over v2 baselines.

## Notes
- All config snapshots and READMEs are in `Good Models/` (current physics).
- Model weights (`best_model.zip`) are present locally but excluded from git by `.gitignore`.
- Legacy results and configs live in `legacy/Good Models/`.
- v4 experiment configs created: `simple_reward_exp.yaml`, `supersimple_reward_exp.yaml`, `square_test_reward_exp.yaml`, `track2_reward_exp.yaml`, `track4_reward_exp.yaml`, `track3_grid.yaml`, `track3_grid_reward_exp.yaml`.
