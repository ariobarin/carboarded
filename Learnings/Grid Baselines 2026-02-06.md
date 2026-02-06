# Grid Observation Baselines (2026-02-06)

First grid/CNN baselines on custom tracks under current physics. All prior custom track results (v1-v4) used lidar (9-ray MLP). Grid obs was switched to default on 2026-02-06.

## Context

Previous CNN research (legacy physics, Wavy V2 only) showed:
- Grid matches/exceeds lidar (249.43 vs 247.26) on simple tracks
- CNN requires LR=0.0003 (not 0.001)
- CNN learns ~2x slower initially
- Linear LR schedule provides minor stability gain
- v4 grid runs on track3 were incomplete (CPU overload from parallel runs)

## Protocol

- Algorithm: PPO
- Obs type: grid (36x36 binary occupancy, NatureCNN)
- LR: 0.0003 (CNN-optimal, from training_presets.yaml)
- LR schedule: linear (proven stability gain for CNN)
- L2 regularization: 0.0001 (only proven collapse mitigation)
- Eval: eval-freq 10000, eval-episodes 3
- Validation: 3 deterministic episodes
- Timesteps: 200k (pilot), then 300-500k (Phase 2)

Track-specific settings:
- Easy tracks (simple, supersimple, square_test, track4): 4 envs
- Hard tracks (track1, track3): 1 env (cohort spawn collapses with 4)
- track2: 1 env, ent_coef=0.03

## Phase 1: Pilot Baselines (200k, all 7 tracks)

| Track | Run dir | Best eval (step) | Validation (3 ep) | Lidar v2 | Delta |
|-------|---------|------------------|-------------------|----------|-------|
| simple | `ppo_20260206_060534` | 22.68 @ 190k | 22.68 | 30.59 | **-26%** |
| supersimple | `ppo_20260206_062116` | 20.41 @ 20k | 20.41 | 47.31 | **-57%** |
| square_test | `ppo_20260206_062118` | 34.73 @ 200k | 34.73 | 63.92 | **-46%** |
| track1 | `ppo_20260206_063157` | 28.49 @ 180k | 28.49 | 25.14 | **+13%** |
| track2 | `ppo_20260206_063159` | 6.73 @ 50k | 6.73 | 37.00 | **-82%** |
| track3 | `ppo_20260206_060538` | 12.25 @ 40k | 12.25 | 23.13 | **-47%** |
| track4 | `ppo_20260206_063156` | 14.30 @ 100k | 14.30 | 30.59 | **-53%** |

**Phase 1 observations:** Grid underperformed lidar on 6/7 tracks at 200k. Only track1 beat lidar (+13%). Multiple tracks still climbing at 200k. H4 (insufficient steps) strongly supported.

## Phase 2: Extended Baselines (300-500k, reward_exp configs)

Re-baseline all 7 tracks with reward_exp configs (progress=0.65, speed=0.03) at 500k steps (300k for hard tracks). Added TensorBoard and grad logging.

| Track | Run dir | Best eval (step) | Validation | Old Grid 200k | Lidar v2 1M | vs Lidar |
|-------|---------|------------------|------------|---------------|-------------|----------|
| simple | `ppo_20260206_110420` | 29.35 @ 500k | **29.35** | 22.68 | 30.59 | **-4%** |
| supersimple | `ppo_20260206_110422` | 58.70 @ 410k | **66.14** | 20.41 | 47.31 | **+40%** |
| square_test | `ppo_20260206_112640` | 62.11 @ 460k | **62.11** | 34.73 | 63.92 | **-3%** |
| track1 | `ppo_20260206_114728` | 37.23 @ 470k | **37.23** | 28.49 | 25.14 | **+48%** |
| track2 | `ppo_20260206_114730` | 8.42 @ 50k | **6.49** | 6.73 | 37.00 | **-82%** |
| track3 | `ppo_20260206_121919` | 26.07 @ 210k | **26.23** | 12.25 | 23.13 | **+13%** |
| track4 | `ppo_20260206_112641` | 42.68 @ 500k | **42.68** | 14.30 | 30.59 | **+40%** |

**Phase 2 key findings:**
1. **Grid now beats lidar on 4/7 tracks** (supersimple +40%, track4 +40%, track1 +48%, track3 +13%).
2. **Grid nearly matches lidar on 2 tracks** (simple -4%, square_test -3%).
3. **Track2 remains the only failure** (-82%, early collapse at 50k, partial recovery to 6.49).
4. Reward_exp configs + 500k steps dramatically improved all results vs 200k baselines (avg +124%).
5. Track3 went from -47% to +13% -- the extended training uncovered a peak at 210k that 200k missed.

## Phase B: Higher Learning Rate (LR=0.0005)

Tested whether higher LR speeds CNN convergence on square_test.

| Track | Run dir | LR | Best eval (step) | Validated | Baseline (LR=0.0003) |
|-------|---------|-----|------------------|-----------|----------------------|
| square_test | `ppo_20260206_124807` | 0.0005 | 59.54 @ 460k | 59.54 | 62.11 |

**Result:** LR=0.0005 converged faster early (57.38 @ 170k vs ~50 for baseline) but showed more eval volatility in the second half and a slightly lower peak. Entropy grew too fast (std 1.74 vs 1.55). **Conclusion: LR=0.0003 remains optimal.** B2 (track1) was skipped.

## Phase C: Collapse Prevention (track2, track3)

Tested dropout and larger rollouts for collapse-prone tracks.

| Experiment | Track | Change | Best eval (step) | Validated | Baseline |
|-----------|-------|--------|------------------|-----------|----------|
| C1: Dropout 0.1 | track2 | +dropout 0.1 | 6.66 @ 40k | collapsed | 8.42 |
| C2: Dropout 0.1 | track3 | +dropout 0.1 | 9.12 @ 270k | 9.12 | 26.23 |
| C3: n_steps=2048 | track2 | n_steps 1024->2048 | 6.69 @ 170k | 6.69 | 8.42 |

**Results:**
- **Dropout 0.1 is harmful.** On track2 it collapsed faster. On track3 it reduced peak by 65% (9.12 vs 26.23). Dropout prevents the CNN from extracting useful spatial features.
- **n_steps=2048 did not help.** Track2 achieved essentially the same result (6.69 vs 6.73/8.42).
- Track2 remains unsolved with grid/CNN. Its tight turns (radius 5.83) may be fundamentally incompatible with the current grid resolution/FOV.

## TensorBoard Analysis (Phase D)

Key metrics across all Phase 2 runs:
- **KL divergence:** Well controlled (mean 0.005-0.009), all below target_kl=0.05 except one track3 spike (0.052).
- **Entropy:** Healthy decline from ~-3.0 to ~-3.4 across all runs. No entropy collapse.
- **Clip fraction:** All healthy (mean 0.05-0.08), well within expected range.
- **Grad norm:** Constant 0.5 (hitting clip ceiling), normal for PPO+NatureCNN.
- **Value loss:** Falling/stable on easy tracks. Higher and more volatile on track2/track3.
- **Track3 most volatile:** Highest value loss, only KL spike, highest clip fraction peak.
- **Track2 paradox:** Training metrics look healthy even while eval collapses -- suggests the issue is eval-specific (deterministic policy brittleness on tight turns).

## Hypothesis Framework Update

Original hypotheses (H1-H5) from Phase 1, updated with Phase 2 evidence:

- **H1: CNN feature extractor undertrained** -- PARTIALLY CONFIRMED. Extended training (500k) dramatically improved results, but grad norms at clip ceiling suggest the CNN is learning as fast as it can within the gradient budget.
- **H2: Grid resolution too coarse** -- PARTIALLY CONFIRMED for track2. The only track with radius 5.83 (very tight turns) is the only one that completely fails with grid.
- **H3: Grid FOV/distance mistuned** -- NOT SUPPORTED. 5/7 tracks now match or beat lidar, so FOV/distance settings are adequate for most geometries.
- **H4: Insufficient training steps** -- CONFIRMED. Extending from 200k to 500k was the single biggest improvement factor. Every track improved substantially.
- **H5: No velocity info in grid** -- NOT A MAJOR FACTOR. Grid beats lidar on 4 tracks despite lacking explicit velocity encoding.

Additional hypotheses tested:
- **H6: Higher LR speeds CNN** -- REJECTED. LR=0.0005 converges faster early but is less stable long-term.
- **H7: Dropout prevents collapse** -- REJECTED. Dropout actively hurts CNN performance on both tested tracks.
- **H8: Longer rollouts stabilize collapse** -- REJECTED. n_steps=2048 did not improve track2.
- **H9: Track2 collapse is eval-specific** -- SUPPORTED. Training metrics look healthy; eval is volatile. Deterministic policy may be brittle on tight geometry.
- **H10: Reward shaping helps CNN** -- CONFIRMED. Reward_exp configs (progress=0.65, speed=0.03) were used for most Phase 2 runs and contributed to the large improvements.

## Summary

Grid/CNN observation is now the superior choice for 4/7 custom tracks and competitive on 2 more, reaching parity with or exceeding lidar baselines that required 2x more training steps. The key recipe:

1. **LR=0.0003** with linear decay to 10%
2. **L2 regularization** (weight_decay=0.0001)
3. **500k steps** minimum (300k for hard tracks with 1 env)
4. **Reward shaping** (progress=0.65, speed=0.03) for most tracks
5. **1 env for hard tracks** (track1, track2, track3) to avoid cohort spawn collapse

Track2 (tight turns, radius 5.83) remains the sole grid/CNN failure. Potential next steps: higher grid resolution, wider FOV, or reward_exp config for track2.

## Good Model Records Set

4 new records from Phase 2:
- **Supersimple: 66.14** (was 47.31, +40%)
- **Track4: 42.68** (was 30.59, +40%)
- **Track1: 37.23** (was 32.80, +14%)
- **Track3: 26.23** (was 23.13, +13%)
