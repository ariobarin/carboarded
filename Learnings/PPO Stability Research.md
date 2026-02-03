# PPO Stability Research

**Goal:** Reduce violent eval oscillation and achieve stable improvement during CNN training.
**Date:** 2026-02-01
**Budget:** ~4 hours
**Baseline:** CNN 249.43 at 220k steps (LR=0.0003, ent=0.02, seed=42) with wild oscillation (249 -> 12 -> 243 -> 26)

---

## Research Findings

Key sources consulted:
- [SB3 PPO Docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [SB Issue #837 - Catastrophic reward drops](https://github.com/hill-a/stable-baselines/issues/837)
- [NeurIPS 2024 Plasticity Paper](https://arxiv.org/html/2405.19153v1)
- [SpinningUp PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

### Key Insight: target_kl

The `target_kl` parameter limits KL divergence between policy updates. When the mean KL divergence exceeds the threshold, remaining epochs in that update are skipped. This prevents the policy from diverging too far in a single update.

- SB3 default: None (no KL limit!)
- OpenAI SpinningUp default: 0.01-0.015
- Our finding: 0.02 works well for CNN, 0.01 is too aggressive

### Key Insight: n_epochs

Reducing n_epochs from 5 to 3 reduces overfitting to stale rollout data, but combined with target_kl, it over-constrains learning.

---

## Experiment Results

### Baseline Reference

| Metric | Value |
|--------|-------|
| Peak eval | **249.43** |
| Peak step | 220k |
| Final eval (300k) | 26.2 |
| n_updates | ~750 |
| Policy std | 1.08 -> collapse |

### Exp 1: target_kl=0.02

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | **242.63** | -6.8 (-2.7%) |
| Peak step | 120k | 100k earlier! |
| Final eval (300k) | 238.21 | +212 (!!!) |
| n_updates | 711 | -5% |
| Policy std | 1.18 (stable) | No collapse |

**Observations:**
- KL early stopping activates on most updates ("Early stopping at step X due to reaching max kl")
- Policy std stays at 1.18 throughout -- NO COLLAPSE
- Eval still oscillates but much less extreme: 242 -> 234 -> 59 -> 62 -> 237 -> 241 -> 5 -> 12 -> 234 -> 238
- Final eval is 238 vs baseline's 26 -- the model ends near peak instead of collapsed

**Verdict: WINNER.** target_kl=0.02 provides significant stability improvement.

### Exp 2: target_kl=0.01

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.01 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | ~59 | -190 (-76%) |
| Final eval (300k) | 52.42 | +26 |
| n_updates | 319 | -57% |
| Policy std | 1.06 | Very stable |
| Wall time | 14 min | Fastest |

**Observations:**
- Nearly every update stops at epoch 0 due to KL exceeding 0.01
- Only 319 n_updates vs 711 for target_kl=0.02
- Learning is too constrained -- policy barely improves
- The KL threshold is too aggressive for CNN training

**Verdict: FAILED.** target_kl=0.01 is too aggressive; learning cannot progress.

### Exp 3: target_kl=0.02 + n_epochs=3

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 --n-epochs 3 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | 0.00 | -249 (complete failure) |
| Final eval (300k) | 0.00 | -26 |
| Final rollout | -10 | Negative! |
| Policy std | 1.24 | Drifted high |

**Observations:**
- Complete training failure -- model never learned
- With only 3 epochs + KL early stopping, updates are too constrained
- The policy std drifted to 1.24 (high entropy, no learning signal)
- Eval stays at 0.00 for all checkpoints from 140k onwards

**Verdict: CATASTROPHIC FAILURE.** n_epochs=3 with target_kl=0.02 over-constrains learning.

### Exp 4: target_kl=0.02 + clip_range=0.15

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 --clip-range 0.15 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | 223.25 | -26.18 (-10.5%) |
| Peak step | 300k | 80k later |
| Final eval (300k) | 223.25 | +197 |
| n_updates | ~1270 | +70% |
| Policy std | 1.37-1.38 | Stable |
| Wall time | ~15 min | Similar |

**Observations:**
- Very slow learning: only 13.31 at 200k, then 43.00 at 220k, jumped to 220.68 at 280k
- The tighter clip_range=0.15 constrains policy updates too much
- More n_updates (1270 vs 711) because fewer early stops, but each update is smaller
- Final 223 is decent but worse than target_kl=0.02 alone (238)
- Eval trajectory: 13.31 (200k) -> 43.00 (220k) -> 220.68 (280k) -> 223.25 (300k)

**Verdict: WORSE.** clip_range=0.15 slows learning too much. Default clip_range=0.2 is better.

### Exp 5: target_kl=0.02 extended to 500k steps

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Exp 1 (300k) |
|--------|-------|-----------------|
| Peak eval | **247.64** | +5.01 (+2.1%) |
| Peak step | 420k | 300k later |
| Final eval (500k) | 53.52 | Collapsed |
| n_updates | ~1370 | +93% (proportional) |
| Policy std | 1.32 | Stable until collapse |
| Wall time | ~28 min | Expected |

**Observations:**
- Extended training finds HIGHER peaks: 247.64 vs 242.63 for 300k
- Eval progression: 240.62 (120k) -> 243.95 (320k) -> 247.64 (420k) -> 12.29 (480k) -> 53.52 (500k)
- Collapse happened around 460-480k steps despite target_kl
- The model DID improve past 300k: 242.63 (300k) -> 247.64 (420k)
- target_kl DELAYS collapse but doesn't prevent it entirely

**Verdict: VALUABLE.** Extended training with target_kl=0.02 can find higher peaks than 300k alone. Save checkpoints and use the best.

### Exp 6: target_kl=0.02, seed=1 (seed validation)

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 1
```

| Metric | Value | vs Exp 1 (seed=42) |
|--------|-------|-------------------|
| Peak eval | 239.68 | -2.95 (-1.2%) |
| Peak step | 140k | 20k later |
| Final eval (300k) | 0.00 | COLLAPSED |
| Policy std | 1.22 | Drifted higher |
| Wall time | ~19 min | Similar |

**Observations:**
- Seed=1 peaks at 239.68 (similar to seed=42's 242.63)
- But seed=1 COLLAPSES to 0.00 by 280k despite target_kl=0.02!
- Eval trajectory: 239.68 (140k) -> oscillation -> 0.00 (280k-300k)
- target_kl does NOT guarantee collapse prevention for all seeds
- Seed variance is still ~10+ points and collapse timing varies

**Verdict: CONCERNING.** target_kl=0.02 does not fully prevent collapse for all seeds. Seed matters.

---

## Key Findings

1. **target_kl=0.02 significantly improves stability.** Final eval 238 vs baseline's 26. The model ends near peak instead of collapsed.

2. **target_kl=0.01 is too aggressive for CNN.** Learning is too constrained; only 319 updates in 300k steps.

3. **n_epochs=3 + target_kl=0.02 is over-constrained.** Complete training failure. The combination removes too much learning signal.

4. **n_epochs=5 (default) is the right balance** when combined with target_kl=0.02.

5. **Policy std is a key stability indicator.** Baseline collapses (std diverges), target_kl=0.02 stays at 1.18 (stable).

6. **clip_range=0.15 slows learning too much.** Tighter clip_range produces more n_updates but each is smaller, resulting in slower overall learning. Default 0.2 is better.

7. **Extended training (500k) with target_kl CAN find higher peaks.** Peak improved from 242.63 (300k) to 247.64 (420k). But collapse still occurs eventually (~460k).

8. **Seed variance persists even with target_kl.** Seed=1 collapsed to 0.00 at 280k while seed=42 maintained 238 at 300k. Multi-seed runs are still essential.

9. **target_kl delays collapse but doesn't eliminate it.** Collapse is fundamental to PPO. The mitigation strategy remains: save checkpoints frequently and use the best model.

---

## Recommended Configuration

For best stability:
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

For maximum peak (with more training time):
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

Key points:
- **target_kl=0.02** is the most impactful stability parameter
- 300k steps with target_kl: peak 242.63, final 238.21 (stable)
- 500k steps with target_kl: peak 247.64 at 420k, then collapse at 480k
- Always use eval callback to save best model (collapse is still possible)
- Consider running multiple seeds to find the best performer

---

## Summary Table

| Experiment | Peak Eval | Peak Step | Final Eval | Policy std | Notes |
|------------|-----------|-----------|------------|------------|-------|
| **Baseline** (no target_kl) | **249.43** | 220k | 26.2 | collapse | Reference |
| Exp 1 (target_kl=0.02, 300k) | 242.63 | 120k | **238.21** | 1.18 | **WINNER for stability** |
| Exp 2 (target_kl=0.01) | 59 | - | 52.42 | 1.06 | Too constrained |
| Exp 3 (n_epochs=3) | 0 | - | 0 | 1.24 | Complete failure |
| Exp 4 (clip_range=0.15) | 223.25 | 300k | 223.25 | 1.37 | Too constrained |
| Exp 5 (target_kl=0.02, 500k) | **247.64** | 420k | 53.52 | 1.32 | Higher peak, late collapse |
| Exp 6 (seed=1) | 239.68 | 140k | 0.00 | 1.22 | Seed-dependent collapse |

**Conclusions:**

1. **target_kl=0.02 is the best stability parameter.** Reduces oscillation and delays collapse.
2. **Extended training (500k) finds higher peaks** (247.64 vs 242.63) but collapse still occurs.
3. **Seed variance remains significant.** Seed=1 collapsed while seed=42 did not at 300k.
4. **Best strategy:** Use target_kl=0.02, run multiple seeds, save frequent checkpoints, keep best model.
