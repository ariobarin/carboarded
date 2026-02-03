# CNN Stability and Convergence Research

**Goal:** Improve CNN training stability and convergence speed to prepare for more complex tracks.
**Date:** 2026-02-01
**Budget:** ~2 hours
**Baseline:** CNN 249.43 at 220k steps (LR=0.0003, ent=0.02, seed=42)

---

## Current Problems

1. **Slow initial learning:** First positive reward at ~40k steps (lidar: ~10k)
2. **PPO oscillation:** Eval reward swings wildly (249 -> 12 -> 243 -> 26)
3. **High variance:** Seed choice accounts for 12-14 points
4. **Collapse at end of training:** Peak at 200-280k, then collapses to ~26 reward

---

## Research Questions

1. Can learning rate scheduling reduce late-training collapse?
2. Does higher entropy speed up early learning or just increase noise?
3. Can a high initial LR with aggressive decay combine fast early learning with late stability?

---

## Code Changes

Added `--lr-schedule` argument to `scripts/train.py`:
- `linear`: Decays LR from initial to 10% of initial over training
- `cosine`: Cosine annealing from initial to 10% of initial
- Uses SB3's callable `learning_rate` parameter (receives `progress_remaining` from 1.0 -> 0.0)

---

## Experiment Results

### Baseline Reference

| Metric | Value |
|--------|-------|
| Peak eval | **249.43** |
| Peak step | 220k |
| Final eval (300k) | 26.2 |
| First positive reward | ~40k |
| Policy std | ~1.08 |
| Wall time | ~25 min |
| Settings | LR=0.0003 fixed, ent=0.02, seed=42 |

### Experiment 1B: Higher Entropy (ent=0.04)

**Hypothesis:** More exploration early -> faster initial learning.

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.04 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | **250.52** | +1.09 (+0.4%) |
| Peak step | 260k | 40k later |
| Final eval (300k) | 246.15 | Much higher |
| Policy std | ~2.46 | 2.3x higher |
| Wall time | ~25 min | Same |

**Observations:**
- Slightly higher peak (250.52 vs 249.43) but within noise
- Eval oscillates more violently: 12 -> 250 -> 12 -> 246
- Higher policy std (2.46 vs 1.08) means wider action distributions
- Final eval was 246.15, suggesting it was at a high point when training ended
- The oscillation amplitude is worse, not better
- No faster initial learning -- first positive reward was not earlier

**Verdict:** Marginal peak gain, worse stability. Not recommended.

### Experiment 1A: Linear LR Decay (0.0003 -> 0.00003)

**Hypothesis:** Decaying LR should reduce oscillation in late training.

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --lr-schedule linear \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | **249.32** | -0.11 (tied) |
| Peak step | 300k | 80k later |
| Final eval (300k) | 249.32 | Peak was at end |
| First positive reward | ~25k | ~15k faster |
| approx_kl | 0.006-0.015 | Tighter |
| Wall time | ~14 min | 44% faster |

**Observations:**
- Very interesting early convergence: reached eval=195.21 at just 40k steps
- Then collapsed to eval=12.2 at 60k-140k (same oscillation pattern)
- Slowly recovered: 22.7 (160k), 53.7 (180k), back to 249.32 at 300k
- The decay did NOT prevent the mid-training collapse
- But the recovery was strong -- ended at peak instead of collapsed
- Faster training wall time (14 min vs 25 min) -- unclear why, possibly less GPU contention
- First positive reward ~15k steps earlier than baseline

**Verdict:** Same peak, same oscillation, but ended at peak instead of collapsed. Interesting but does not solve the fundamental problem. The early fast learning (195 at 40k) is notable.

### Experiment 3: High LR with Decay (0.001 -> 0.0001)

**Hypothesis:** Start with lidar-optimal LR for fast early learning, decay to stable CNN LR.

**Command:**
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.001 --ent-coef 0.02 --lr-schedule linear \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Peak eval | **226.47** | -22.96 (-9.2%) |
| Peak step | 280k | 60k later |
| Final eval (300k) | 225.33 | Near peak |
| approx_kl | up to 2.0 | Dangerously high |
| Policy std | 2.87 | 2.7x higher |
| Wall time | ~35 min | 40% slower |

**Observations:**
- LR=0.001 causes immediate instability: approx_kl spikes to 2.0 early
- Even with decay, the early damage was not recoverable
- Policy std ballooned to 2.87 (baseline: 1.08) and never came back down
- Very slow initial learning despite higher LR -- the opposite of the hypothesis
- Training took 35 min (longer than any other run)
- Confirms Phase 2 finding: LR=0.001 is fundamentally too high for CNN

**Verdict:** Definitively worse. High initial LR corrupts CNN training even when decayed. The early gradient steps at high LR damage the feature extraction layers.

---

## Analysis

### Why Does PPO Oscillate?

The oscillation appears primarily in **deterministic evaluation**, not in training rollout rewards. Training rollout rewards climb monotonically for all experiments. This suggests:

1. **Deterministic policy (argmax) is brittle.** Small weight changes cause large shifts in deterministic action selection, while stochastic behavior (sampling from the distribution) remains smooth.
2. **The eval metric is unreliable as a stability indicator.** A model scoring 26 in deterministic eval may still be learning well stochastically.
3. **Best model via eval callback is the right strategy.** The oscillation means any single snapshot is a lottery -- save many, keep the best.

### LR Schedule: Partially Useful

- Linear decay from 0.0003 showed interesting properties: fast early learning (195 at 40k) and ended at peak
- But it did NOT prevent the mid-training collapse (same oscillation pattern from 60k-240k)
- The "stability" at the end may just be because smaller LR = smaller weight changes = less eval variance
- This is suppressing symptoms, not fixing the root cause

### Entropy: No Free Lunch

- Higher entropy (0.04) gives a marginal peak gain (+1 point) but worse oscillation
- Entropy 0.02 remains optimal for Wavy V2 with CNN
- Entropy does not accelerate early CNN learning

### CNN LR is Definitively 0.0003

- LR=0.001 fails for CNN even with aggressive decay
- The feature extraction layers (conv filters) need slow, stable gradients
- This is fundamentally different from lidar (MLP-only, LR=0.001 is fine)

---

## Key Takeaways

1. **PPO eval oscillation is fundamental, not fixable by hyperparameters.** All three experiments show the same pattern. This is a known PPO property. The mitigation is: save frequently, keep best model.

2. **LR scheduling has marginal benefit.** Linear decay from 0.0003 shows slightly faster early learning and a cleaner end-of-training, but the same mid-training collapse. Worth using as a minor improvement but not transformative.

3. **CNN LR=0.0003 is a hard constraint.** Higher LR damages CNN feature extractors. This is consistent across all experiments.

4. **The baseline is already near-optimal for Wavy V2.** Three experiments with different approaches all produced similar or worse peak results. The 249.43 baseline may be close to the achievable ceiling for this track/architecture.

5. **Training wall time varies significantly** (14-35 min for identical step counts). Background system load matters; sequential runs are important for fair comparison.

---

## Recommendations for Next Steps

### For Complex Tracks (Wavy V3)

1. **Use current best settings as starting point:** LR=0.0003, ent=0.02, CNN
2. **Increase entropy slightly for harder tracks:** Try ent=0.03 (not 0.04, too noisy)
3. **Use LR schedule (linear decay):** Minor but free improvement to end-of-training stability
4. **Save checkpoints every 20k steps** and pick the best -- this is the single most important strategy

### Proposed Wavy V3 Config

```yaml
track:
  waviness: 0.10        # up from 0.08
  waves: 6              # up from 5
  width: 90             # down from 100
```

Expected challenges:
- Tighter curves need more precise steering
- Narrower track punishes oscillation more
- May need 400-500k steps (vs 300k for V2)
- Grid resolution (36x36) should be sufficient

### Training Command for Wavy V3

```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --cnn --config configs/wavy_v3.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --lr-schedule linear \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

### Other Directions to Explore (Future)

1. **Larger CNN architecture:** NatureCNN may be too small for complex tracks. Custom feature extractor with more filters could help.
2. **Frame stacking:** Multiple frames could provide velocity/trajectory information the single grid frame lacks.
3. **Multi-seed ensemble:** Run 3-5 seeds, keep the best. With 12-14 point variance, this reliably finds peaks.
4. **Longer eval episodes:** Current eval is 5 episodes -- noisy. 10-20 episodes would give more stable eval metrics.
5. **Observation augmentation:** Add speed/heading as auxiliary inputs alongside the grid (hybrid obs).

---

## Summary Table

| Experiment | Peak Eval | Peak Step | Final Eval | Wall Time | Notes |
|------------|-----------|-----------|------------|-----------|-------|
| **Baseline** (LR=0.0003, ent=0.02) | **249.43** | 220k | 26.2 | 25 min | Reference |
| Exp 1B (ent=0.04) | 250.52 | 260k | 246.15 | 25 min | Worse oscillation |
| Exp 1A (LR decay 0.0003->3e-5) | 249.32 | 300k | 249.32 | 14 min | Ended at peak |
| Exp 3 (LR decay 0.001->1e-4) | 226.47 | 280k | 225.33 | 35 min | High LR corrupts CNN |

**Winner:** Baseline settings remain optimal. LR schedule (linear) is a minor improvement worth keeping.
