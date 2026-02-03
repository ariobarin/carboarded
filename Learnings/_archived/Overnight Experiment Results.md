# Overnight Experiment Results (2026-01-31)

## Executive Summary

Ran 15 experiments overnight to push PPO performance past 243.71 on Wavy V2.

**New all-time best: 247.26 reward** (ent=0.02, LR=0.001, seed=42) -- 96.2% of theoretical max (~257).
Also scores **370.44** on 3000-step episodes (up from 366.76).

Saved to: `Good Models/PPO Wavy V2 Ent0p02 LR0p001 500k - 247.26 Reward/`

---

## All Experiment Results

### Tier 1: Baseline Verification

| ID | Description | LR | Ent | Steps | Best Eval | At Step | Final Eval |
|---|---|---|---|---|---|---|---|
| 1A | PPO 500k (seed=42) | 0.001 | 0.04 | 500k | 243.71 | 280k | -8.27 (460k) |
| 1C | Cohort spawn | 0.001 | 0.04 | 300k | 221.77 | 60k | 10.97 (300k) |
| 1D | seed=123 | 0.001 | 0.04 | 300k | 231.42 | 300k | 231.42 |
| 1E | seed=7 | 0.001 | 0.04 | 300k | 241.55 | 220k | 44.28 (300k) |

**Findings:**
- LR=0.001 peaks at 280k then collapses by 320k. Running to 500k doesn't help.
- Seed variance is significant: 231.42 to 243.71 (12 point range) with ent=0.04.
- Cohort spawn from scratch is very unstable (221.77 best, severe oscillation).

### Tier 2: Learning Rate Sweep

| ID | Description | LR | Ent | Steps | Best Eval | At Step | Final Eval |
|---|---|---|---|---|---|---|---|
| 2A | LR=0.0003 | 0.0003 | 0.04 | 500k | 224.32 | 80k | 220.81 |
| 2C | LR=0.002 | 0.002 | 0.04 | 500k | 237.86 | 180k | 11.07 |
| 2E | SAC LR=0.001 | 0.001 | auto | 300k | 185.55 | 240k | 38.06 |
| 2F | SAC random start | 0.001 | auto | 300k | -7.05* | 40k* | N/A (too slow) |

*2F may not have completed full 300k. 2E completed and peaked at 185.55 (slight improvement over prior SAC record of 183.7).

**Findings:**
- **LR=0.001 is optimal.** LR=0.0003 has lower ceiling (224 vs 244). LR=0.002 collapses faster.
- LR=0.0003 is somewhat more stable (stays in 215-223 range) but never reaches high peaks.
- LR=0.002 peaks at 237.86 then fully collapses to negative scores by 260k.
- SAC is 10-20x slower than PPO per step. Not viable for quick iteration.

### Tier 3: Architecture and Entropy Variations

| ID | Description | LR | Ent | Steps | Best Eval | At Step | Final Eval |
|---|---|---|---|---|---|---|---|
| 3A | Quality preset (n_steps=2048) | 0.001 | 0.04 | 500k | 231.34 | 280k | 112.28 |
| **3D** | **ent=0.02** | **0.001** | **0.02** | **500k** | **247.26** | **220k** | 52.51 |
| 3E | ent=0.06 | 0.001 | 0.06 | 500k | 227.53 | 140k | 0.00 |

**Findings:**
- **ent=0.02 is the new optimal entropy for Wavy V2.** Beat previous best by 3.55 points.
- Quality preset (n_steps=2048) doesn't help. Same collapse pattern, lower peak.
- Higher entropy (0.06) lowers the ceiling and still collapses. Complete death by 400k.

### Tier 4: Follow-up Experiments

| ID | Description | LR | Ent | Seed | Steps | Best Eval | At Step |
|---|---|---|---|---|---|---|---|
| 4A | Even lower entropy | 0.001 | 0.01 | 42 | 500k | 238.16 | 400k |
| 4B | Repeat winner (duplicate) | 0.001 | 0.02 | 42 | 500k | 247.25 | 220k |
| 4C | Real seed variant | 0.001 | 0.02 | 7 | 500k | 233.75 | 400k |

**Findings:**
- ent=0.01 is too low: slow initial learning, peaks at 238 (below ent=0.02's 247).
- 4B was accidentally identical to 3D (default seed is 42 in train.py).
- Seed=7 with ent=0.02 only reaches 233.75 -- **13.5 points below seed=42**.
- Seed variance at ent=0.02 is HUGE (233-247 range).

---

## Key Discoveries

### 1. NEW RECORD: 247.26 (96.2% of theoretical max)
- Config: PPO, LR=0.001, ent=0.02, fast preset, seed=42
- Validated: 100 episodes, 0 std, 100% success rate
- Also 370.44 on 3000-step episodes

### 2. Optimal Entropy: 0.02 for Wavy V2
Entropy coefficient ranking (all LR=0.001):
| Entropy | Best Peak | Interpretation |
|---------|-----------|----------------|
| 0.01 | 238.16 | Too little exploration, slow convergence |
| **0.02** | **247.26** | **Optimal balance** |
| 0.04 | 243.71 | Slightly too much exploration |
| 0.06 | 227.53 | Way too much, can't exploit |

### 3. All PPO Runs Oscillate/Collapse
Every single PPO experiment exhibited the same pattern:
1. Initial learning phase (0-100k): rapid improvement
2. Peak performance (100k-300k): best model saved by eval callback
3. Collapse phase (200k-500k): policy degrades, sometimes partially recovers

This is fundamental to PPO on this task, not fixable by hyperparameter tuning. The eval callback's "best model" strategy is the correct approach.

### 4. Seed Variance Dominates
| Config | Seed=42 | Seed=7 | Seed=123 | Range |
|--------|---------|--------|----------|-------|
| ent=0.04, LR=0.001 | 243.71 | 241.55 | 231.42 | 12.3 |
| ent=0.02, LR=0.001 | 247.26 | 233.75 | N/A | 13.5 |

Seed choice accounts for 12-14 points of variance. To reliably find peak performance, run multiple seeds and keep the best.

### 5. Learning Rate is Settled
| LR | Best Peak | Collapse Timing | Stability |
|-----|-----------|-----------------|-----------|
| 0.0003 | 224.32 | Oscillates, never fully collapses | Most stable |
| **0.001** | **247.26** | ~280-320k | Best peaks |
| 0.002 | 237.86 | ~220k | Fast collapse |
| 0.003 | 233.34 | ~120k | Earliest collapse |

### 6. Larger Batch Size Doesn't Help
Quality preset (n_steps=2048, batch_size=256, n_epochs=10) peaked at 231.34 vs fast preset's 247.26. More data per update doesn't fix the oscillation.

### 7. Cohort Spawn Doesn't Work
Both fine-tuning with cohort spawn and training from scratch with cohort spawn produced poor, unstable results. The changing spawn positions create too much gradient noise for PPO's on-policy updates.

### 8. SAC is Too Slow
SAC runs at ~50-100 steps/sec vs PPO's ~800 steps/sec. For this task, PPO is dramatically more efficient. SAC experiments didn't produce meaningful results in the available time.

### 9. Default Seed Bug
`train.py` uses `--seed 42` as default. All runs without explicit `--seed` used the same seed, making them identical. This was discovered when 4B (seed=42) produced the exact same trajectory as 3D.

---

## Optimal Config (Current Best)

```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --learning-rate 0.001 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

For pushing higher: run with multiple seeds (--seed 1, 2, 3, ...) and keep the best model.

---

## What's Left to Try

1. **Multi-seed sweep:** Run 10+ seeds with ent=0.02, LR=0.001. Statistical best should be ~250+.
2. **Entropy scheduling:** Start with ent=0.04 (explore), anneal to ent=0.01 (exploit).
3. **Network architecture:** Larger policy net (256x256 vs default 64x64).
4. **PPO-specific tuning:** clip_range=0.1 (tighter updates), more epochs per rollout.
5. **SAC with patience:** Run SAC for 1M+ steps to see if it eventually surpasses PPO.
6. **CNN grid observation:** Replace lidar with 10x10 grid for richer spatial info.
