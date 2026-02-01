# Phase One Summary

**Date:** January 2026
**Goal:** Train PPO and SAC agents on increasingly difficult 2D racing tracks.

---

## Final Results (Validated)

All models validated with `validate.py --episodes 100 --deterministic` on 2026-01-30.

| Algorithm | Track | Best Reward | Steps | Model |
|-----------|-------|-------------|-------|-------|
| PPO | Simple | **252.49** | 80k | Good Models/Fast Iter V3 Complex Progress0p5.../ |
| PPO | Wavy V1 | **237.57** | 80k | Good Models/Fast Iter V3 Complex Wavy V1.../ |
| PPO | Wavy V2 | **247.26** | 220k | Good Models/PPO Wavy V2 Ent0p02 LR0p001 500k.../ |
| SAC | Wavy V1 | 209.09 | 40k | Good Models/SAC Wavy V1 GradSteps4.../ |
| SAC | Wavy V2 | 183.70 | 90k | Good Models/SAC Wavy V2 Random Start.../ |

All PPO models: 0.00 std dev, 100% success rate (>200). Deterministic mode produces identical rewards every episode.

---

## Key Discoveries

### 1. Progress reward shaping is the primary convergence driver
`progress_reward_scale` of 0.5-0.75 dramatically speeds learning. Values >= 0.8 cause instability. 0.75 is the proven maximum for Wavy V2.

### 2. Optimal hyperparameters per track (PPO)

| Parameter | Simple | Wavy V1 | Wavy V2 |
|-----------|--------|---------|---------|
| learning_rate | 0.003 | 0.003 | 0.001 |
| ent_coef | 0.02 | 0.03 | 0.02 |
| progress_scale | 0.5 | 0.5 | 0.75 |

### 3. Learning rate is settled for Wavy V2 (LR sweep, 15 experiments)

| LR | Best Peak | Collapse Timing | Notes |
|----|-----------|-----------------|-------|
| 0.0003 | 224.32 | Oscillates, no full collapse | Most stable, lowest ceiling |
| **0.001** | **247.26** | ~280-320k | **Best peaks** |
| 0.002 | 237.86 | ~220k | Fast collapse |
| 0.003 | 233.34 | ~120k | Earliest collapse |

### 4. Entropy 0.02 is optimal for Wavy V2

| Entropy | Best Peak | Notes |
|---------|-----------|-------|
| 0.01 | 238.16 | Too little exploration |
| **0.02** | **247.26** | Optimal |
| 0.04 | 243.71 | Slightly too much |
| 0.06 | 227.53 | Way too much |

### 5. All PPO runs oscillate and collapse
Every PPO experiment shows the same pattern: rapid improvement (0-100k), peak (100k-300k), collapse (200k-500k). This is fundamental to PPO on this task. Strategy: save snapshots via eval callback, keep the peak.

### 6. Seed variance is 12-14 points
Seed=42 vs seed=7 accounts for a 13.5-point range at ent=0.02. Run multiple seeds and keep the best model.

### 7. SAC vs PPO trade-offs
- SAC beats PPO on simple tracks (420 at 25k vs 250 at 30k with gradient_steps=8)
- SAC cannot match PPO on Wavy V2 (183.7 vs 247.26) within comparable budgets
- SAC requires `--ent-coef auto` with default target entropy (custom targets hurt performance)
- SAC `gradient_steps` scales inversely with track difficulty: 8 for simple, 4 for wavy
- Random starts help SAC (183.7) but destroy PPO (17.7)
- Curriculum learning (fine-tuning on harder tracks) destroys SAC policy
- SAC is 10-20x slower per step than PPO

### 8. Failed approaches (summary)
- Time penalty, slowdown penalty, collision penalty changes: no improvement
- clip_range reduction (0.1, 0.05): slows learning
- Gamma/GAE tweaks: broke learning
- Progress >= 0.8: unstable
- Quality preset (n_steps=2048): no improvement over fast preset
- Combining multiple hyperparameter changes: catastrophic interference

See `Learnings/What Didnt Work.md` for the full anti-pattern guide.

---

## Best Training Commands

```bash
# PPO Wavy V2 (lidar) -- 247.26 reward
py scripts/train.py --algo ppo --preset fast --total-timesteps 500000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --learning-rate 0.001 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5

# SAC Wavy V2 (random start) -- 183.70 reward
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --random-start --ent-coef auto --gradient-steps 4 --learning-starts 0
```

---

## Archived Detail Files

Raw experiment data preserved in `Learnings/_archive/`:
- `Overnight Experiment Results.md` -- full 15-experiment LR/entropy sweep
- `Validation Results.md` -- per-model validation details
- `Phase One - Experiment Log (Archive).md` -- 20KB raw experiment log
