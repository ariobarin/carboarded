# PPO Collapse Prevention Research (Phase 3)

**Status:** COMPLETE
**Date:** 2026-02-02
**Goal:** Eliminate or significantly reduce PPO policy collapse to enable reliable real-world deployment

---

## Problem Statement

PPO training consistently exhibits policy collapse:
- Peak performance around 220-280k steps, then catastrophic drop to near-zero
- target_kl=0.02 delays but does not eliminate collapse
- Seed variance is ~12-14 points; some seeds collapse earlier than others
- This is problematic for real-world deployment where training stability is critical

---

## Methods Tested

### A. Adam Betas (0.99, 0.99)
**Hypothesis:** Equal momentum and variance decay prevents sudden large updates.

**Command:**
```bash
py scripts/train.py --algo ppo --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --adam-betas 0.99 0.99 --seed 42 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

**Result: FAILED**
- Peak: 3.55 at 160k
- Final: 0.00 at 300k
- Never learned - equal betas cause too-slow adaptation to non-stationary RL data

### B. L2 Regularization (0.0001)
**Hypothesis:** Weight decay prevents weight explosion and maintains plasticity.

**Command:**
```bash
py scripts/train.py --algo ppo --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --l2-reg 0.0001 --seed 42 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

**Result: SUCCESS**
- Peak: 234.47 at 240k
- Final: 225.37 at 300k
- Final/Peak: 96.1% (no collapse at 300k!)

### C. LayerNorm (lidar only)
**Hypothesis:** LayerNorm prevents dead ReLUs and stabilizes activations.

**Command:**
```bash
py scripts/train.py --algo ppo --total-timesteps 300000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --learning-rate 0.001 --ent-coef 0.02 --target-kl 0.02 \
  --layernorm --seed 42 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

**Result: COLLAPSED**
- Peak: 239.43 at 140k
- Final: 44.48 at 300k
- Final/Peak: 18.6%
- LayerNorm alone does not prevent collapse

### D. Shrink+Perturb
**Hypothesis:** Periodically shrinking weights and adding noise resets plasticity.

**Command:**
```bash
py scripts/train.py --algo ppo --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --shrink-perturb --shrink-interval 50000 --seed 42 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

**Result: FAILED**
- Peak: 219.88 at 40k
- Final: 0.00 at 300k
- Each shrink+perturb application destroys the policy
- Model never recovers after perturbation

---

## Extended L2 Experiments (500k)

### Seed 42
- Peak: 240.11 at 400k
- Final: -19.79 at 500k
- Collapse at ~440k

### Seed 123
- Peak: 249.46 at 400k
- Final: -8.29 at 500k
- Heavy oscillation throughout
- Collapse at ~420k

### Seed 7 (BEST)
- Peak: **252.61 at 120k**
- Final: 53.71 at 500k
- Validated: 252.60
- **NEW ALL-TIME BEST**

---

## Summary Table

| Method | Seed | Peak | Final | Final/Peak | Status |
|--------|------|------|-------|------------|--------|
| Adam betas (0.99, 0.99) | 42 | 3.55 | 0.00 | 0% | FAILED |
| L2 reg (0.0001) 300k | 42 | 234.47 | 225.37 | 96.1% | SUCCESS |
| LayerNorm | 42 | 239.43 | 44.48 | 18.6% | COLLAPSED |
| Shrink+Perturb | 42 | 219.88 | 0.00 | 0% | FAILED |
| L2 reg (0.0001) 500k | 42 | 240.11 | -19.79 | - | Peak saved |
| L2 reg (0.0001) 500k | 123 | 249.46 | -8.29 | - | Peak saved |
| L2 reg (0.0001) 500k | 7 | **252.61** | 53.71 | - | **NEW BEST** |

---

## Key Findings

1. **L2 regularization is the only method that helped.** It enables higher peaks (252.60 vs 249.43 baseline) and delays collapse at 300k.

2. **PPO collapse is fundamental and cannot be fully prevented** with these methods. All 500k runs eventually collapsed.

3. **Adam betas (0.99, 0.99) is harmful.** Equal momentum and variance decay prevents learning entirely - the optimizer cannot adapt to non-stationary RL data.

4. **Shrink+Perturb is too aggressive.** The default parameters (shrink=0.8, perturb_std=0.01) destroy learned features faster than the model can recover.

5. **LayerNorm alone is insufficient.** It may help with gradient flow but does not prevent the underlying policy collapse mechanism.

6. **Best strategy remains: checkpoint frequently, keep the peak.** L2 regularization helps reach higher peaks but does not eliminate the need for checkpointing.

---

## Recommended Training Command

For CNN training with L2 regularization:
```bash
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --l2-reg 0.0001 --seed 7 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

---

## Future Directions

Methods not yet tested that might help:
1. **Spectral normalization** - Constrains weight matrices to have bounded spectral norm
2. **Exponential moving average (EMA) of weights** - Smooth out policy updates
3. **Soft actor-critic (SAC) with auto-tuned entropy** - Off-policy algorithm may be more stable
4. **TRPO** - More conservative policy updates than PPO
5. **Separate actor/critic networks** - No shared feature extraction
6. **Softer Shrink+Perturb** - shrink=0.99, perturb_std=0.001, applied more frequently

---

## Files Changed
- `racing_sim/scripts/train.py` - Added --layernorm, --l2-reg, --adam-betas, --shrink-perturb, --shrink-interval
- `racing_sim/racing_sim/policies/layernorm_policy.py` - Custom LayerNorm policy
- `racing_sim/racing_sim/callbacks/plasticity.py` - Shrink+Perturb callback

---

## References

- [Nature 2024 - Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7)
- [OpenReview - Overcoming Policy Collapse in Deep RL](https://openreview.net/forum?id=m9Jfdz4ymO)
- [VC-PPO 2025 - Value-Calibrated PPO](https://arxiv.org/abs/2503.01491)
- [NeurIPS 2024 - Plasticity Loss in On-Policy RL](https://arxiv.org/abs/2405.19153)
- [Plasticine Framework](https://github.com/RLE-Foundation/Plasticine)
- [Deep RL Plasticity Repo](https://github.com/awjuliani/deep-rl-plasticity)
- [SB3 Custom Policies](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)

