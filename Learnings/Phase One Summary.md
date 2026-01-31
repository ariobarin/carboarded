# Phase One: Racing RL Training - Summary

**Date:** January 2026  
**Goal:** Speed up convergence while maintaining reward quality and avoiding collapse.

---

## Key Findings

1. **Progress reward is the main convergence accelerator.** Adding `progress_reward_scale` (0.5-0.7) dramatically speeds up learning.

2. **Lower LR + modest entropy stabilizes PPO.** Use `lr=3e-3` with `ent_coef=0.02-0.04` instead of defaults (`lr=1e-2`, `ent_coef=0.05`).

3. **Harder tracks need higher entropy.** Simple track: `ent_coef=0.02`, Wavy V1: `0.03`, Wavy V2: `0.04`.

4. **SAC beats PPO on simple tracks** when using `gradient_steps=8` (305 reward at 20k vs PPO's 250 at 30k).

5. **SAC requires auto entropy with default target.** Fixed entropy coefficients (0.05-0.2) don't work. Use `--ent-coef auto` without custom `--target-entropy`.

6. **Gradient steps scale inversely with track difficulty.** `gradient_steps=8` works for simple tracks, `gradient_steps=4` works for wavy tracks.

7. **Curriculum learning doesn't help SAC.** Pretrained models lose their learned policy when fine-tuned on harder tracks.

8. **SAC with gradient_steps=4 matches PPO on Wavy V1!** Achieved 209 at 40k (vs PPO's 220 at 30k).

9. **SAC can learn Wavy V2 with gradient_steps=4.** Achieved 154 at 100k (vs PPO's 225) - first time SAC >100 on Wavy V2!

---

## What Worked / What Didn't

| Approach | Result |
|----------|--------|
| Progress reward shaping (0.5-0.7) | Worked - main convergence driver |
| Lower LR (3e-3 vs 1e-2) | Worked - more stable |
| Higher entropy for harder tracks | Worked - prevents collapse |
| SAC gradient_steps=8 (simple track) | Worked - faster than PPO |
| SAC auto entropy (default target) | Worked - enables learning |
| Lowering clip_range (0.1, 0.05) | Failed - slows learning |
| Time penalty | Failed - hurts convergence |
| Higher progress (0.8, 0.9) | Failed - unstable |
| Speed bonus, checkpoint reward boosts | Failed - no improvement |
| Collision penalty changes | Failed - no improvement |
| Gamma/GAE tweaks | Failed - broke learning |
| SAC fixed entropy (0.05-0.2) | Failed - no learning |
| SAC custom target entropy (-0.5) | Partial - worse than default on hard tracks |
| SAC gradient_steps=8 on Wavy V2 | Failed - too aggressive |
| SAC gradient_steps=4 on Wavy V1 | **Worked** - 209 at 40k, matches PPO! |
| SAC gradient_steps=4 on Wavy V2 | **Worked** - 154 at 100k, best SAC result! |
| SAC curriculum learning | Failed - policy destroyed |

---

## Recommended Configurations

### PPO Simple Track
- **Config:** `racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml`
- **Overrides:** `--learning-rate 0.003 --ent-coef 0.02`
- **Verified:** 217 at 30k, **246 at 80k** (stable)

### PPO Wavy V1 (waves=3, waviness=0.06)
- **Config:** `racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml`
- **Overrides:** `--learning-rate 0.003 --ent-coef 0.03`
- **Verified:** 216 at 30k, **223 at 80k** (stable)

### PPO Wavy V2 (waves=5, waviness=0.08)
- **Config:** `racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Overrides:** `--learning-rate 0.003 --ent-coef 0.04`
- **Verified:** 216 at 50k, **225 at 80k** (stable)

### SAC Simple Track (BEST - faster than PPO)
- **Config:** `racing_sim/configs/fast_iter_v3_complex_sac_bootstrap.yaml`
- **Overrides:** `--learning-rate 0.003 --ent-coef auto --gradient-steps 8 --learning-starts 0 --batch-size 256 --buffer-size 200000 --n-envs 4 --vec-env subproc`
- **Verified:** **420 at 25k** (best), 409 at 50k (some instability but recovered)
- Note: SAC shows typical dip-and-recover pattern but achieves highest rewards

### SAC Wavy V1 (NEW - matches PPO!)
- **Config:** `racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml`
- **Overrides:** `--learning-rate 0.003 --ent-coef auto --gradient-steps 4 --learning-starts 0 --batch-size 256 --buffer-size 200000 --n-envs 4 --vec-env subproc`
- **Verified:** **209 at 40k/50k** (matches PPO's 220!), some instability at 60k/90k but recovers
- Model: `racing_sim/models/sac_fast_20260129_125514/best/best_model.zip`

### SAC Wavy V2 (NEW - first SAC >100!)
- **Config:** `racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Overrides:** `--learning-rate 0.003 --ent-coef auto --gradient-steps 4 --learning-starts 0 --batch-size 256 --buffer-size 200000 --n-envs 4 --vec-env subproc`
- **Verified:** **154 at 100k** (vs PPO's 225), 148 at 60k, typical SAC instability
- Model: `racing_sim/models/sac_fast_20260129_144044/best/best_model.zip`

---

## Training Commands

```bash
# PPO Simple Track (80k steps)
py scripts/train.py --algo ppo --preset fast --total-timesteps 80000 \
  --config racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.02

# PPO Wavy V1 (80k steps)
py scripts/train.py --algo ppo --preset fast --total-timesteps 80000 \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.03

# PPO Wavy V2 (80k steps)
py scripts/train.py --algo ppo --preset fast --total-timesteps 80000 \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef 0.04

# SAC Simple Track (50k steps) - FASTEST
py scripts/train.py --algo sac --preset fast --total-timesteps 50000 \
  --config racing_sim/configs/fast_iter_v3_complex_sac_bootstrap.yaml \
  --eval-freq 5000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 8 \
  --n-envs 4 --vec-env subproc

# SAC Wavy V1 (100k steps) - matches PPO!
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc

# SAC Wavy V2 (100k steps) - best SAC on hard track
py scripts/train.py --algo sac --preset fast --total-timesteps 100000 \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --eval-freq 10000 --eval-episodes 5 --learning-rate 0.003 --ent-coef auto \
  --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 4 \
  --n-envs 4 --vec-env subproc
```

---

## Play Commands

```bash
# Play with a trained model (replace MODEL_PATH with actual path)
py scripts/play.py --algo ppo --model "models/MODEL_PATH/best/best_model.zip" \
  --config racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml \
  --episodes 5 --deterministic

# Play archived good models
py scripts/play.py --algo ppo \
  --model "Good Models/Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04/best_model.zip" \
  --config racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml \
  --episodes 5 --deterministic
```

---

## SAC Stability Improvement Plan Results (January 29, 2026)

**Goal:** Improve SAC stability on Wavy V2 to close the gap with PPO (225).

### Phase 1 Experiments (100k steps each)

| Experiment | Setting | Best Eval | Final (100k) | Stability |
|------------|---------|-----------|--------------|-----------|
| Baseline | default | 154 (100k) | 154 | Reference |
| 1B | tau=0.002 | **14.6** | 13.5 | FAILED - too slow |
| 1C | batch_size=512 | **209** (90k) | 131 | Unstable (169->14->209->131) |
| 1D | lr=0.001 | **158** (90k) | 146 | STABLE (~8% drop) |

**Findings:**
- **tau=0.002**: Complete failure. Slower target network updates prevented learning.
- **batch_size=512**: Highest peak (209 - 36% above baseline) but highly unstable with wild swings.
- **lr=0.001**: Matched baseline with better end stability. Most consistent learner.

### Phase 2 Combined Experiment (150k steps)

Settings: `--learning-rate 0.001 --batch-size 512 --buffer-size 500000 --gradient-steps 2`

| Step | Eval Reward |
|------|-------------|
| 20k | -15.32 |
| 40k | -17.49 |
| 60k | -15.34 |
| 80k | 14.53 |
| 100k | 1.27 |
| 120k | **16.66** (best) |
| 140k | -2.78 |

**Result: FAILED** - Best only 16.66, **89% below baseline (154)**.

The combined settings catastrophically interfered:
- Larger buffer (500k) + slower lr (0.001) + fewer gradient steps (2) = too conservative
- Model learned to survive (ep_len 2000 at 100k) but not make progress
- High critic loss early (2e+03 to 5e+03) indicated value function instability

### Key Learnings

1. **Hyperparameter combinations don't work additively** - settings that work individually can destroy performance when combined.
2. **batch_size=512 shows most promise** - achieved 209 peak, but needs original lr (0.003), not slower.
3. **Lower lr alone (0.001) provides stability but not improvement** - matched baseline but didn't exceed it.
4. **Phase 2 was too conservative** - all changes slowed learning; needed fewer simultaneous changes.

### Recommendations for Future SAC Wavy V2 Experiments

1. Try batch_size=512 with original settings (lr=0.003, grad_steps=4, buffer=200k)
2. If trying lr changes, use 0.002 instead of 0.001 (less aggressive reduction)
3. Change only ONE hyperparameter at a time from baseline
4. Consider PPO for production use on Wavy V2 until SAC stability improves

---

## Phase 2: SAC Wavy V2 Improvement Experiments (January 30, 2026)

**Goal:** Beat PPO baseline (225 at 80k) with SAC using same training budget.

### PPO Reference (Target)
- 216 at 50k steps
- **225 at 80k steps** (stable)

### Experiment Results (all at 80k steps)

| Experiment | Settings | 30k | 50k | 80k | Peak | Status |
|------------|----------|-----|-----|-----|------|--------|
| **A: batch_size=512** | bs=512, gs=4 | 145 | 14.7 | **193** | 169 | Below target, unstable |
| **B: TD3** | - | - | - | - | - | Not supported in train.py |
| **C: grad_steps=8** | bs=256, gs=8 | 128 | 27.2 | **111** | 169 | Failed - worst final |
| **D: Combined** | bs=512, gs=8 | -8.4 | 145 | **128** | 149 | Failed - most unstable |

### Key Observations

1. **None of the SAC configurations beat PPO on Wavy V2.**
   - Best SAC final: 193 (Exp A) vs PPO: 225
   - Best SAC peak: 169 (Exp A & C at 40k/60k)

2. **All SAC runs showed significant instability:**
   - Exp A: 169 @ 40k -> 14.7 @ 50k -> 193 @ 80k (recovered)
   - Exp C: 128 @ 30k -> 27.2 @ 50k -> 111 @ 80k (degraded)
   - Exp D: 49 @ 20k -> -8.4 @ 30k -> 149 @ 40k -> 38.4 @ 60k -> 128 @ 80k (wild swings)

3. **Larger batch sizes help peaks but not stability.**
   - batch_size=512 achieved highest peaks (169, 149) but couldn't maintain them.

4. **Higher gradient_steps=8 hurts Wavy V2:**
   - Confirms earlier finding: grad_steps=4 is the sweet spot for hard tracks.
   - grad_steps=8 causes over-fitting/instability.

5. **TD3 not available** - train.py only supports PPO and SAC.

### Conclusion

**SAC cannot match PPO on Wavy V2 within 80k steps.** The fundamental issue is SAC's instability on complex tracks. While SAC can reach competitive peaks (169-193), it consistently loses performance through mid-training collapses that PPO avoids.

**Recommendation:** Use PPO for Wavy V2 production workloads. SAC remains better for simple tracks where it converges 2x faster than PPO.

---

## Next Steps (Future Work)

1. ~~Try SAC with `gradient_steps=4` on Wavy V1/V2~~ **DONE - works on V1**
2. ~~Try SAC stability improvements (tau, batch_size, lr)~~ **DONE - batch_size=512 best peak**
3. ~~Try batch_size=512 with original lr on Wavy V2~~ **DONE - 193 at 80k, still below PPO**
4. ~~Consider TD3~~ **Not available in train.py**
5. **SAC can't beat PPO on Wavy V2** - use PPO for hard tracks
6. Future: Try SAC with prioritized experience replay (PER) for stability
7. Future: Try ensemble critics for SAC stability

---

## Random Starts Breakthrough (January 30, 2026)

**MAJOR UPDATE:** SAC with random starts achieves **183.7 at 90k steps** - very close to 200 target!

### Results
| Algorithm | Random Start | Best Reward | Steps | Status |
|-----------|-------------|-------------|-------|---------|
| PPO baseline | No | 225 | 80k | Stable baseline |
| PPO + random | Yes | 17.7 | 30k | **Failed** |
| **SAC + random** | **Yes** | **183.7** | **90k** | **Promising!** |

### Key Insights
1. Random starts work for SAC, not PPO (SAC: 183.7, PPO: 17.7)
2. 183.7 is 82% of PPO's 225 - within striking distance of >200 goal
3. Model saved: `Good Models/SAC Wavy V2 Random Start 183.7 at 90k/`
4. Training continuing to push past 200

See `Learnings/Random Starts Research - Jan 30 2026.md` for full details.
