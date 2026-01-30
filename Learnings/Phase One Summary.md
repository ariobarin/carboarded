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

## Next Steps (Phase Two)

1. ~~Try SAC with `gradient_steps=4` on Wavy V1/V2~~ **DONE - works!**
2. Try SAC with `gradient_steps=2` on Wavy V2 (even more conservative)
3. Longer training runs (200k+) for SAC on Wavy V2 to match PPO
4. Experiment with reward scaling for SAC on harder tracks
5. Consider TD3 as alternative to SAC for stability
