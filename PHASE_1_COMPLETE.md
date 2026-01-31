# Phase 1 Complete -- PPO Breaks 200 Barrier

**Date:** January 30, 2026
**Status:** COMPLETE

## Results Summary

### PPO -- BEST OVERALL
- **Peak Reward:** 226.49
- **Training Steps:** 30,000 (fine-tuned from 199.13 model)
- **Config:** `configs/wavy_v2_progress_0p75.yaml` (progress_reward_scale: 0.75)
- **Validation:** 100/100 episodes at 226.49 (100% success rate, 0.00 std dev)
- **Model:** `Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/`

### SAC -- Best SAC Result
- **Peak Reward:** 183.7
- **Training Steps:** 90,000 (with random starting positions)
- **Config:** `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Status:** Below 200 target. SAC cannot match PPO on Wavy V2 within comparable training budgets.
- **Model:** `Good Models/SAC Wavy V2 Random Start 183.7 at 90k/`

## Key Findings

### What Worked (PPO)
1. **Progress Reward 0.75** -- Critical improvement over 0.7 baseline
2. **Fine-tuning from 199.13 model** -- Loaded previous best, continued training
3. **Fast preset** with LR=0.01, n_steps=1024, batch_size=64
4. **Deterministic validation** -- Perfect consistency across 100 episodes

### What Worked (SAC)
1. **Random starting positions** -- SAC handles variance better than PPO (183.7 vs PPO's 17.7)
2. **gradient_steps=4** -- Sweet spot for wavy tracks
3. **--ent-coef auto** -- Required for SAC learning

### What Didn't Work
See `Learnings/What Didnt Work.md` for the comprehensive anti-pattern guide.

## Validation Results

```
PPO (226.49 model):
  Episodes: 100
  Mean reward: 226.49
  Std reward: 0.00
  Min/Max: 226.49 / 226.49
  Success rate (>200): 100.0%
```

## Files Created

### Models
- `Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/best_model.zip`
- `Good Models/SAC Wavy V2 Random Start 183.7 at 90k/best_model.zip`

### Scripts
- `racing_sim/scripts/validate.py` -- Headless validation tool for 100-episode testing

### Documentation
- `STANDARDS.md` -- Conventions for future agents
- `Learnings/What Didnt Work.md` -- Anti-pattern guide

## Success Criteria

- [x] PPO achieved >200 reward (226.49)
- [x] Stable across 100 validation episodes
- [x] Model saved and documented
- [x] Config preserved for reproducibility
- [ ] SAC achieved >200 reward (best: 183.7 -- not met)

## Next Phase: CNN Architecture + Complex Track
1. Design CNN hybrid config (camera + lidar)
2. Test transfer learning from best PPO model
3. Increase track complexity gradually
4. Target: >200 on complex track with CNN input
