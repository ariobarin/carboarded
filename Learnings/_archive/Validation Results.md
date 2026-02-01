# Model Validation Results

Validation performed on 2026-01-30 using `validate.py` with 100 episodes in deterministic mode.

## Summary

All 6 models in `Good Models/` validated successfully. Results match or exceed documented performance.

| Model | Config | Expected | Actual | Status |
|-------|--------|----------|--------|--------|
| PPO Wavy V2 Progress 0.75 | `wavy_v2_progress_0p75.yaml` | 226.49 | **226.49** | PASS |
| SAC Wavy V2 Random Start | `fast_iter_v3_complex_wavy_v2_progress_0p7.yaml` | 183.7 | **183.70** | PASS |
| PPO Wavy V2 Progress 0.7 | `fast_iter_v3_complex_wavy_v2_progress_0p7.yaml` | 225 | **225.47** | PASS |
| SAC Wavy V1 | `fast_iter_v3_complex_wavy_v1.yaml` | 209 | **209.09** | PASS |
| PPO Wavy V1 | `fast_iter_v3_complex_wavy_v1.yaml` | 223 | **237.57** | PASS (exceeds) |
| PPO Simple | `fast_iter_v3_complex_progress_0p5.yaml` | 246 | **252.49** | PASS (exceeds) |

## Detailed Results

### PPO Wavy V2 Progress 0.75 (Priority 1)
- **Model:** `Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/best_model.zip`
- **Config:** `configs/wavy_v2_progress_0p75.yaml`
- **Mean reward:** 226.49
- **Std deviation:** 0.00
- **Success rate (>200):** 100.0%
- **Notes:** Perfect match with documented performance. Zero variance in deterministic mode.

### SAC Wavy V2 Random Start (Priority 2)
- **Model:** `Good Models/SAC Wavy V2 Random Start 183.7 at 90k/best_model.zip`
- **Config:** `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Mean reward:** 183.70
- **Std deviation:** 0.00
- **Success rate (>200):** 0.0%
- **Notes:** Perfect match. Note: validated with `random_start=False` (fixed starting position).

### PPO Wavy V2 Progress 0.7 (Priority 3)
- **Model:** `Good Models/Fast Iter V3 Complex Wavy V2 Progress0p7 LR3e-3 Ent0p04/best_model.zip`
- **Config:** `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Mean reward:** 225.47
- **Std deviation:** 0.00
- **Success rate (>200):** 100.0%
- **Notes:** Slightly exceeds documented 225. Zero variance.

### SAC Wavy V1 (Priority 4)
- **Model:** `Good Models/SAC Wavy V1 GradSteps4 LR3e-3 AutoEnt/best_model.zip`
- **Config:** `configs/fast_iter_v3_complex_wavy_v1.yaml`
- **Mean reward:** 209.09
- **Std deviation:** 0.00
- **Success rate (>200):** 100.0%
- **Notes:** Matches documented 209. Zero variance.

### PPO Wavy V1 (Priority 5)
- **Model:** `Good Models/Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03/best_model.zip`
- **Config:** `configs/fast_iter_v3_complex_wavy_v1.yaml`
- **Mean reward:** 237.57
- **Std deviation:** 0.00
- **Success rate (>200):** 100.0%
- **Notes:** Exceeds documented 223 by 14.57 points. This may indicate the documented value was from mid-training evaluation rather than final model.

### PPO Simple (Priority 6)
- **Model:** `Good Models/Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02/best_model.zip`
- **Config:** `configs/fast_iter_v3_complex_progress_0p5.yaml`
- **Mean reward:** 252.49
- **Std deviation:** 0.00
- **Success rate (>200):** 100.0%
- **Notes:** Exceeds documented 246 by 6.49 points.

## Validation Command

```bash
cd racing_sim
py scripts/validate.py \
  --model "../Good Models/[MODEL_FOLDER]/best_model.zip" \
  --config configs/[CONFIG].yaml \
  --episodes 100 --deterministic
```

## Observations

1. **Deterministic mode produces zero variance.** All models return identical rewards every episode when `deterministic=True`. This is expected behavior for deterministic policies.

2. **Two models exceed documented performance.** PPO Wavy V1 (237.57 vs 223) and PPO Simple (252.49 vs 246) perform better than their documented values. This could mean:
   - Documented values were from eval callbacks during training (noisy estimates)
   - Models continued improving after the documented checkpoint
   - Different evaluation parameters were used originally

3. **All models are reproducible.** The training pipeline produces models that can be reliably validated against their documented performance.

4. **SAC Wavy V2 Random Start validated without random starts.** The 183.7 reward was achieved with random starting positions during training, but validation uses fixed starts. The policy still performs well from the standard start position.

## Recommendations

1. Update CLAUDE.md with corrected validation scores where they differ significantly.
2. Consider re-documenting model READMEs with validated scores.
3. For future models, always validate with `validate.py` before documenting final performance.
