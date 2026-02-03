# PPO Wavy V2 Progress 0.75 - BEST OVERALL (226.49 at 30k)

**Status:** RECOMMENDED
**Date:** 2026-01-30
**Algorithm:** PPO
**Track:** Wavy V2 (waves=5, waviness=0.08)

## Performance
| Metric | Value |
|--------|-------|
| Best reward | 226.49 at 30k steps |
| Validation (100 ep) | mean=226.49, std=0.00 |
| Success rate (>200) | 100% (100/100 episodes) |
| Episode length | 2000 (full laps) |

## Eval History
| Steps | Reward |
|-------|--------|
| 10k | 223.01 |
| 20k | 221.78 |
| 30k | 226.49 (peak) |
| 40k | -10.62 (temporary crash) |
| 50k | 222.96 (recovered) |

## Training
**Config:** `../Good Models/_archived/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/config.yaml`
**Approach:** Fine-tuned from a 199.13-reward model (progress=0.7) with progress bumped to 0.75.
**Command (fine-tuning):**
```bash
cd racing_sim
py scripts/train.py --algo ppo --preset fast --total-timesteps 60000 \
  --config ../Good Models/_archived/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/config.yaml \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5 \
  --load-model models/ppo_fast_20260130_180303/ppo_final.zip
```

**Command (from scratch):**
```bash
cd racing_sim
py scripts/train.py --algo ppo --preset fast --total-timesteps 50000 \
  --config ../Good Models/_archived/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/config.yaml \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5
```

## Usage
```bash
cd racing_sim

# Validate (headless, 100 episodes)
py scripts/validate.py \
  --model "../Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/best_model.zip" \
  --config ../Good Models/_archived/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/config.yaml --episodes 1 --deterministic

# Play (visual)
py scripts/play.py --algo ppo \
  --model "../Good Models/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/best_model.zip" \
  --config ../Good Models/_archived/PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k/config.yaml --episodes 5 --deterministic
```

## Files
- `best_model.zip` -- Best policy (226.49 reward, 100% validation)
- `config.yaml` -- Copy of training config

## Notes
- BEST OVERALL model across all algorithms and tracks
- progress_reward_scale=0.75 was the key difference over 0.7
- Fine-tuning from 199 to 226 took only 30k additional steps
- Perfectly deterministic: 0.00 std dev across 100 episodes
- Ready for CNN transition phase
