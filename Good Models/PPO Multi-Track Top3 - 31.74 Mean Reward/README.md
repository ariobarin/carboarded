# PPO Multi-Track Top3 - 31.74 Mean Reward

**Status:** PROVEN
**Date:** 2026-02-05
**Algorithm:** PPO
**Tracks:** square_test, supersimple, track2 (round-robin)
**Physics:** Current (`racing_sim/`)

## Performance

### Overall
| Metric | Value |
|--------|-------|
| Best eval mean reward | 31.74 at ~190k steps |
| Training steps | 1.5M (1 env, round-robin, ent_coef=0.03) |

### Per-Track Validation (3 ep, deterministic)
| Track | Multi-track Model | Individual Best | Retention |
|-------|-------------------|-----------------|-----------|
| square_test | 41.05 | 63.92 | 64% |
| supersimple | 50.37 | 47.31 | 106% |
| track2 | 3.79 | 37.00 | 10% |
| **Mean** | **31.74** | **49.41** | **64%** |

## Training
**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 1500000 \
  --config-list "configs/custom_tracks/square_test.yaml,configs/custom_tracks/supersimple.yaml,configs/custom_tracks/track2.yaml" \
  --multi-track-mode round_robin --n-envs 1 --ent-coef 0.03 \
  --eval-freq 10000 --eval-episodes 3 --no-progress
```

## Usage
```bash
cd racing_sim

# Validate on a specific track
py scripts/validate.py --model "../Good Models/PPO Multi-Track Top3 - 31.74 Mean Reward/best_model.zip" \
  --config configs/custom_tracks/supersimple.yaml --episodes 3 --deterministic
```

## Files
- `best_model.zip` -- Best eval checkpoint

## Notes
- Track selection: top 3 individual performers (square_test=63.92, supersimple=47.31, track2=37.00).
- Supersimple actually improves under multi-track training (+6% over individual), possibly benefiting from skill transfer.
- Track2 doesn't transfer well; it required higher entropy (0.03) and different optimization dynamics than the easier tracks. Task interference is the likely culprit.
- No one-hot track conditioning was used. The shared policy relies on lidar observations alone to disambiguate tracks.
- Model weights are not committed to git.
