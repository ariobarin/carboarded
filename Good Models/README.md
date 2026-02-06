# Good Models Index (Current Physics)

Baselines for the current physics model using the custom track configs in `racing_sim/configs/custom_tracks/`.

**Legacy models:** See `legacy/Good Models/README.md` for the historical results and legacy physics snapshots.
**Note:** Model weights (`.zip`) are not committed to git.

| Model | Track | Reward | Steps (best) |
|-------|-------|--------|--------------|
| PPO Simple Custom Track - 30.59 Reward | simple | 30.59 | 440k |
| PPO Supersimple Custom Track - 47.31 Reward | supersimple | 47.31 | ~890k |
| PPO Square Test Custom Track - 63.92 Reward | square_test | 63.92 | ~790k |
| PPO Track1 Custom Track - 25.14 Reward | track1 | 25.14 | ~630k |
| PPO Track2 Custom Track - 37.00 Reward | track2 | 37.00 | 480k |
| PPO Track3 Custom Track - 23.13 Reward | track3 | 23.13 | ~440k |
| PPO Track4 Custom Track - 30.59 Reward | track4 | 30.59 | 640k |

## Multi-Track Models

| Model | Tracks | Mean Reward | Steps (best) |
|-------|--------|-------------|--------------|
| PPO Multi-Track Top3 - 31.74 Mean Reward | square_test, supersimple, track2 | 31.74 | ~190k |

All models validated with `validate.py --episodes 3 --deterministic`.
