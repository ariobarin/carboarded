# Good Models Index (Current Physics)

Baselines for the current physics model using the custom track configs in `racing_sim/configs/custom_tracks/`.

**Legacy models:** See `legacy/Good Models/README.md` for the historical results and legacy physics snapshots.
**Note:** Model weights (`.zip`) are not committed to git.

| Model | Track | Reward | Steps (best) | Obs |
|-------|-------|--------|--------------|-----|
| PPO Simple Custom Track - 30.59 Reward | simple | 30.59 | 440k | lidar |
| PPO Supersimple Custom Track - 66.14 Reward | supersimple | 66.14 | ~410k | grid |
| PPO Square Test Custom Track - 63.92 Reward | square_test | 63.92 | ~790k | lidar |
| PPO Track1 Custom Track - 37.23 Reward | track1 | 37.23 | ~470k | grid |
| PPO Track2 Custom Track - 37.00 Reward | track2 | 37.00 | 480k | lidar |
| PPO Track3 Custom Track - 26.23 Reward | track3 | 26.23 | ~210k | grid |
| PPO Track4 Custom Track - 42.68 Reward | track4 | 42.68 | 500k | grid |

## Multi-Track Models

| Model | Tracks | Mean Reward | Steps (best) |
|-------|--------|-------------|--------------|
| PPO Multi-Track Top3 - 31.74 Mean Reward | square_test, supersimple, track2 | 31.74 | ~190k |

All models validated with `validate.py --episodes 3 --deterministic`.

## Superseded Models (kept for reference)

| Model | Track | Reward | Superseded by |
|-------|-------|--------|---------------|
| PPO Supersimple Custom Track - 47.31 Reward | supersimple | 47.31 | 66.14 (grid/CNN) |
| PPO Track1 Custom Track - 32.80 Reward | track1 | 32.80 | 37.23 (grid/CNN) |
| PPO Track3 Custom Track - 23.13 Reward | track3 | 23.13 | 26.23 (grid/CNN) |
| PPO Track4 Custom Track - 30.59 Reward | track4 | 30.59 | 42.68 (grid/CNN) |
