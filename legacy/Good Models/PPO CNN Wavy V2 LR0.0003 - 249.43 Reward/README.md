# PPO CNN Wavy V2 LR0.0003 - 249.43 Reward

**Physics:** Legacy (`legacy/racing_sim`).

## Results
- **Algorithm:** PPO with CnnPolicy (NatureCNN)
- **Track:** Wavy V2 (waves=5, waviness=0.08)
- **Observation:** 36x36 homographic grid (obs_type=grid)
- **Validated Reward:** 249.43 (0 std over 10 episodes)
- **Training Steps:** 300,000 (best at 220k)
- **Success Rate:** 100% over 10 episodes

## Config
`config.yaml` (legacy snapshot)

## Training Command
```bash
cd legacy/racing_sim
py scripts/train.py --algo ppo --total-timesteps 300000 \
  --cnn --config "../Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/config.yaml" \
  --learning-rate 0.0003 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

## Key Finding
CNN with homographic grid observation matches/exceeds lidar performance when using
a lower learning rate (0.0003 vs 0.001 for lidar). This is the **all-time best model**
at 249.43 reward, exceeding the previous lidar best (247.26) by 1%.

Learning rate comparison (all ent=0.02, 300k steps):
- LR=0.0003: **249.43** (this model) - stable training, policy std ~1.25
- LR=0.001: 188.21 peak - unstable, approx_kl spikes 3.0+, policy std ~3.1

CNN learns slower than lidar (first positive reward at ~66k vs ~10k for lidar) but
catches up by 220k steps. The homographic grid provides equivalent or better information
than 9-ray lidar while being more generalizable to visual inputs.

## Usage
```bash
cd legacy/racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/best_model.zip" --config "../Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/config.yaml" --show-grid --deterministic

# Validate
py scripts/validate.py --model "../Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/best_model.zip" --config "../Good Models/PPO CNN Wavy V2 LR0.0003 - 249.43 Reward/config.yaml" --episodes 1 --deterministic
```

## Files
- `best_model.zip` - Best model from eval callback at 220k steps

## Notes
- Requires `--cnn` flag or `obs_type: grid` in config for correct observation format
- Grid config: 36x36, near=30, far=200, FOV=60 degrees, camera pitch=45 degrees
- See `Learnings/CNN Grid Research.md` for full experiment log

