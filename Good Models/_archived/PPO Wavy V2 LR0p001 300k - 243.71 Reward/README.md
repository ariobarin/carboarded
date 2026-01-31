# PPO Wavy V2 LR0.001 300k - 243.71 Reward

## Results
- **Algorithm:** PPO
- **Track:** Wavy V2 (waves=5, waviness=0.08)
- **Validated Reward:** 243.71 (2000 steps) / 366.76 (3000 steps)
- **Training Steps:** 300,000 (best at 280k)
- **Checkpoints:** 212 (2000 steps) / 319 (3000 steps)
- **Mean Speed:** 160.6 (96.1% of terminal velocity)

## Config
`configs/wavy_v2_progress_0p75.yaml`

## Training Command
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --config configs/wavy_v2_progress_0p75.yaml \
  --learning-rate 0.001 --ent-coef 0.04 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

## Key Finding
Lower LR (0.001 vs 0.003) prevents policy collapse past 120k steps.
LR=0.003 peaked at 233.34 at 120k then collapsed. LR=0.001 continued
improving through 300k, reaching 243.71 at 280k with stable performance.

This model at 95% of the physics ceiling (terminal velocity = 166.7 px/sec,
theoretical max reward = ~257 at 2000 steps).

When evaluated with 3000 max_episode_steps, scores 366.76 because the
fast driving policy transfers to longer episodes.
