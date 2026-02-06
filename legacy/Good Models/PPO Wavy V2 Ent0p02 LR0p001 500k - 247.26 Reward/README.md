# PPO Wavy V2 Ent0.02 LR0.001 500k - 247.26 Reward

**Physics:** Legacy (`legacy/racing_sim`).

## Results
- **Algorithm:** PPO
- **Track:** Wavy V2 (waves=5, waviness=0.08)
- **Validated Reward:** 247.26 (2000 steps) / 370.44 (3000 steps)
- **Training Steps:** 500,000 (best at 220k)
- **Mean Speed:** ~160+ (96.2% of terminal velocity)
- **Success Rate:** 100% over 100 episodes, 0 std

## Config
`config.yaml` (legacy snapshot)

## Training Command
```bash
cd legacy/racing_sim
py scripts/train.py --algo ppo --total-timesteps 500000 \
  --config "../Good Models/PPO Wavy V2 Ent0p02 LR0p001 500k - 247.26 Reward/config.yaml" \
  --learning-rate 0.001 --ent-coef 0.02 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

## Key Finding
Lower entropy coefficient (0.02 vs 0.04) with LR=0.001 produces the highest peak
performance. Less exploration lets the policy exploit the optimal driving line more
precisely. This is the **all-time best model** at 96.2% of the theoretical maximum
(~257 at 2000 steps).

Entropy coefficient comparison (all LR=0.001, 500k steps):
- ent=0.02: **247.26** (this model)
- ent=0.04: 243.71 (previous best)
- ent=0.06: 227.53 (too much exploration, lower ceiling)

The policy still collapses after ~260k steps, but the eval callback captures the
peak at 220k. All PPO runs on this task exhibit oscillation/collapse regardless of
hyperparameters -- the best model is always a snapshot from the peak.

