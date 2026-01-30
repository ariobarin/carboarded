# default: PPO vs SAC (Playable Steering)

## Setup
- **Config:** `configs/default.yaml`
- **Eval:** 50 episodes, deterministic policy
- **Agents:** PPO (fast), SAC (fast), tuned rule-based, random

## Key Takeaways
- PPO stays alive for the full episode and clears **~88 checkpoints** on average.
- SAC and rule-based baselines **collide early** with low checkpoint counts.
- The default track is much harder; reward scale differs from fast_iter.

## Metrics Snapshot
| Agent | Mean Reward | Mean Checkpoints | Collision Rate | Mean Steps |
| --- | ---:| ---:| ---:| ---:|
| PPO fast | 0.86 | 88.0 | 0.00 | 1000 |
| SAC fast | -20.46 | 3.0 | 1.00 | 92 |
| Rule-based tuned | -17.37 | 2.0 | 1.00 | 61 |
| Random | -20.63 | 2.82 | 1.00 | 92 |

## Visuals
### Mean Reward
![Mean Reward](../../../plots/phase1_part1/eval_mean_reward_bar_default_ppo_sac.png)

### Collision Rate
![Collision Rate](../../../plots/phase1_part1/eval_collision_rate_bar_default_ppo_sac.png)

### Reward vs Checkpoints
![Reward vs Checkpoints](../../../plots/phase1_part1/eval_reward_vs_checkpoints_default_ppo_sac.png)

### Rule-Based Grid (Reward Surface)
![Rule-Based Heatmap](../../../plots/phase1_part1/rule_based_heatmap_default_ppo_sac.png)

## Interpretation
PPO is the only agent consistently finishing the episode. SAC needs more training and likely stronger near-wall speed penalties to learn braking instead of stopping at the wall.
