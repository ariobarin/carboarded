# PPO Wavy V2 3k Steps 300k - 342.80 Reward

## Results
- **Algorithm:** PPO
- **Track:** Wavy V2 (waves=5, waviness=0.08)
- **Validated Reward:** 342.80 (3000 steps)
- **Training Steps:** 300,000 (best at 270k)
- **Checkpoints:** 298
- **Mean Speed:** 152.2 (91.3% of terminal velocity)

## Config
`configs/wavy_v2_progress_0p75_3k_steps.yaml`

## Training Command
```bash
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --config configs/wavy_v2_progress_0p75_3k_steps.yaml \
  --learning-rate 0.001 --ent-coef 0.04 \
  --save-freq 50000 --eval-freq 30000 --eval-episodes 5
```

## Notes
Trained directly on 3000-step episodes. Reaches 342.80 but is slower
(91.3% terminal velocity) than the 2k-step model (96.1%). The model
trained on shorter 2k episodes drives faster and actually scores higher
(366.76) when evaluated with 3000 steps. This suggests training on
shorter episodes forces better speed optimization.
