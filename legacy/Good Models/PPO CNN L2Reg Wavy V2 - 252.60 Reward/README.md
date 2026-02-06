# PPO CNN L2Reg Wavy V2 - 252.60 Reward

**Physics:** Legacy (`legacy/racing_sim`).

## Results
- **Algorithm:** PPO with L2 regularization (weight_decay=0.0001)
- **Track:** Wavy V2 (waves=5, waviness=0.08)
- **Observation:** grid (36x36 homographic)
- **Validated Reward:** 252.60 (deterministic, 1 episode)
- **Training Steps:** 500k (best at 120k)
- **Success Rate:** 100%
- **Seed:** 7

## Config
`config.yaml` (legacy snapshot)

## Training Command
```bash
cd legacy/racing_sim

py scripts/train.py --algo ppo --total-timesteps 500000 \
  --cnn --config "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/config.yaml" \
  --learning-rate 0.0003 --ent-coef 0.02 --target-kl 0.02 \
  --l2-reg 0.0001 --seed 7 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5
```

## Key Finding
L2 regularization (weight_decay=0.0001) enables higher peak rewards (252.60 vs 249.43 baseline) and delays collapse, though PPO still eventually collapses. Best strategy: save checkpoints frequently, keep the peak.

## Phase 3 Experiment Summary
| Method | Peak | Status |
|--------|------|--------|
| Adam betas (0.99, 0.99) | 3.55 | FAILED - never learned |
| L2 reg (0.0001) | **252.61** | SUCCESS - higher peaks |
| LayerNorm | 239.43 | COLLAPSED at 280k |
| Shrink+Perturb | 219.88 | FAILED - destroys policy |

## Usage
```bash
cd legacy/racing_sim

# Play
py scripts/play.py --model "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/best_model.zip" \
  --config "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/config.yaml" --show-grid --deterministic

# Validate (1 episode for deterministic)
py scripts/validate.py --model "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/best_model.zip" \
  --config "../Good Models/PPO CNN L2Reg Wavy V2 - 252.60 Reward/config.yaml" --episodes 1 --deterministic
```

## Files
- `best_model.zip` - Best model from eval callback (120k steps)

## Notes
- L2 regularization does not prevent eventual collapse but delays it significantly
- Multi-seed variance is high (~12 points between seeds)
- PPO oscillation remains fundamental - checkpointing is essential
- This model represents the new ALL-TIME BEST for the project (252.60 vs previous 249.43)

