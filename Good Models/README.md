# Good Models Index

Trained models that set records or demonstrate novel techniques. Each folder contains the model weights, config, and a README with training details.

## Active Models

| Model | Algorithm | Track | Obs Type | Reward |
|-------|-----------|-------|----------|--------|
| PPO CNN L2Reg Wavy V2 - 252.60 Reward | PPO+CNN | Wavy V2 | Grid 36x36 | **252.60** |
| PPO CNN Wavy V2 LR0.0003 - 249.43 Reward | PPO+CNN | Wavy V2 | Grid 36x36 | 249.43 |
| PPO Wavy V2 Ent0p02 LR0p001 500k - 247.26 Reward | PPO | Wavy V2 | Lidar 9-ray | 247.26 |
| Fast Iter V3 Complex Wavy V1 Progress0p5 LR3e-3 Ent0p03 | PPO | Wavy V1 | Lidar | 237.57 |
| Fast Iter V3 Complex Progress0p5 LR3e-3 Ent0p02 | PPO | Simple | Lidar | 252.49 |
| SAC Wavy V1 GradSteps4 LR3e-3 AutoEnt | SAC | Wavy V1 | Lidar | 209.09 |
| SAC Wavy V2 Random Start 183.7 at 90k | SAC | Wavy V2 | Lidar | 183.70 |

All models validated with `validate.py --episodes 1 --deterministic`.

## Archived

Superseded models are in `_archived/` with their README status set to ARCHIVED and a pointer to the replacement.
