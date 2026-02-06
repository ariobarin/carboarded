# Multi-Track PPO Baseline 2026-02-05

Multi-track PPO experiments using the config list mode to cycle tracks per episode.

## Experiment 1: All 5 Tracks (v1 baselines)

### Setup
- Tracks: `simple`, `track1`, `track2`, `track3`, `track4`
- Mode: `round_robin` (switch track each episode)
- Algorithm: PPO defaults (includes `weight_decay=0.0001`)
- Timesteps: 500,000
- Eval: every 20,000 steps, 1 episode
- Run dirs: `logs/ppo_20260205_062820`, `models/ppo_20260205_062820`

### Per-Track Validation (Deterministic, 1 Episode)

| Track | Reward |
| --- | --- |
| simple | 27.18 |
| track1 | -4.87 |
| track2 | 24.08 |
| track3 | -3.87 |
| track4 | 35.83 |

**Average across 5 tracks:** 15.67

The shared policy learns simple/track2/track4 but fails track1/track3 (negative reward).

---

## Experiment 2: Top 3 Tracks (v2 baselines)

### Setup
- Tracks: `square_test`, `supersimple`, `track2` (top 3 by individual v2 baseline reward)
- Mode: `round_robin`
- Algorithm: PPO with `ent_coef=0.03` (higher entropy to accommodate track2)
- Timesteps: 1,500,000
- Envs: 1
- Eval: every 10,000 steps, 3 episodes
- Run dirs: `logs/ppo_20260205_124149`, `models/ppo_20260205_124149`

### Command
```bash
cd racing_sim
py scripts/train.py --algo ppo --total-timesteps 1500000 \
  --config-list "configs/custom_tracks/square_test.yaml,configs/custom_tracks/supersimple.yaml,configs/custom_tracks/track2.yaml" \
  --multi-track-mode round_robin --n-envs 1 --ent-coef 0.03 \
  --eval-freq 10000 --eval-episodes 3 --no-progress
```

### Per-Track Validation (Deterministic, 3 Episodes)

| Track | Multi-track | Individual Best | Retention |
|-------|-------------|-----------------|-----------|
| square_test | 41.05 | 63.92 | 64% |
| supersimple | 50.37 | 47.31 | 106% |
| track2 | 3.79 | 37.00 | 10% |
| **Mean** | **31.74** | **49.41** | **64%** |

### Analysis
- **supersimple improves** under multi-track training (+6% over individual). The simpler track benefits from the richer policy learned on harder tracks.
- **square_test retains 64%** -- decent generalization from shared weights.
- **track2 fails to transfer** (10% retention). Track2 required very different training dynamics (higher entropy, specific learning rate) and collapses when sharing weights with easier tracks. This is classic task interference in shared-policy multi-task RL.

## Key Findings
- Multi-track PPO works for tracks with similar difficulty. Mixing a hard track (track2) with easy tracks (square_test, supersimple) causes the hard track to be outcompeted.
- No one-hot track conditioning was used; the shared policy relies solely on lidar observations to disambiguate tracks.
- Possible improvements: curriculum learning (start with easy, gradually add harder tracks), per-task gradient scaling, or separate value heads per track.
