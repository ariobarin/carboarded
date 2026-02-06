# Legacy Physics Snapshot

This folder is a self-contained snapshot of the pre-"new physics" code used to produce the published scores. It is intentionally isolated under `legacy/` so it does not affect the main package in `racing_sim/`.

Use this snapshot when:
- Reproducing the results in `legacy/Good Models/`
- Validating the reward numbers reported in the root README

Why it exists:
- The physics implementation in `racing_sim/` has changed since the original experiments.
- The legacy configs alone are not enough to reproduce the original scores.

## Setup

```bash
cd legacy/racing_sim
uv venv
uv pip install -e .
```

## Reproduce A Model

Each model folder in `legacy/Good Models/` contains a config snapshot and the exact training command. Run the command from `legacy/racing_sim` so it uses the legacy physics.

Example (PPO Simple, 80k steps):

```bash
py scripts/train.py --algo ppo --total-timesteps 80000 \
  --config configs/fast_iter_v3_complex_progress_0p5.yaml \
  --learning-rate 0.003 --ent-coef 0.02 \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5
```

## Validate

```bash
py scripts/validate.py --model models/<run>/best_model.zip \
  --config configs/fast_iter_v3_complex_progress_0p5.yaml \
  --episodes 1 --deterministic
```

## More Details

See `legacy/racing_sim/README.md` for the original README and `legacy/racing_sim/analysis/` for archived plots and reports.
