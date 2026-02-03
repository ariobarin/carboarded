# Repository Guidelines

## Project Structure & Module Organization
- Primary code lives in `racing_sim/racing_sim/` (envs, physics, sensors, rendering, config, utils).
- CLI entry points and workflows are in `racing_sim/scripts/` (train/play/validate/create_track).
- Tests are in `racing_sim/tests/` and follow `test_*.py` naming.
- YAML configs live in `racing_sim/configs/`; deprecated configs belong in `racing_sim/configs/deprecated/`.
- Trained artifacts are stored in `Good Models/` with a README per model folder.

## Build, Test, and Development Commands
Run commands from `racing_sim/` unless noted:
- `uv venv && uv pip install -e .` – create venv and install in editable mode.
- `py scripts/train.py --preset fast --total-timesteps 200000` – quick training run.
- `py scripts/play.py` – manual keyboard control (visualization).
- `py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 1 --deterministic` – headless validation.
- `pytest` or `pytest --cov=racing_sim` – run tests (coverage optional).
- `tensorboard --logdir logs` – inspect training curves.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, PEP 8 conventions, `snake_case` for functions/vars.
- Configs are YAML and loaded via dataclasses; avoid hardcoding env parameters.
- Model folder naming: `[Algorithm] [Track] [Key Setting] [Result]` (e.g., `PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k`).
- Documentation files must not use emoji; keep docs concise and factual.

## Testing Guidelines
- Framework: pytest with optional `pytest-cov`.
- Naming: `test_*.py` and `test_*` functions.
- Add tests for new sensors, reward logic, or CLI flags; keep env sanity checks in tests when possible.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and often scoped (e.g., `envs: fix checkpoint order`, `Add Phase 2 CNN grid observation`).
- PRs should include: a summary, commands run (e.g., `pytest`), and results or screenshots if UI/visual output changed.
- If you add or update models, include a `Good Models/<model>/README.md` and update `CLAUDE.md` when best results/configs change.

## Experiment & Config Standards
- Change one hyperparameter per experiment and validate before claiming results.
- Proven configs belong in `racing_sim/configs/`; experimental ones go to `racing_sim/configs/deprecated/` with deprecation headers.
- Archive superseded models to `Good Models/_archived/` rather than deleting.
