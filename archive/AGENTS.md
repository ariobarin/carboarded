# Repository Guidelines

## Project Structure & Module Organization
- `racing_sim/` is the main Python package (Gymnasium environment and core components).
- `racing_sim/envs/`, `racing_sim/physics/`, `racing_sim/sensors/`, `racing_sim/rendering/`, and `racing_sim/config/` contain the primary modules.
- `scripts/` holds CLI entry points (`train.py`, `play.py`).
- `configs/` contains YAML configuration files (start with `configs/default.yaml`).
- `logs/` and `models/` store training outputs.

## Build, Test, and Development Commands
```bash
# create venv and install
uv venv
uv pip install -e .
uv pip install tqdm rich

# train / play
py scripts/train.py --total-timesteps 500000
py scripts/play.py

# tensorboard
tensorboard --logdir logs

# quick env sanity check
py -c "from gymnasium.utils.env_checker import check_env; from racing_sim import RacingEnv; check_env(RacingEnv())"
```
If you are not using `uv`, install with `pip install -e .`.

## Coding Style & Naming Conventions
- Python style: 4-space indentation and PEP 8 spacing.
- Naming: snake_case for functions/variables, PascalCase for classes, lowercase module names.
- Keep configuration in YAML and load via the config dataclasses in `racing_sim/config`.

## Testing Guidelines
- Test tooling: `pytest` and `pytest-cov` are listed as dev dependencies.
- No tests are checked in yet; add new tests under `tests/` using `test_*.py`.
- Run `pytest` or `pytest --cov=racing_sim` locally.

## Commit & Pull Request Guidelines
- This repo copy does not include a git history, so no established commit convention was detected.
- Use concise, imperative messages (e.g., `envs: fix checkpoint order`).
- PRs should include a short summary, commands run, and any config or training-output changes.

## Configuration & Artifacts
- `configs/default.yaml` is the baseline for environment settings; add new configs under `configs/`.
- Training runs write to `logs/` and `models/`; describe any updates to these outputs in your PR.
