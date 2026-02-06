# Development Standards

Rules and conventions for working on this project.

---

## Config Management

- Only proven, validated configs belong in `racing_sim/configs/`.
- Experimental or failed configs should be removed, not left in the tree.
- See `racing_sim/configs/README.md` for the config guide.

---

## Model Saving Conventions

### When to save to Good Models/
- Save when a model sets a new record for its algorithm/track combination.
- Save when a model demonstrates a novel technique that works (e.g., random starts).
- Do NOT save incremental improvements of less than 5% unless they demonstrate a new finding.
- When a model is superseded, remove it from `Good Models/` entirely.

### Legacy physics caveat
All existing models in `Good Models/` were trained under the legacy physics snapshot (preserved in `legacy/`). The current codebase uses updated physics parameters. New training runs will not reproduce those exact reward numbers. Treat them as reference baselines, not reproduction targets.

### Folder naming
Use descriptive names: `[Algorithm] [Track] [Key Setting] [Result]`
Examples:
- `PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k`
- `SAC Wavy V2 Random Start 183.7 at 90k`

### README template
Every model folder must have a README.md following this structure:
```
# [Descriptive Name]

**Status:** PROVEN / SUPERSEDED
**Date:** YYYY-MM-DD
**Algorithm:** PPO / SAC
**Track:** [type] ([params])

## Performance
(table with best reward, validation results)

## Training
(config file, exact command used)

## Usage
(play and validate commands)

## Files
(list of files)

## Notes
(key observations)
```

Status values:
- **PROVEN** -- Active model, validated and currently best for its track/algo.
- **SUPERSEDED** -- No longer the best; kept temporarily for reference. Include a pointer to the replacement.

---

## Testing

- Add tests for new sensors, reward logic, or CLI flags.
- See `CLAUDE.md` for test commands and directory structure.

---

## Commit and Pull Request Guidelines

- Commit messages are short, imperative, and often scoped (e.g., `envs: fix checkpoint order`).
- PRs should include: a summary, commands run (e.g., `pytest`), and results or screenshots if visual output changed.
- If you add or update models, include a `Good Models/<model>/README.md` and update docs when best results change.

---

## For Contributors: Training Experiments

### Experiment protocol
- Change ONE hyperparameter at a time from a known-good baseline.
- Minimum training steps before drawing conclusions:
  - Simple track: 30k steps
  - Wavy V1: 50k steps
  - Wavy V2: 80k steps
  - Custom tracks: 100k steps
- Run headless validation before declaring a result:
  - **Deterministic runs** (`--deterministic`): 1 episode sufficient (all episodes identical)
  - **Stochastic runs** (without `--deterministic`): 10-100 episodes to capture variance
  ```bash
  cd racing_sim
  # Deterministic - 1 episode is enough
  py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 1 --deterministic
  # Stochastic - use more episodes
  py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 100
  ```
- Record eval rewards at every checkpoint (10k intervals), not just the final value.
- Always compare against the current best model for that algorithm/track combination.

### Pre-training checklist
- **Manual-play the current config** before training to verify the sim feels correct.
- **Log and inspect action stats** (mean/variance of steering + throttle) early in training.
- **Compare apples-to-apples:** same config, reward scale, episode length across runs.
- **One active config path** per run; print it in logs and reports.
- **Keep a baseline PPO run** for each track revision to anchor progress.

### Early stopping
Do not waste compute on failing runs. Stop early and try something different.

**Stop immediately if:**
- **Ellipse/Wavy tracks:** `eval_reward < 50` after 100k steps (should be 150+ by then)
- **Custom tracks (grid obs):** `eval_reward < 20` after 100k steps (custom tracks learn slower)
- `eval_reward` drops > 100 points from peak in a single eval (catastrophic collapse)
- `approx_kl > 0.5` for 3 consecutive epochs (policy exploding)
- `explained_variance < -0.5` (value network is harmful)

**Warning signs (monitor closely):**
- `approx_kl > 0.1` (above target_kl but not catastrophic)
- `eval_reward` variance > 50 between consecutive evals
- `entropy` dropping below 0.5 (premature convergence)

**After stopping:**
1. Document why the run was stopped in the experiment notes
2. Analyze what went wrong (check TensorBoard logs)
3. Formulate a hypothesis for the next experiment
4. Do NOT repeat the same configuration -- try something different

---

## DO / DON'T Quick Reference

DO:
- Change one hyperparameter at a time
- Run validation before saving models (1 ep deterministic, 10-100 ep stochastic)
- Save exact training commands in model READMEs
- Use `--ent-coef auto` for SAC, fixed values for PPO
- Use `gradient_steps=4` for SAC on wavy tracks

DON'T:
- Use progress_reward_scale >= 0.8 (causes instability)
- Use fixed entropy for SAC (prevents learning)
- Use gradient_steps=8 on wavy tracks (over-fits)
- Fine-tune SAC models on different tracks (destroys policy)
- Combine multiple untested hyperparameter changes
- Draw conclusions from runs shorter than minimum step counts
