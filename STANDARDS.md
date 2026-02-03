# Standards for Future Agents

Rules and conventions for anyone (human or AI) working on this project.

---

## Experiment Protocol

- Change ONE hyperparameter at a time from a known-good baseline.
- Minimum training steps before drawing conclusions:
  - Simple track: 30k steps
  - Wavy V1: 50k steps
  - Wavy V2: 80k steps
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

---

## Model Saving Conventions

### When to save to Good Models/
- Save when a model sets a new record for its algorithm/track combination.
- Save when a model demonstrates a novel technique that works (e.g., random starts).
- Do NOT save incremental improvements of less than 5% unless they demonstrate a new finding.

### Folder naming
Use descriptive names: `[Algorithm] [Track] [Key Setting] [Result]`
Examples:
- `PPO Wavy V2 Progress 0.75 - 226.49 Reward at 30k`
- `SAC Wavy V2 Random Start 183.7 at 90k`

### README template
Every model folder must have a README.md following this structure:
```
# [Descriptive Name]

**Status:** RECOMMENDED / PROVEN / ARCHIVED
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

### Archiving
When a model is superseded, move it to `Good Models/_archived/` and set its README status to ARCHIVED with a reason and pointer to the replacement.

---

## Config Management

- Only proven, validated configs belong in `racing_sim/configs/`.
- Experimental, failed, or untested configs go in `racing_sim/configs/deprecated/` with a deprecation header:
  ```yaml
  # DEPRECATED - DO NOT USE
  # Reason: [specific reason]
  # Use instead: [recommended config]
  ```
- Labels: DEPRECATED (superseded), UNVALIDATED (never tested), UNSTABLE (causes problems), UNTESTED (future use).
- Currently there is 1 active config in configs/ (physics_v2.yaml). Legacy configs live in configs/deprecated/legacy_2026_02/.

---

## Documentation Rules

- No emoji in documentation files. Use plain text markers like DONE, TODO, FAILED.
- Do not create session-specific tracking files (experiment logs, execution readiness docs). These become stale immediately.
- Keep CLAUDE.md concise. It is a quick reference, not a comprehensive guide.
- Put detailed findings in `Learnings/` (one document per research topic).
- Put failure documentation in `Learnings/What Didnt Work.md`.
- Reference STANDARDS.md for conventions rather than repeating them.

---

## Pre-Training Guardrails

Before starting any training run:
- **Manual-play the current config** before training to verify the sim feels correct.
- **Log and inspect action stats** (mean/variance of steering + throttle) early in training.
- **Compare apples-to-apples:** same config, reward scale, episode length across runs.
- **One active config path** per run; print it in logs and reports.
- **Keep a baseline PPO run** for each track revision to anchor progress.
- **Default to fine-tuning from the strongest available model** unless there's a reason to re-train from scratch (e.g., dead neurons, exploding gradients, obvious policy corruption).

---

## Early Stopping Protocol

**CRITICAL: Do not waste compute on failing runs. Stop early and try something different.**

### Stop Immediately If:
- **Ellipse/Wavy tracks:** `eval_reward < 50` after 100k steps (should be 150+ by then)
- **Custom tracks (grid obs):** `eval_reward < 20` after 200k steps (custom tracks learn slower)
- `eval_reward` drops > 100 points from peak in a single eval (catastrophic collapse)
- `approx_kl > 0.5` for 3 consecutive epochs (policy exploding)
- `explained_variance < -0.5` (value network is harmful)

### Warning Signs (Monitor Closely):
- `approx_kl > 0.1` (above target_kl but not catastrophic)
- `eval_reward` variance > 50 between consecutive evals
- `entropy` dropping below 0.5 (premature convergence)

### How to Stop a Background Run:
```bash
# Find the process ID, then:
# kill <PID>    (Linux/Mac)
# taskkill /PID <PID>    (Windows)
# Or press Ctrl+C in the training terminal
```

### After Stopping:
1. Document why the run was stopped in the experiment notes
2. Analyze what went wrong (check TensorBoard logs)
3. Formulate a hypothesis for the next experiment
4. Do NOT repeat the same configuration - try something different

---

## DO / DON'T Quick Reference

DO:
- Change one hyperparameter at a time
- Run validation before saving models (1 ep deterministic, 10-100 ep stochastic)
- Save exact training commands in model READMEs
- Update CLAUDE.md when configs or best results change
- Archive superseded models rather than deleting them
- Use `--ent-coef auto` for SAC, fixed values for PPO
- Use `gradient_steps=4` for SAC on wavy tracks

DON'T:
- Use progress_reward_scale >= 0.8 (causes instability)
- Use fixed entropy for SAC (prevents learning)
- Use gradient_steps=8 on wavy tracks (over-fits)
- Fine-tune SAC models on different tracks (destroys policy)
- Combine multiple untested hyperparameter changes
- Create new root-level tracking/status documents
- Draw conclusions from runs shorter than minimum step counts
