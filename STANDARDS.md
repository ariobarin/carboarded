# Standards for Future Agents

Rules and conventions for anyone (human or AI) working on this project.

---

## Experiment Protocol

- Change ONE hyperparameter at a time from a known-good baseline.
- Minimum training steps before drawing conclusions:
  - Simple track: 30k steps
  - Wavy V1: 50k steps
  - Wavy V2: 80k steps
- Run 100-episode headless validation before declaring a result:
  ```bash
  cd racing_sim
  py scripts/validate.py --model MODEL_PATH --config CONFIG_PATH --episodes 100 --deterministic
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
- Currently there are 4 proven configs in configs/. Do not add new ones without validation.

---

## Documentation Rules

- No emoji in documentation files. Use plain text markers like DONE, TODO, FAILED.
- Do not create session-specific tracking files (experiment logs, execution readiness docs). These become stale immediately.
- Keep CLAUDE.md under ~200 lines. It is a quick reference, not a comprehensive guide.
- Put detailed findings in `Learnings/Phase One Summary.md` (or Phase Two when that begins).
- Put failure documentation in `Learnings/What Didnt Work.md`.
- Reference STANDARDS.md for conventions rather than repeating them.

---

## DO / DON'T Quick Reference

DO:
- Change one hyperparameter at a time
- Run 100-episode validation before saving models
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
