# Learnings Index

Quick reference for all experiment documentation. Read `CLAUDE.md` first for commands, architecture, and current results.

---

## Active Documents

### Phase One Summary.md
Consolidated results from Phase 1 (January 2026). Contains validated final results table for all PPO/SAC models, optimal hyperparameters per track, LR sweep findings, entropy tuning, seed variance analysis, and PPO oscillation patterns. Start here for any training-related context.

### What Didnt Work.md
Anti-pattern guide from Phase 1. Lists every failed reward shaping approach, PPO hyperparameter dead-end, and SAC failure mode with brief explanations. Read this before trying any hyperparameter changes to avoid repeating mistakes.

### CNN Grid Research.md
Phase 2 experiment log (February 2026). Documents the CNN grid observation experiments that achieved 249.43 reward on Wavy V2, exceeding the lidar baseline (247.26). Covers grid configuration (36x36 homographic), LR sweep (0.0003 optimal for CNN), and key differences from lidar training.

---

## Archive

Raw experiment data and incomplete research in `_archive/`:

- **Overnight Experiment Results.md** -- Full 15-experiment LR/entropy sweep data (pre-merge source for Phase One Summary)
- **Validation Results.md** -- Per-model 100-episode validation details (pre-merge source for Phase One Summary)
- **Phase One - Experiment Log (Archive).md** -- 20KB raw experiment log with per-step results for all Phase 1 runs
- **CNN Stability Research.md** -- Incomplete/abandoned CNN stability improvement plan, superseded by CNN Grid Research
