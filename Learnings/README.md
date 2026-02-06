# Learnings Index

Experiment documentation for the racing simulation RL project.

**Start here:** [Phase One Summary.md](Phase%20One%20Summary.md) for consolidated results, then [Glossary.md](Glossary.md) if you're new to the RL terms.

---

## Active Documents

### Phase One Summary.md
Consolidated results from Phase 1. Contains validated final results table for all PPO/SAC models, optimal hyperparameters per track, LR sweep findings, entropy tuning, seed variance analysis, and PPO oscillation patterns.

### What Didnt Work.md
Anti-pattern guide. Lists every failed reward shaping approach, PPO hyperparameter dead-end, and SAC failure mode with brief explanations. Read this before trying any hyperparameter changes to avoid repeating mistakes.

### CNN Grid Research.md
CNN grid observation experiments that achieved 249.43 reward on Wavy V2, exceeding the lidar baseline (247.26). Covers grid configuration (36x36 homographic), LR sweep (0.0003 optimal for CNN), and key differences from lidar training.

### PPO Stability Research.md
Tests `target_kl` as a stability mechanism for PPO+CNN training. Key finding: `target_kl=0.02` significantly improves stability (final eval 238 vs baseline's 26) but does not prevent eventual collapse.

### CNN Stability Research.md
Tests LR scheduling (linear/cosine decay) and higher entropy for CNN training stability. Key finding: PPO eval oscillation is fundamental and not fixable by LR scheduling or entropy tuning. LR=0.001 is too high for CNN even with aggressive decay.

### PPO Collapse Prevention Research.md
Tests four methods to prevent PPO collapse: Adam betas, L2 regularization, LayerNorm, and Shrink+Perturb. Key finding: L2 regularization (weight_decay=0.0001) is the only method that helped, enabling the all-time best reward of 252.60.

### Custom Tracks Research.md
Track4 baseline experiments on custom tracks after centerline-based progress reward update. Compares random_start on/off and entropy increase, with early-stop outcomes and next-step recommendations.

### Custom Track Baselines 2026-02-05.md
Baseline PPO runs on all current custom track configs using the simple.yaml protocol. Includes run directories, best eval steps, and validation results.

### Multi-Track PPO Baseline 2026-02-05.md
Baseline multi-track PPO run across simple + track1-4 using config-list cycling. Includes per-track validation results.

### Glossary.md
Definitions of RL/ML terms used throughout the project documentation. Covers algorithms (PPO, SAC), training metrics (approx_kl, entropy), stability concepts (plasticity loss, weight decay), observation types, and reward components.

---

## Archive

Raw experiment data and incomplete research in `_archived/`:

- **Overnight Experiment Results.md** -- Full 15-experiment LR/entropy sweep data (pre-merge source for Phase One Summary)
- **Validation Results.md** -- Per-model 100-episode validation details (pre-merge source for Phase One Summary)
- **Phase One - Experiment Log (Archive).md** -- 20KB raw experiment log with per-step results for all Phase 1 runs
