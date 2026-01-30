# Handoff: Continue SAC Stability Improvement Plan

Use this prompt to guide another agent to finish the SAC stability work.

---

## Copy this prompt to the next agent

**Task: Continue and complete the SAC Stability Improvement Plan.**

**Context:** This repo is a racing RL sim (Gymnasium env). We are improving SAC stability and convergence on the hardest track (Wavy V2). The full plan is in `.cursor/plans/sac_stability_improvement_20ca138b.plan.md` — **do not edit the plan file**. Follow the experiment order and success metrics defined there.

**What is already done:**
1. **Code:** `--tau` CLI argument was added to `racing_sim/scripts/train.py` (SAC target-network soft-update coefficient). No further code changes are required for the plan.
2. **Experiment 1A (gradient_steps=2):** Completed. SAC Wavy V2 with `--gradient-steps 2` was run for 100k steps. Result: best eval **159 at 70k**, but unstable (final ~11.6 at 100k). Baseline for comparison: previous best with gradient_steps=4 was **154 at 100k**.

**What you must do (in order):**

1. **Experiment 1B** — Run SAC Wavy V2 with `--tau 0.002` (all other Phase 1 defaults: lr=0.003, batch_size=256, buffer_size=200000, gradient_steps=4). Total steps: 100k. Run from repo root or `racing_sim/`; config: `configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`. Use `--n-envs 4 --vec-env subproc`, `--no-progress --no-tensorboard`, and the same eval/ent-coef/learning-starts as in the plan.

2. **Experiment 1C** — Same setup, but `--batch-size 512` (no tau override). 100k steps.

3. **Experiment 1D** — Same setup, but `--learning-rate 0.001` (no batch-size override). 100k steps.

4. **Analyze Phase 1** — Compare 1A, 1B, 1C, 1D on: (a) best eval reward, (b) stability (no eval drop >50% from peak). Pick the best single change(s) for Phase 2.

5. **Phase 2** — Run one combined experiment with the best settings from Phase 1. Plan suggests: `--learning-rate 0.001 --batch-size 512 --buffer-size 500000 --gradient-steps 2`, 150k steps. Adjust if your analysis suggests a different combination.

6. **Save results** — If Phase 2 beats the current best SAC Wavy V2 (154), copy the best model (e.g. `models/<run>/best/best_model.zip` and optionally `*_final.zip`) into `Good Models/` with a clear folder name and a README.md (config used, command, best reward, stability notes). Update `Learnings/Phase One Summary.md` with: new SAC Wavy V2 numbers, which levers helped stability, and any updated training/play commands.

**Success metrics (from plan):**
- Stability: no eval drop >50% from peak during training.
- Performance: aim for >180 reward on Wavy V2 (PPO gets 225).
- Convergence: first >150 reward within 60k steps if possible.

**Important:** Training runs are long (each 100k-step run can take ~30+ minutes). Do not stop with experiments unfinished. Mark off todos as you complete each experiment and the analysis/Phase 2/save steps. Use the existing todo list (e.g. `exp-1b-tau`, `exp-1c-batchsize`, `exp-1d-lr`, `analyze-phase1`, `exp-phase2`, `save-results`) and update their status as you go.

**Reference:** Baseline SAC Wavy V2 command (gradient_steps=4) is in `Learnings/Phase One Summary.md`. All experiments use that config and env; only the listed hyperparameters change per experiment.
