# Custom Tracks Research (Track4 Baseline)

**Date:** 2026-02-03
**Goal:** Establish PPO+CNN baselines on custom tracks after centerline-based progress reward change.
**Budget:** ~8 hours (sequential runs)

---

## Why This Matters (External Research)

- Domain randomization improves sim-to-real robustness by training across varied conditions; multi-track exposure is a form of environment randomization. (Peng et al., 2017; Tobin et al., 2017)
- Theoretical and survey work shows domain randomization can improve generalization, but requires careful calibration to avoid destabilizing learning. (Tajeri et al., 2021)
- Continual/multi-task RL faces interference and forgetting; naive mixing of tasks can degrade peak per-task performance. (Kirk et al., 2021; Rolnick et al., 2019)
- SB3 supports multiple parallel environments, which can be leveraged for randomized track sampling when we design a multi-track env wrapper. (SB3 VecEnv docs)
- Racing reward design literature emphasizes dense progress rewards plus speed incentives to accelerate learning, aligning with our progress-based shaping. (Sci. Rep. 2025)

---

## Baseline Assumptions

- PPO + CNN (grid) with known-good defaults:
  - learning_rate=0.0003
  - ent_coef=0.02 (unless overridden)
  - target_kl=0.05
  - l2_reg=0.0001
- Fine-tune from the strongest available base model unless there is a reason to reset.
- Eval/save cadence: every 10k steps, 1 deterministic episode.
- Early stop (custom tracks): eval_reward < 20 after 200k steps.

---

## Experiment Log

### Exp A: Track4 baseline (random_start=true)
**Command:**
```
py scripts/train.py --total-timesteps 300000 \
  --config configs/custom_tracks/track4.yaml \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5
```
**Logs:** `logs/ppo_fast_20260202_232948`
**Models:** `models/ppo_fast_20260202_232948`
**Key evals:**
- 10k: 0
- 20k: -8
- 30k: 0
- 40k: 0
- 50k: 0
- 60k: -4
- 70k: -12
- 80k: -3.8
- 90k: -4
- 100k: -8
**Decision:** Stopped after ~137k steps due to eval < 50 at 100k (early stop rule).

---

### Exp B: Track4 baseline with random_start=false
**Config:** `configs/deprecated/custom_tracks/track4_no_random_start.yaml` (UNTESTED)
**Command:**
```
py scripts/train.py --total-timesteps 300000 \
  --config configs/deprecated/custom_tracks/track4_no_random_start.yaml \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5
```
**Logs:** `logs/ppo_fast_20260202_233954`
**Models:** `models/ppo_fast_20260202_233954`
**Key evals:**
- 20k: 0.86
- 40k: 0.22
- 50k: 6.20
- 60k: 7.20
- 70k: 15.75
- 80k: 7.34
- 90k: 7.39
- 100k: -4.35
**Decision:** Stopped at 100k due to eval < 50.

---

### Exp C: Track4 random_start=false + ent_coef=0.03
**Command:**
```
py scripts/train.py --total-timesteps 300000 \
  --config configs/deprecated/custom_tracks/track4_no_random_start.yaml \
  --ent-coef 0.03 \
  --save-freq 10000 --eval-freq 10000 --eval-episodes 5
```
**Logs:** `logs/ppo_fast_20260202_234625`
**Models:** `models/ppo_fast_20260202_234625`
**Key evals:**
- 40k: 7.39
- 50k: 7.34
- 60k: 7.36
- 70k: 9.36
- 80k: 7.22
- 90k: 96.78 (single spike)
- 100k: 13.83
**Decision:** Stopped after 100k due to eval < 50. Notable transient spike suggests stochastic/deterministic eval instability.

---

### Exp D: Track4 random_start=false to 200k (lenient early stop)
**Command:**
```
py scripts/train.py \
  --config configs/deprecated/custom_tracks/track4_no_random_start.yaml \
  --eval-freq 10000 --save-freq 10000
```
**Logs:** `logs/ppo_fast_20260203_001023`
**Models:** `models/ppo_fast_20260203_001023`
**Key evals (1 ep deterministic):**
- 10k: 0.00
- 20k: 0.86
- 50k: 6.20
- 70k: 15.75
- 100k: -4.35
- 140k: -1.67
- 160k: 156.22
- 170k: 165.01
- 180k: 98.85
- 190k: 166.30
- 200k: 7.18
**Decision:** Completed 200k. Late learning spike after ~160k, followed by drop at 200k. Best checkpoint ~190k.

---

### Exp E: Track4 with cohort spawn (random checkpoint per rollout) to 200k
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track4.yaml \
  --cohort-spawn \
  --eval-freq 10000 --save-freq 10000
```
**Logs:** `logs/ppo_fast_20260203_002106`
**Models:** `models/ppo_fast_20260203_002106`
**Key evals (1 ep deterministic):**
- 10k: 0.00
- 50k: 0.00
- 100k: 0.00
- 120k: 139.00
- 140k: 149.67
- 160k: 157.94
- 170k: 170.94
- 180k: 173.92
- 190k: 96.68
- 200k: 173.96
**Decision:** Completed 200k. Strong late-stage learning and higher stable peak than Exp D.

---

### Exp F: Track4 cohort spawn fine-tune to 300k (from Exp E best)
**Base model:** `models/ppo_fast_20260203_002106/best/best_model.zip`
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track4.yaml \
  --cohort-spawn \
  --load-model models/ppo_fast_20260203_002106/best/best_model.zip \
  --total-timesteps 300000
```
**Logs:** `logs/ppo_fast_20260203_003423`
**Models:** `models/ppo_fast_20260203_003423`
**Key evals (1 ep deterministic):**
- 20k: 139.86
- 40k: 11.62
- 100k: 162.41
- 140k: 177.56
- 180k: 14.24
- 200k: 174.52
- 220k: 180.68 (best)
- 300k: 152.46
**Decision:** Completed 300k. Peak improved vs Exp E (180.68 at 220k), but late-stage drift after 240k.

---

### Exp G: Track1 cohort spawn fine-tune to 200k (from Exp F best)
**Base model:** `models/ppo_fast_20260203_003423/best/best_model.zip`
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track1.yaml \
  --cohort-spawn \
  --load-model models/ppo_fast_20260203_003423/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_004843`
**Models:** `models/ppo_fast_20260203_004843`
**Key evals (1 ep deterministic):**
- 20k: -14.63
- 60k: 126.91
- 80k: -7.82
- 120k: 121.47
- 160k: 152.38
- 180k: 157.39 (best)
- 200k: 157.37
**Decision:** Completed 200k. Late-stage recovery and stable peak ~157.

---

### Exp H: Track2 cohort spawn fine-tune to 200k (from Exp F best)
**Base model:** `models/ppo_fast_20260203_003423/best/best_model.zip`
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track2.yaml \
  --cohort-spawn \
  --load-model models/ppo_fast_20260203_003423/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_005756`
**Models:** `models/ppo_fast_20260203_005756`
**Key evals (1 ep deterministic):**
- 20k: -13.21
- 60k: -12.16
- 100k: 8.38 (best)
- 140k: -13.23
- 180k: -13.34
- 200k: -13.37
**Decision:** Completed 200k. Failed to learn (best 8.38 at 100k).

---

### Exp I: Track3 cohort spawn fine-tune to 200k (from Exp F best)
**Base model:** `models/ppo_fast_20260203_003423/best/best_model.zip`
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track3.yaml \
  --cohort-spawn \
  --load-model models/ppo_fast_20260203_003423/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_010628`
**Models:** `models/ppo_fast_20260203_010628`
**Key evals (1 ep deterministic):**
- 20k: -9.08
- 80k: -10.15
- 120k: -10.22
- 160k: -10.14
- 200k: -8.96 (best)
**Decision:** Completed 200k. Failed to learn (best -8.96).

---

### Exp J: Track2 cohort spawn fine-tune to 200k (from Track1 best)
**Base model:** `models/ppo_fast_20260203_004843/best/best_model.zip`
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track2.yaml \
  --cohort-spawn \
  --load-model models/ppo_fast_20260203_004843/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_011557`
**Models:** `models/ppo_fast_20260203_011557`
**Key evals (1 ep deterministic):**
- 20k: -14.70
- 40k: -5.54
- 60k: 61.40
- 100k: -13.27
- 160k: -5.62
- 180k: 61.82 (best)
- 200k: -6.81
**Decision:** Completed 200k. Large oscillations but two strong spikes (~61).

---

### Exp K: Track3 cohort spawn fine-tune to 200k (from Track1 best)
**Base model:** `models/ppo_fast_20260203_004843/best/best_model.zip`
**Command:**
```
py scripts/train.py \
  --config configs/custom_tracks/track3.yaml \
  --cohort-spawn \
  --load-model models/ppo_fast_20260203_004843/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_012428`
**Models:** `models/ppo_fast_20260203_012428`
**Key evals (1 ep deterministic):**
- 20k: 39.11
- 40k: 38.68
- 60k: 32.49
- 80k: -10.09
- 120k: 2.95
- 160k: 47.46 (best)
- 200k: 0.47
**Decision:** Completed 200k. Some late learning but unstable; best 47.46 at 160k.

---

### Exp L: Track2 cohort spawn from scratch to 200k (grad logging)
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --total-timesteps 200000 --grad-log-freq 20000
```
**Logs:** `logs/ppo_fast_20260203_081214`
**Models:** `models/ppo_fast_20260203_081214`
**Key evals (1 ep deterministic):**
- 20k: 0.00
- 40k: 0.00
- 60k: 0.00
- 80k: 0.00
- 100k: 0.00
- 120k: 0.00
- 140k: 0.00
- 160k: 0.00
- 180k: 0.00
- 200k: 0.00
**Grad/update norms:** grad_norm ~0.5; update_norm ~3.2-3.8 (no spikes)
**Decision:** Completed 200k. No learning signal.

---

### Exp M: Track3 cohort spawn from scratch to 200k (grad logging)
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track3.yaml --cohort-spawn --total-timesteps 200000 --grad-log-freq 20000
```
**Logs:** `logs/ppo_fast_20260203_082632`
**Models:** `models/ppo_fast_20260203_082632`
**Key evals (1 ep deterministic):**
- 20k: 0.00
- 40k: 0.00
- 60k: 0.00
- 80k: 0.00
- 100k: 0.00
- 120k: -0.87
- 140k: 1.45
- 160k: 1.19
- 180k: 4.32 (best)
- 200k: -10.14
**Grad/update norms:** grad_norm ~0.5; update_norm ~2.8-3.9 (no spikes)
**Decision:** Completed 200k. Minimal learning signal.

---

### Exp N: Track2 cohort spawn fine-tune to 200k (Track1 base, freeze+clamp)
**Base model:** `models/ppo_fast_20260203_004843/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --load-model models/ppo_fast_20260203_004843/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --grad-log-freq 20000 --total-timesteps 200000
```
**Logs:** `logs/ppo_fast_20260203_083720`
**Models:** `models/ppo_fast_20260203_083720`
**Key evals (1 ep deterministic):**
- 20k: -5.65
- 40k: 61.07
- 60k: 162.71 (best)
- 80k: 76.61
- 100k: -12.13
- 120k: -5.73
- 140k: 140.09
- 160k: 67.73
- 180k: 162.54
- 200k: -5.52
**Grad/update norms:** grad_norm ~0.5; update_norm ~3.2-5.1 (no spikes)
**Decision:** Completed 200k. Spiky learning persists; clamp+freeze did not stabilize.

---

### Exp O: Track3 cohort spawn fine-tune to 200k (Track1 base, freeze+clamp)
**Base model:** `models/ppo_fast_20260203_004843/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track3.yaml --cohort-spawn --load-model models/ppo_fast_20260203_004843/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --grad-log-freq 20000 --total-timesteps 200000
```
**Logs:** `logs/ppo_fast_20260203_084829`
**Models:** `models/ppo_fast_20260203_084829`
**Key evals (1 ep deterministic):**
- 20k: 42.32
- 40k: 37.68
- 60k: 37.53
- 80k: 42.11
- 100k: 1.34
- 120k: 4.66
- 140k: 36.63
- 160k: 47.20
- 180k: 66.22 (best)
- 200k: -4.55
**Grad/update norms:** grad_norm ~0.5; update_norm ~2.8-5.7 (no spikes)
**Decision:** Completed 200k. Spiky, ends negative; clamp+freeze did not stabilize.

---

### Exp P: Track2 cohort spawn fine-tune to 300k (from Track2 ft best, freeze+clamp)
**Base model:** `models/ppo_fast_20260203_083720/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --load-model models/ppo_fast_20260203_083720/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --grad-log-freq 20000 --total-timesteps 300000
```
**Logs:** `logs/ppo_fast_20260203_100330`
**Models:** `models/ppo_fast_20260203_100330`
**Key evals (1 ep deterministic):**
- 20k: 61.36
- 40k: -6.90
- 60k: -5.65
- 80k: -6.88
- 100k: -12.20
- 120k: -12.13
- 140k: 60.10
- 160k: 81.30
- 180k: 76.76
- 200k: 61.58
- 220k: 61.55
- 240k: 171.79 (best)
- 260k: 61.27
- 280k: 61.23
- 300k: -13.37
**Grad/update norms:** grad_norm ~0.5; update_norm ~3.0-3.9 (no spikes)
**Decision:** Completed 300k. Best 171.79 at 240k; ends negative; still spiky but higher peak.

---

### Exp Q: Track3 cohort spawn fine-tune to 300k (from Track3 ft best, freeze+clamp)
**Base model:** `models/ppo_fast_20260203_084829/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track3.yaml --cohort-spawn --load-model models/ppo_fast_20260203_084829/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --grad-log-freq 20000 --total-timesteps 300000
```
**Logs:** `logs/ppo_fast_20260203_101900`
**Models:** `models/ppo_fast_20260203_101900`
**Key evals (1 ep deterministic):**
- 20k: 46.37
- 40k: 46.78
- 60k: 66.70 (best)
- 80k: 43.58
- 100k: -6.03
- 120k: 65.68
- 140k: 25.10
- 160k: -4.84
- 180k: -5.98
- 200k: -6.01
- 220k: -4.76
- 240k: 65.92
- 260k: -4.63
- 280k: -4.76
- 300k: 65.67
**Grad/update norms:** grad_norm ~0.5; update_norm ~2.7-4.5 (no spikes)
**Decision:** Completed 300k. Modest peaks; remains volatile.

---

### Exp R: Track2 cohort spawn fine-tune to 300k (from Track2 ft best, lr=3e-4, clamp=0.2)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.2 --learning-rate 0.0003 --grad-log-freq 20000 --total-timesteps 300000
```
**Logs:** `logs/ppo_fast_20260203_104014`
**Models:** `models/ppo_fast_20260203_104014`
**Key evals (1 ep deterministic):**
- 20k: 70.62
- 40k: 172.33 (best)
- 60k: 61.43
- 80k: -11.36
- 100k: -11.95
- 120k: -1.41
- 140k: 61.62
- 160k: -12.04
- 180k: 61.05
- 200k: -11.70
- 220k: 70.17
- 240k: 61.41
- 260k: 61.44
- 280k: 177.70 (best)
- 300k: 70.99
**Grad/update norms:** grad_norm ~0.5; update_norm ~3.0-3.3 (no spikes)
**Decision:** Completed 300k. Lower policy std (~1.22) but still spiky; stochastic mean is lower than Exp P.

---

### Exp S: Track2 cohort spawn fine-tune to 200k (target_kl=0.02)
**Base model:** `models/ppo_fast_20260203_104014/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --load-model models/ppo_fast_20260203_104014/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.2 --learning-rate 0.0003 --target-kl 0.02 --grad-log-freq 20000 --total-timesteps 200000
```
**Logs:** `logs/ppo_fast_20260203_105808`
**Models:** `models/ppo_fast_20260203_105808`
**Key evals (1 ep deterministic):**
- 20k: 178.93 (best)
- 40k: -5.48
- 60k: 2.04
- 80k: 161.84
- 100k: -5.64
- 120k: -11.54
- 140k: -11.89
- 160k: -11.96
- 180k: -12.03
- 200k: -11.58
**Grad/update norms:** grad_norm ~0.5; update_norm ~2.5; frequent KL early-stops.
**Decision:** Completed 200k. target_kl=0.02 triggers early stopping and performance collapses after early spikes.

---

### Exp T: Track2 cohort spawn fine-tune to 200k (target_kl=0.03)
**Base model:** `models/ppo_fast_20260203_104014/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --load-model models/ppo_fast_20260203_104014/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.2 --learning-rate 0.0003 --target-kl 0.03 --grad-log-freq 20000 --total-timesteps 200000
```
**Logs:** `logs/ppo_fast_20260203_111037`
**Models:** `models/ppo_fast_20260203_111037`
**Key evals (1 ep deterministic):**
- 20k: 182.47 (best)
- 40k: 69.95
- 60k: 70.79
- 80k: -5.62
- 100k: -5.77
- 120k: -8.05
- 140k: -6.84
- 160k: -8.11
- 180k: -11.33
- 200k: -8.18
**Grad/update norms:** grad_norm ~0.5; update_norm ~2.6; frequent KL early-stops.
**Decision:** Completed 200k. target_kl=0.03 still triggers early stopping; spikes remain with late collapse.

---

### Exp U: Track2 diagnostics with rollout stats (100k)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --rollout-log-freq 1 --total-timesteps 100000
```
**Logs:** `logs/ppo_fast_20260203_114811`
**Models:** `models/ppo_fast_20260203_114811`
**Key eval (1 ep deterministic):**
- 100k: -11.58
**Rollout stats (mean over 49 rollouts):**
- adv_std: 1.66 (min 0.678, max 2.95)
- adv_abs_mean: 0.885
- ret_std: 4.49, value_std: 4.24
- episode_reward_mean: 52.93, episode_reward_std: 20.26
- term_collision_rate: 0.75, term_timeout_rate: 0.25
**Decision:** Diagnostics complete. Collisions dominate terminations; advantage variance is high.

---

### Exp V: Track3 diagnostics with rollout stats (100k)
**Base model:** `models/ppo_fast_20260203_101900/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track3.yaml --cohort-spawn --load-model models/ppo_fast_20260203_101900/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --rollout-log-freq 1 --total-timesteps 100000
```
**Logs:** `logs/ppo_fast_20260203_115349`
**Models:** `models/ppo_fast_20260203_115349`
**Key eval (1 ep deterministic):**
- 100k: 88.78
**Rollout stats (mean over 49 rollouts):**
- adv_std: 1.40 (min 0.597, max 4.98)
- adv_abs_mean: 0.820
- ret_std: 5.06, value_std: 4.89
- episode_reward_mean: 31.06, episode_reward_std: 16.63
- term_collision_rate: 0.90, term_timeout_rate: 0.10
**Decision:** Diagnostics complete. Collisions dominate terminations; advantage variance remains high.

---

### Exp W: Track2 off-track penalty (200k)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Config:** `configs/custom_tracks/track2_offtrack.yaml`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2_offtrack.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --grad-log-freq 20000 --rollout-log-freq 1 --total-timesteps 200000
```
**Logs:** `logs/ppo_fast_20260203_121103`
**Models:** `models/ppo_fast_20260203_121103`
**Key evals (1 ep deterministic):**
- 20k: -847.99
- 40k: 90.40
- 60k: 0.00
- 100k: 0.00
- 140k: 0.00
- 200k: 0.00
**Rollout stats (late):** adv_std ~0.18, term_timeout_rate ~1.0, collisions ~0.0
**Decision:** Completed 200k. Off-track penalty destabilized Track2; agent times out with low reward.

---

### Exp X: Track3 off-track penalty (200k)
**Base model:** `models/ppo_fast_20260203_101900/best/best_model.zip`
**Config:** `configs/custom_tracks/track3_offtrack.yaml`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track3_offtrack.yaml --cohort-spawn --load-model models/ppo_fast_20260203_101900/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --grad-log-freq 20000 --rollout-log-freq 1 --total-timesteps 200000
```
**Logs:** `logs/ppo_fast_20260203_122140`
**Models:** `models/ppo_fast_20260203_122140`
**Key evals (1 ep deterministic):**
- 20k: 202.97
- 60k: 209.39
- 120k: 220.44
- 140k: 237.90 (best)
- 200k: 232.42
**Rollout stats (late):** adv_std ~0.62, term_timeout_rate ~1.0, collisions ~0.0
**Decision:** Completed 200k. Off-track penalty stabilizes Track3 and exceeds 200 consistently.

---

### Exp Y: Track2 off-track penalty v2 (172k, early stop)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Config:** `configs/custom_tracks/track2_offtrack_v2.yaml` (collision_penalty=-0.2, off_track_penalty=-1.0, max_off_track_steps=20)
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2_offtrack_v2.yaml --load-model models/ppo_fast_20260203_100330/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_125810`
**Models:** `models/ppo_fast_20260203_125810`
**Key evals (1 ep deterministic):**
- 20k: -8.52
- 40k: 9.39
- 60k: -378.96
- 140k: 0.00
- 160k: 0.00
**Decision:** Early-stopped at ~172k due to collapse (evals flat at 0; negative rollout rewards). Deterministic validation (1 ep) mean 0.00.

---

### Exp Z: Track2 no termination / no collision penalty (200k)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Config:** `configs/custom_tracks/track2_no_terminate.yaml` (collision_penalty=0.0, terminate_on_collision=false)
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2_no_terminate.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip
```
**Logs:** `logs/ppo_fast_20260203_131323`
**Models:** `models/ppo_fast_20260203_131323`
**Key evals (1 ep deterministic):**
- 20k: 14.26
- 40k: 174.35 (best)
- 80k: 16.19
- 100k: 174.02
- 120k: 104.10
- 140k: 103.77
- 160k: 8.01
- 180k: 8.17
- 200k: 88.33
**Decision:** Completed 200k. Spiky but not collapsed; no deterministic eval >200.

---

## Stochastic Validation (10 episodes, non-deterministic)

- Track4 FT (best): mean 122.95, std 49.83, min 12.91, max 153.84
  - Model: `models/ppo_fast_20260203_003423/best/best_model.zip`
  - Config: `configs/custom_tracks/track4.yaml`
- Track1 FT (best): mean 138.81, std 23.50, min 68.51, max 149.07
  - Model: `models/ppo_fast_20260203_004843/best/best_model.zip`
  - Config: `configs/custom_tracks/track1.yaml`
- Track2 scratch: mean -13.85, std 1.26, min -15.37, max -11.59
  - Model: `models/ppo_fast_20260203_081214/best/best_model.zip`
  - Config: `configs/custom_tracks/track2.yaml`
- Track3 scratch: mean 0.61, std 5.00, min -6.18, max 5.55
  - Model: `models/ppo_fast_20260203_082632/best/best_model.zip`
  - Config: `configs/custom_tracks/track3.yaml`
- Track2 ft freeze+clamp 300k (best): mean 114.79, std 60.08, min -5.71, max 164.03
  - Model: `models/ppo_fast_20260203_100330/best/best_model.zip`
  - Config: `configs/custom_tracks/track2.yaml`
- Track2 off-track penalty 200k: mean -651.16, std 425.97, min -925.35, max 166.29
  - Model: `models/ppo_fast_20260203_121103/best/best_model.zip`
  - Config: `configs/custom_tracks/track2_offtrack.yaml`
- Track2 no termination (best): mean 117.89, std 43.86, min 14.62, max 164.93
  - Model: `models/ppo_fast_20260203_131323/best/best_model.zip`
  - Config: `configs/custom_tracks/track2_no_terminate.yaml`
- Track3 off-track penalty 200k: mean 211.23, std 2.69, min 208.20, max 216.63
  - Model: `models/ppo_fast_20260203_122140/best/best_model.zip`
  - Config: `configs/custom_tracks/track3_offtrack.yaml`
- Track2 ft lr=3e-4 clamp=0.2 300k: mean 68.84, std 57.94, min -5.78, max 143.61
  - Model: `models/ppo_fast_20260203_104014/best/best_model.zip`
  - Config: `configs/custom_tracks/track2.yaml`
- Track2 ft target_kl=0.02 200k: mean 92.25, std 61.46, min -5.66, max 165.42
  - Model: `models/ppo_fast_20260203_105808/best/best_model.zip`
  - Config: `configs/custom_tracks/track2.yaml`
- Track2 ft target_kl=0.03 200k: mean 121.83, std 57.94, min -5.57, max 170.38
  - Model: `models/ppo_fast_20260203_111037/best/best_model.zip`
  - Config: `configs/custom_tracks/track2.yaml`
- Track3 ft freeze+clamp 300k (best): mean 60.13, std 8.79, min 46.57, max 66.14
  - Model: `models/ppo_fast_20260203_101900/best/best_model.zip`
  - Config: `configs/custom_tracks/track3.yaml`

---

## Findings

1. **Random_start=true on track4 fails to reach any positive eval by 100k.**
2. **Disabling random_start improves early learning signal** (eval hits ~7-16 by 70k), but still below success threshold by 100k.
3. **Higher entropy (0.03) produced a single high eval spike (~97 at 90k),** but the gain did not persist at 100k.
4. **Lenient early stopping was justified.** Exp D showed a major learning jump after ~160k that would have been missed under the 100k cutoff.
5. **Cohort spawn is the strongest performer so far.** Exp E reached ~174 at 200k with sustained late-stage improvements.
6. **Fine-tuning from a strong base improves peaks but still oscillates.** Exp F reached 180.68 at 220k, higher than Exp E, but drifted down by 300k.
7. **Track1 adapts reasonably from the Track4 base.** Exp G reached ~157 by 180k after volatile early phases.
8. **Track2 and Track3 do not transfer from the Track4 base.** Exp H and Exp I stayed negative through 200k.
9. **Track1 base improves Track2/3 transfer but remains unstable.** Exp J and Exp K show positive spikes but do not sustain gains to 200k.
10. **Track2/3 from scratch show little to no learning by 200k.** Exp L and Exp M stayed near zero.
11. **Freeze+clamp at 200k did not stabilize Track2/3.** Exp N and Exp O still spike and end negative.
12. **Extending Track2 to 300k improved peaks.** Exp P hit 171.79 at 240k, with a strong stochastic mean (114.79).
13. **Extending Track3 to 300k yielded only modest gains.** Exp Q peaked at 66.70 with a stochastic mean of 60.13.
14. **Lowering log_std and learning rate reduced policy std but not spikiness.** Exp R kept std ~1.22 yet deterministic evals still oscillated.
15. **Lower target_kl (0.02-0.03) caused frequent KL early-stops and regression.** Exp S and Exp T spiked early then collapsed.
16. **Diagnostics show high advantage variance.** Exp U/V adv_std averaged 1.66 (Track2) and 1.40 (Track3).
17. **Terminations are collision-dominated.** Exp U collision rate ~0.75 and Exp V ~0.90.
18. **Off-track penalty stabilizes Track3 but hurts Track2.** Exp X is consistently >200; Exp W collapses to timeouts.
19. **Track2 off-track tuning (v2) still collapses.** Reducing penalty magnitude and max off-track steps did not rescue learning (Exp Y).
20. **Removing collision termination does not stabilize Track2.** Exp Z shows early spikes and late drops; stochastic mean improves but still no >200 success.
21. **Gradients are clipped and updates are stable.** grad_norm sits at ~0.5 and update_norm stays in ~2.5-4.5 range, with no spikes.
22. **Stochastic evals are more consistent than deterministic spikes.** Track2 improves with longer training; Track3 off-track becomes stable.
23. **Deterministic evals remain volatile on custom tracks.** Exp D-Z show large spikes and dips between adjacent checkpoints.

---

## Interpretation

- For custom tracks, PPO+CNN often learns **late (120k+)**, so early-stop rules must be more lenient.
- Random start adds variance and appears harmful for PPO on these custom tracks based on observed runs.
- Cohort spawn provides randomized starting positions **without incoherent PPO gradients**, and appears to unlock stronger late-stage learning.
- Entropy increase (0.03) appears to increase exploration but also volatility; not enough evidence to keep it as default.
- Fine-tuning from a strong base model accelerates learning, but does not remove late-stage oscillation.
- Cross-track transfer is inconsistent: Track1 adapts from Track4, but Track2/3 did not, suggesting either topology mismatch or a need for different base models.
- Using the Track1 base partially helps Track2/3 but does not produce stable learning, so additional changes or longer runs may be required.
- Scratch runs failing imply the reward signal on Track2/3 is too sparse or misaligned for PPO to bootstrap without additional curriculum or shaping.
- Clamp+freeze alone does not fix volatility, which suggests the issue is not just representation drift but also policy head exploration or track-specific reward dynamics.
- Lowering log_std reduces policy variance but did not eliminate spikes, implying reward or termination structure still dominates instability.
- Very low target_kl throttles updates and leads to KL early-stops, which appears to hurt sustained learning.
- Off-track penalties change termination dynamics: Track3 becomes stable with timeouts and low collision rates, while Track2 collapses, suggesting track-specific reward sensitivity.
- Track2 remains unstable even with softer off-track penalties (v2), implying the issue is not just penalty severity but likely reward/geometry interaction or policy exploitation paths.
- No gradient explosion is evident with max_grad_norm=0.5 and stable update_norms, aligning with the weight-inspection finding that weights are not blowing up.

---

## Root Cause Analysis (Spikiness)

**Likely drivers (ordered):**
- **On-policy PPO instability:** PPO updates on freshly collected data with multiple epochs can cause non-monotonic returns; the clipped objective is not a hard trust region, so large policy shifts can still happen and performance can dip between evals.
- **High-variance advantage estimates:** GAE trades bias for variance; if the value function is imperfect, advantage noise can spike policy updates.
- **Evaluation noise:** Single-episode deterministic evals are noisy; stochasticity and episode start variation can make adjacent checkpoints look wildly different.
- **Reward/termination structure on Track2/3:** Scratch runs fail and transfer is unstable, suggesting sparse or misaligned reward and/or termination traps that amplify variance.
- **Exploration variance in the policy head:** log_std is a major contributor; clamping reduces variance but did not fully stabilize, pointing to reward/termination dynamics as deeper issues.

**Evidence from our runs:**
- Gradients are clipped and update norms are stable, so spikiness is not from exploding gradients.
- Lowering log_std and learning rate reduced policy std but did not remove spikes.
- Very low target_kl (0.02-0.03) triggered frequent KL early-stops and performance collapse after early spikes.
- Track2 improved with longer training (300k), Track3 did not, implying Track3 is more reward/structure-limited.

---
## Weight Inspection (2026-02-03)

We inspected PPO+CNN weights for Track4 base/FT and Track1/2/3 fine-tunes. Outputs (plots + CSVs) are in `Learnings/figures/`.

**Summary stats (weights only, all models):**
- L2 norms are tightly clustered (~19.9 to 20.9).
- Max abs weights are all < 1.0 (largest observed ~0.99).
- p99(abs) sits around ~0.18-0.22.
- Bias max abs is small (< 0.19).

**Interpretation:**
- No evidence of exploding weights or runaway magnitudes.
- Instability appears behavioral (policy oscillation) rather than numerical blow-up.
- The highest max-abs parameters are typically `log_std` (policy exploration), not core CNN weights, suggesting variance tuning may be a driver of oscillations.

**Artifacts:**
- `Learnings/figures/summary.csv` (per-model aggregate stats)
- `Learnings/figures/*_hist.png` and `*_logabs_hist.png` (weight distributions)
- `Learnings/figures/*_top30_layer_l2.png` and `*_top30_layer_maxabs.png` (largest layers)
- `Learnings/figures/overall_l2.png`, `overall_max_abs.png` (cross-model comparisons)
- `Learnings/figures/weight_diff.csv` (L2 drift vs Track4 base by module)
- `Learnings/figures/weight_similarity.csv` (cosine similarity vs Track4 base by module)

**Drift and similarity highlights (vs Track4 base):**
- Track4 fine-tune (best performer) shows the *smallest* feature drift and highest cosine similarity (feature cos ~0.73, action cos ~0.84).
- Track2/3 fine-tunes from Track1 show the *largest* feature drift (diff_rel ~1.06) and lowest cosine similarity (feature cos ~0.42, action cos ~0.65).
- Action nets drift more than value nets across all fine-tunes, suggesting policy head instability rather than value head blow-up.

**Exploration parameter (`log_std`) trends:**
- Better models keep `log_std` lower (Track4 FT mean ~0.41, std ~1.5).
- Poor transfer runs have higher `log_std` (means ~0.68-0.89, std ~2.0-2.4), indicating noisier actions, which likely hurts deterministic evals.

---

## Wall Contact Penalty Research (Mode 3)

**Date:** 2026-02-03
**Goal:** Test whether physical wall bounce (non-terminal) with optional per-step penalty improves learning compared to Mode 1 (terminal collision) and Mode 2 (off-track ghost).

**Implementation:** Added `touching_wall` flag to Car (cleared each step, set by `pre_solve` callback during physics step), `wall_contact_penalty` and `max_wall_contact_steps` config fields, integrated in racing_env.py reward/termination logic.

**Key difference from prior experiments:** The car physically bounces off walls (losing speed) but the episode does not terminate on collision. Optional per-step penalty applied only during active contact frames.

### Exp AA: Track2 physics-only wall bounce (no penalty, no termination)
**Config:** `configs/custom_tracks/track2_wall_contact.yaml` (collision_penalty=0.0, terminate_on_collision=false, wall_contact_penalty=0.0)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2_wall_contact.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --rollout-log-freq 1 --total-timesteps 200000 --eval-freq 20000 --eval-episodes 1
```
**Logs:** `logs/ppo_fast_20260203_164550`
**Models:** `models/ppo_fast_20260203_164550`
**Key evals (1 ep deterministic):**
- 20k: 180 (best)
- 40k: 178
- 60k: 15.5
- 80k: 8.43
- 100k: 13.4
- 120k: 8.31
- 140k: 25.5
- 160k: 13.2
- 180k: 19.9
- 200k: 12.9
**Rollout stats:** term_collision_rate oscillates 0-1 (episodes end by timeout when no wall hit, by collision flag when touching).
**Decision:** Completed 200k. Early peak (180 at 20k) then collapsed. Best model saved at 20k eval.

---

### Exp AB: Track2 light wall contact penalty (-0.5/step)
**Config:** `configs/custom_tracks/track2_wall_contact_light.yaml` (collision_penalty=0.0, terminate_on_collision=false, wall_contact_penalty=-0.5)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2_wall_contact_light.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --rollout-log-freq 1 --total-timesteps 200000 --eval-freq 20000 --eval-episodes 1
```
**Logs:** `logs/ppo_fast_20260203_165704`
**Models:** `models/ppo_fast_20260203_165704`
**Key evals (1 ep deterministic):**
- 20k: -930
- 40k: 184 (best)
- 60k: -48.2
- 80k: 116
- 100k: -22.7
- 120k: 103
- 140k: 121
- 160k: -191
- 180k: -898
- 200k: 6.17
**Decision:** Completed 200k. Extremely volatile. The per-step penalty compounds massively when wall-riding (-930 at 20k, -898 at 180k). Peak 184 at 40k but wildly unstable.

---

### Exp AC: Track2 medium wall contact penalty (-1.0/step, max 60 step termination)
**Config:** `configs/custom_tracks/track2_wall_contact_medium.yaml` (collision_penalty=0.0, terminate_on_collision=false, wall_contact_penalty=-1.0, max_wall_contact_steps=60)
**Base model:** `models/ppo_fast_20260203_100330/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track2_wall_contact_medium.yaml --cohort-spawn --load-model models/ppo_fast_20260203_100330/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --rollout-log-freq 1 --total-timesteps 200000 --eval-freq 20000 --eval-episodes 1
```
**Logs:** `logs/ppo_fast_20260203_170705`
**Models:** `models/ppo_fast_20260203_170705`
**Key evals (1 ep deterministic):**
- 20k: -46.9 (ep_len=229)
- 40k: -46.9 (ep_len=231)
- 60k: 68.8 (ep_len=1960)
- 80k: -51.7 (ep_len=171)
- 100k: -52.2 (ep_len=206)
- 120k: 21.1 (ep_len=1200)
- 140k: 109 (best, ep_len=2048)
- 160k: 70.9 (ep_len=2048)
- 180k: 53.7 (ep_len=2048)
- 200k: -46.9 (ep_len=249)
**Decision:** Completed 200k. Early episodes terminate quickly from 60-step wall contact limit (ep_len ~200). Mid-training recovery (109 at 140k) with full-length episodes, then relapse. The termination acts like Mode 1 in early training but allows recovery.

---

### Exp AD: Track3 physics-only wall bounce (no penalty, no termination)
**Config:** `configs/custom_tracks/track3_wall_contact.yaml` (collision_penalty=0.0, terminate_on_collision=false, wall_contact_penalty=0.0)
**Base model:** `models/ppo_fast_20260203_101900/best/best_model.zip`
**Command:**
```
py scripts/train.py --config configs/custom_tracks/track3_wall_contact.yaml --cohort-spawn --load-model models/ppo_fast_20260203_101900/best/best_model.zip --freeze-cnn-layers 2 --log-std-min -2.0 --log-std-max 0.6 --rollout-log-freq 1 --total-timesteps 200000 --eval-freq 20000 --eval-episodes 1
```
**Logs:** `logs/ppo_fast_20260203_171637`
**Models:** `models/ppo_fast_20260203_171637`
**Key evals (1 ep deterministic):**
- 20k: 85.6
- 40k: 85.4
- 60k: 85.2
- 80k: 84.9
- 100k: 66.5
- 120k: 85.1
- 140k: 117 (best)
- 160k: 85.4
- 180k: 86.5
- 200k: 63.4
**Decision:** Completed 200k. Remarkably stable -- 8 of 10 evals between 63-117, no negative rewards, no collapse. Best 117 at 140k.

---

### Wall Contact Stochastic Validation (10 episodes, non-deterministic)

| Experiment | Track | Config | Stoch Mean | Std | Min | Max |
|------------|-------|--------|-----------|-----|-----|-----|
| AA (physics-only) | Track2 | track2_wall_contact.yaml | **136.29** | 48.74 | 14.32 | 165.01 |
| AB (light -0.5/step) | Track2 | track2_wall_contact_light.yaml | -32.06 | 343.64 | -903.00 | 163.71 |
| AC (medium -1.0/step + 60 term) | Track2 | track2_wall_contact_medium.yaml | 124.73 | **9.48** | 113.40 | 137.33 |
| AD (physics-only) | Track3 | track3_wall_contact.yaml | 95.50 | 21.85 | 62.03 | 114.37 |

**Comparison to prior experiments:**
- Track2 Exp Z (no-terminate, no-penalty, Mode 2 ghost): mean 117.89, std 43.86
- Track2 Exp AA (physics bounce, no penalty): mean **136.29**, std 48.74 -- **+16% over Exp Z**
- Track2 Exp AC (physics bounce + penalty + term): mean 124.73, std **9.48** -- **4x lower variance than Exp Z**
- Track3 Exp X (off-track ghost, Mode 2): mean 211.23, std 2.69
- Track3 Exp AD (physics bounce): mean 95.50, std 21.85 -- lower mean but still stable

---

### Wall Contact Findings

1. **Physics-only wall bounce (AA) is the new best Track2 approach.** Stochastic mean 136.29 beats all prior Track2 experiments (Exp Z=117.89, Exp P=114.79, Exp T=121.83).
2. **Per-step penalty is counterproductive on Track2.** Exp AB (-0.5/step) caused catastrophic reward swings (-930 to 184) because wall contact duration varies wildly and the penalty compounds linearly.
3. **Wall contact termination (AC) trades peak performance for consistency.** Std of 9.48 (vs AA's 48.74) but lower mean. The termination acts like a softer Mode 1 that triggers after sustained contact rather than any contact.
4. **Track3 physics bounce (AD) is stable but lower than off-track ghost.** Mean 95.50 vs Mode 2's 211.23. Track3's tighter geometry may benefit from the is_on_track sensor-based approach.
5. **Physical bounce provides implicit speed penalty.** The car loses velocity on wall contact, which reduces speed bonus and progress, acting as an implicit penalty without reward noise.
6. **No experiment reached >200 success rate.** Track2 remains below the 200 threshold that Track3 off-track achieved, suggesting Track2 geometry needs additional curriculum or reward changes.
7. **The implementation works as designed.** touching_wall flag correctly tracks per-step contact, wall_contact_penalty applies only during contact frames, and max_wall_contact_steps terminates after sustained contact.

---

### Wall Contact Interpretation

The physics-only wall bounce (Mode 3 without penalty) is the cleanest approach for Track2:
- It removes the harsh terminal signal (Mode 1) that dominated 75-90% of terminations
- It avoids the geometric confusion of ghost/sensor-only mode (Mode 2) where the car passes through walls
- The implicit speed loss from bouncing provides a natural, proportional penalty
- PPO can learn from longer episodes with consistent reward signals

For Track3, the off-track ghost approach (Mode 2, Exp X) remains superior, likely because Track3's tighter geometry means the car spends more time off-track after bouncing, and the is_on_track sensor provides a cleaner signal.

**Recommendation:** Use physics-only wall bounce (Mode 3, AA config) for Track2 fine-tuning going forward. Continue using off-track ghost (Mode 2, Exp X config) for Track3.

---

## Next Steps (Ordered)

1. **Extend Exp AA to 500k steps** to see if the higher mean survives or collapses like prior long runs.
2. **Try physics bounce + light off-track penalty (hybrid)** -- combine Mode 3 bounce with a small off-track penalty for when the car is pushed outside the track boundary by a wall hit.
3. **Prototype multi-track training** with per-rollout random track selection, using physics bounce (Mode 3) for Track2 and off-track ghost (Mode 2) for Track3.
4. **Consider fixing log_std to a narrow band** (e.g., min=max=-1.0) to remove action variance as a source of volatility.

---

## References

- Peng et al., 2017, *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* (arXiv:1703.06907)
- Tobin et al., 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (CVPR)
- Tajeri et al., 2021, *Domain Randomization for Deep Reinforcement Learning: A Survey* (arXiv:2103.02545)
- Kirk et al., 2021, *Same State, Different Task: Continual Reinforcement Learning without Interference* (arXiv:2106.09779)
- Rolnick et al., 2019, *Experience Replay for Continual Learning* (NeurIPS) [CLEAR]
- Stable-Baselines3 VecEnv docs (v2.4.1)
- Scientific Reports 2025, *New Deep RL-Driven Autonomous Model for Formula One Racecars*
