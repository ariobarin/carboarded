# Live Experiment Tracking - January 30, 2026
## Phase 1: Break 200 & Push to 250+

### ACTIVE EXPERIMENTS

#### 🔥 Experiment 1.1: SAC Progress 0.75 (60k steps)
- **Status:** RUNNING (background)
- **Config:** `opt_progress_0p75.yaml`
- **Base Model:** 183.7 (90k steps)
- **Goal:** Break 200
- **Started:** 2026-01-30 ~16:45
- **ETA:** ~30-40 min
- **Log:** `logs/exp1p1_sac_progress_0p75.log`
- **Checkpoints:** 10k, 20k, 30k, 40k, 50k, 60k

#### 🔥 Experiment 1.5: PPO Extended 120k
- **Status:** RUNNING (background)
- **Config:** `fast_iter_v3_complex_wavy_v2_progress_0p7.yaml`
- **Base Model:** None (from scratch)
- **Goal:** 240+ (beat current 225)
- **Started:** 2026-01-30 ~16:45
- **ETA:** ~35-45 min
- **Log:** `logs/exp1p5_ppo_extended_120k.log`
- **Checkpoints:** 10k, 20k, 30k, 40k, 50k, 60k, 70k, 80k, 90k, 100k, 110k, 120k

---

### QUEUED EXPERIMENTS

#### ⏳ Experiment 1.2: SAC Progress 0.8 + Batch 512
- **Trigger:** After 1.1 completes, if reward >190
- **Config:** `opt_progress_0p8.yaml` + batch_size=512
- **Base:** Best model from 1.1
- **Goal:** 210+

#### ⏳ Experiment 1.6: PPO Progress 0.8
- **Trigger:** After 1.5 completes
- **Config:** progress=0.8
- **Base:** None (from scratch)
- **Goal:** 250+

#### ⏳ Experiment 1.4 & 1.7: Overnight Long Training
- **SAC:** 150k steps with best config
- **PPO:** 150k steps with random starts
- **Run:** Tonight after analysis

---

### RESULTS TRACKING

| Experiment | 10k | 20k | 30k | 40k | 50k | 60k | Best | Status |
|------------|-----|-----|-----|-----|-----|-----|------|--------|
| 1.1 SAC (0.75) | - | - | - | - | - | - | - | 🟡 Running |
| 1.5 PPO (120k) | - | - | - | - | - | - | - | 🟡 Running |

**Legend:**
- 🟡 Running
- ✅ Complete
- ❌ Failed
- ⏳ Queued

---

### SUCCESS CRITERIA

#### Phase 1 Complete When:
- [ ] SAC achieves >200 (single run)
- [ ] SAC achieves stable >210 (3 runs)
- [ ] PPO achieves >240
- [ ] Both pass 100-episode validation

#### Super-Good Model Criteria:
- [ ] SAC: >225 (match PPO)
- [ ] PPO: >250
- [ ] Both: Stable across multiple seeds
- [ ] Both: <50k steps to >200

---

### CHECKPOINT DECISIONS

**For SAC (from 183.7 base):**
- **<180 at 20k:** ❌ Kill - not improving
- **>190 at 30k:** ✅ Continue - on track for 200
- **>200 at 40k:** 🚀 Excellent - push for 225
- **Crash/instability:** 💾 Save best, restart with tweaks

**For PPO (from scratch):**
- **<150 at 30k:** ❌ Kill - check config
- **>200 at 60k:** ✅ Good - continue to 120k
- **>230 at 100k:** 🚀 Great - push for 250
- **Crash:** 💾 Save, may need lower progress reward

---

### NEXT ACTIONS

**In 30 minutes (17:15):**
1. Check experiment logs for progress
2. See if any reached 10k checkpoint
3. Decide on early kills or continues

**In 60 minutes (17:45):**
1. Analyze 1.1 and 1.5 results
2. Queue next experiments based on winners
3. Prepare overnight jobs

**Tonight:**
1. Run long-training experiments
2. Set up monitoring

**Tomorrow morning:**
1. Collect all results
2. Save super-good models
3. Begin CNN prep phase

---

### COMMANDS

**Check running experiments:**
```bash
cd racing_sim
tail -50 logs/exp1p1_sac_progress_0p75.log
tail -50 logs/exp1p5_ppo_extended_120k.log
ls logs/ | grep sac_fast_20260130
ls logs/ | grep ppo_fast_20260130
```

**Check eval results:**
```bash
python -c "
import numpy as np
data = np.load('logs/EXPERIMENT_NAME/evaluations.npz', allow_pickle=True)
print('Timesteps:', data['timesteps'])
print('Rewards:', data['results'].mean(axis=1))
"
```

**Kill stuck experiments:**
```bash
taskkill /F /IM python.exe  # Windows
echo "Experiments killed - restart manually"
```

---

**Last Updated:** 2026-01-30 16:45
**Next Check:** 2026-01-30 17:15 (30 min)
