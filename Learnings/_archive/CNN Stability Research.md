# CNN Stability and Convergence Research

**Goal:** Improve CNN training stability and convergence speed to prepare for more complex tracks.
**Start Date:** 2026-02-01
**Budget:** ~2 hours
**Baseline:** CNN 249.43 at 220k steps (LR=0.0003, ent=0.02)

---

## Current Problems

1. **Slow initial learning:** First positive reward at ~40k steps (lidar: ~10k)
2. **PPO oscillation:** Peak at 200-280k, then collapses to ~26 reward
3. **High variance:** Seed choice accounts for 12-14 points
4. **Long training:** 300k steps = ~1 hour for peak performance

---

## Research Questions

1. Can learning rate scheduling reduce late-training collapse?
2. Can entropy annealing speed up early learning while maintaining late stability?
3. Can more parallel environments reduce variance and speed convergence?
4. Can curriculum learning (simpler track first) accelerate CNN training?

---

## Experiment Plan

### Phase 1: LR + Entropy Scheduling (~45 min)

**Hypothesis:** Decaying LR and entropy should:
- Early: High LR + entropy = fast exploration
- Late: Low LR + entropy = stable exploitation

**Experiment 1A: Linear LR Decay**
- LR: 0.0003 -> 0.0001 (linear decay over training)
- Entropy: fixed 0.02
- Steps: 300k
- Command: Need to add LR schedule support to train.py

**Experiment 1B: Higher Initial Entropy**
- LR: fixed 0.0003
- Entropy: 0.04 (vs baseline 0.02)
- Steps: 300k
- Reasoning: Faster early exploration, may reach positive reward sooner

### Phase 2: Parallelism (~30 min)

**Hypothesis:** More environments = more diverse experience per update = faster convergence.

**Experiment 2A: Double Environments**
- n_envs: 8 (vs baseline 4)
- All other settings same as best CNN config
- Steps: 300k

### Phase 3: Curriculum Learning (~45 min)

**Hypothesis:** Training on easier track first, then fine-tuning on harder track.

**Experiment 3A: Wavy V1 -> Wavy V2 Transfer**
1. Train CNN on Wavy V1 (easier: waviness=0.06, waves=3) for 150k steps
2. Fine-tune on Wavy V2 (harder: waviness=0.08, waves=5) for 150k steps
- Total: 300k steps same as baseline

**Risk:** Phase 1 Summary notes curriculum learning "destroys SAC policy" - need to verify for PPO.

---

## Decision Points

After Experiment 1B:
- If entropy 0.04 reaches positive reward faster: use for future training
- If entropy 0.04 is worse: stick with 0.02

After Experiment 2A:
- If n_envs=8 is faster/more stable: use for all future CNN training
- If same or worse: stick with n_envs=4

After Experiment 3A:
- If curriculum works: use for harder tracks
- If curriculum destroys policy (like SAC): avoid for PPO too

---

## Complex Track Preparation

For future harder tracks (waviness > 0.08, waves > 5, width < 100):

**Track Difficulty Levers:**
- waviness: 0.08 -> 0.12 (tighter curves)
- waves: 5 -> 7 (more turns per lap)
- width: 100 -> 80 (narrower track)

**Expected Challenges:**
- Need more exploration (higher entropy initially)
- May need longer training
- May need higher grid resolution for tight turns

**Proposed Wavy V3 Config (future):**
```yaml
track:
  waviness: 0.10
  waves: 6
  width: 90
```

---

## Metrics to Track

For each experiment:
1. Steps to first positive reward (target: <20k, baseline: ~40k)
2. Peak eval reward (target: >=249, baseline: 249.43)
3. Step at peak (earlier = better, baseline: 220k)
4. Final eval (stability measure, baseline: 26.2 - very collapsed)
5. Policy std over training (stability indicator)

---

## Quick Wins to Try First

Before running full experiments, quick tests:

1. **Higher entropy quick test** (1 run, check first 50k)
   - Does ent=0.04 reach positive reward faster than ent=0.02?

2. **n_envs=8 quick test** (1 run, check first 50k)
   - Does more parallelism speed up initial learning?

---

## Experiment Log

### Experiment 1B: Higher Entropy (ent=0.04)

**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.04 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

- Start time:
- Model dir:
- Results:

### Experiment 2A: More Parallel Envs (n_envs=8)

**Command:**
```bash
cd racing_sim
py scripts/train.py --algo ppo --preset fast --total-timesteps 300000 \
  --cnn --config configs/wavy_v2_cnn.yaml \
  --learning-rate 0.0003 --ent-coef 0.02 --n-envs 8 \
  --save-freq 50000 --eval-freq 20000 --eval-episodes 5 --seed 42
```

- Start time:
- Model dir:
- Results:

---

## Notes

- LR scheduling requires code changes to train.py (SB3 supports callable learning_rate)
- Entropy scheduling also requires code changes (not built-in)
- Focus on experiments that don't require code changes first
- If quick tests show promise, implement scheduling for next phase
