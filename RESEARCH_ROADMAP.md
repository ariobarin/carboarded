# Research Roadmap: From Lidar to CNN

## Current Status
- **Phase 1: COMPLETE.** PPO achieved 247.26 on Wavy V2 (all-time best). SAC achieved 183.7.
- **Phase 1.5: IN PROGRESS.** SAC optimization push -- closing the PPO-SAC gap before CNN transition.
- **Phase 2: NOT STARTED.** Robustness and hardening.
- **Phase 3: NOT STARTED.** CNN architecture.
- **Phase 4: NOT STARTED.** Sim-to-real bridge.

---

## Phase 1: Break 200 (COMPLETE)

**Result:** PPO 247.26 at 220k steps, validated at 100% success rate across 100 episodes.
**Best model:** `Good Models/PPO Wavy V2 Ent0p02 LR0p001 500k - 247.26 Reward/`
**Key finding:** LR=0.001 + progress_reward_scale=0.75 + ent_coef=0.02 was the optimal combination.

Earlier models (226.49, 243.71, 342.80 on 3k steps) are archived in `Good Models/_archived/`.

See `PHASE_1_COMPLETE.md` for full results and `Learnings/Phase One Summary.md` for detailed experiment history.

---

## Phase 1.5: SAC Optimization Push (IN PROGRESS)

**Goal:** Push SAC on Wavy V2 as close to PPO (247.26) as possible. Current SAC best: 183.7.
**Approach:** Systematic single-parameter experiments from a baseline SAC config.
**Decision criteria:** SAC > 220 = success, SAC > 200 = partial success, no experiment > 190 = accept and move on.

See `Learnings/SAC Optimization Results.md` for experiment details (created during Phase B).

---

## Phase 2: Robustness and Hardening

### Validation Suite
For each production model:
1. 100-episode deterministic test -- verify consistency
2. 100-episode stochastic test -- measure variance
3. All-starting-positions test -- verify robustness
4. Failure analysis -- document crash scenarios

### Hyperparameter Sensitivity
Test stability across:
- Learning rate +/-30%
- Entropy coefficient +/-0.01
- Batch size +/-128
- Different random seeds (3 runs)

---

## Phase 3: CNN Architecture Preparation

### 3.1 Track Complexity Increase
Use `configs/deprecated/cnn_prep_complex_track.yaml` as a starting point:
- waviness: 0.12 (up from 0.08)
- waves: 7 (up from 5)
- width: 80 (down from 100, tighter)

### 3.2 Observation Space Evolution
**Stage A:** Enhanced Lidar (before CNN)
- Add velocity to observation
- Add previous action history
- Add time features

**Stage B:** Hybrid CNN + Lidar
- Small CNN on downsampled camera view (32x32)
- Keep best lidar features
- Combined MLP head

**Stage C:** Full CNN
- 64x64 or 128x128 camera input
- 3-4 conv layers
- Transfer learning from best lidar policy

### 3.3 Fast Convergence Techniques
1. Transfer learning: initialize CNN from best lidar policy weights
2. Curriculum: start on simple track, progressively increase difficulty
3. Demonstration learning: use best lidar model to generate expert trajectories
4. Auxiliary tasks: predict velocity, steering angle as side tasks

---

## Phase 4: Sim-to-Real Bridge

### 4.1 Domain Randomization Training
```yaml
lidar_noise: 0.05-0.15
friction_range: [0.3, 0.7]
action_delay: [0, 3]  # frames
motor_noise: 0.05-0.10
```

### 4.2 Efficiency Optimization
Target: >250 in <30k steps
- Prioritized experience replay (PER)
- Hindsight experience replay (HER) for failed episodes
- Multi-step returns (n-step TD)
- TD3 if SAC struggles (requires adding to train.py)

### 4.3 Deployment Architecture
- Model quantization (INT8) for edge inference
- ONNX export for cross-platform compatibility
- Real-time performance validation

---

## Success Criteria

### Phase 1 (COMPLETE):
- [x] PPO > 200 reward (226.49)
- [x] Stable across 100 episodes
- [ ] SAC > 200 reward (best: 183.7)

### Phase 2 Ready When:
- [ ] Sensitivity analysis complete for best models
- [ ] Failure modes documented
- [ ] Multi-seed validation passed

### Phase 3 Ready When:
- [ ] Best lidar policy > 250
- [ ] CNN architecture designed and tested
- [ ] Transfer learning pipeline working
- [ ] Domain randomization config ready

---

## Risk Mitigation

### If CNN Transition Struggles:
- Start with hybrid (CNN + lidar)
- Use extensive transfer learning
- Increase training budget 2-3x for CNN
- Consider smaller CNN (ResNet-18 style)

### If SAC Never Reaches 200:
- Accept PPO as the production algorithm for hard tracks
- Use SAC only for simple tracks where it converges faster
- Focus SAC effort on robustness rather than peak reward
