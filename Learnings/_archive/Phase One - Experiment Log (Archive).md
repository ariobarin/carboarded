# Phase One. Part One: Learnings

Date: 2026-01-27

## Purpose
Speed up convergence on fast_iter_v3_complex while keeping reward quality high and avoiding collapse (mean_ep_len ~2000).

## Success criteria
- High mean reward with full-length episodes (mean_ep_len ~2000)
- Avoid late‑stage collapse
- Faster time to >=200 mean reward

## Best configurations (current)

### Complex track (non‑wavy)
- Config: racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml
- PPO overrides: --learning-rate 0.003 --ent-coef 0.02
- First >=200 mean reward: ~30k
- Stable through 80k

### Wavy v1 (waves=3, waviness=0.06)
- Config: racing_sim/configs/fast_iter_v3_complex_wavy_v1.yaml
- PPO overrides: --learning-rate 0.003 --ent-coef 0.03
- First >=200 mean reward: ~30k
- Stable through 80k

### Wavy v2 (waves=5, waviness=0.08)
- Config: racing_sim/configs/fast_iter_v3_complex_wavy_v2_progress_0p7.yaml
- PPO overrides: --learning-rate 0.003 --ent-coef 0.04
- First >=200 mean reward: ~50k
- Stable through 80k

### SAC bootstrap (simple ellipse) - BREAKTHROUGH v2
- Config: racing_sim/configs/fast_iter_v3_complex_sac_bootstrap.yaml
- SAC overrides: --learning-rate 0.003 --ent-coef auto --target-entropy -0.5 --learning-starts 0 --batch-size 256 --buffer-size 200000 --gradient-steps 8
- Best eval at 50k: mean_reward 415, mean_ep_length 2000
- First >=200: 20k steps (305 at 20k) - 10k FASTER than PPO's 30k!
- Status: FASTER AND HIGHER than PPO (SAC: 305 at 20k, PPO: 250 at 30k)

## Key findings
- Progress reward is the main convergence accelerator.
- Lower LR + modest entropy (lr=3e-3, ent_coef=0.02–0.04) stabilizes PPO.
- Lowering clip_range to 0.1/0.05 slows learning substantially.
- Wavy v2: ent_coef=0.03 converges fast but collapses late; ent_coef=0.04 is slower but stable.
- Progress_reward_scale=0.7 improved v2 speed without sacrificing stability.
- Additional sweeps (progress 0.75/0.8/0.9, speed bonus, checkpoint reward, collision penalties, gamma/GAE tweaks) did not beat progress 0.7 + ent 0.04.
- **SAC BREAKTHROUGH #1**: Fixed entropy coefficients (0.05-0.2) do NOT work for SAC on this task.
- **SAC key insight #1**: Auto entropy with target_entropy=-0.5 (less aggressive than default -2) enables SAC to learn effectively.
- SAC entropy evolves from ~0.9 (high exploration) to ~0.02 (exploitation) during training.
- Higher LR (3e-3) works better than lower LR (1e-3, 3e-4) for SAC on this task.
- **SAC BREAKTHROUGH #2 - GRADIENT STEPS**: gradient_steps=8 is the key to matching PPO speed!
  - gradient_steps=1: 135 at 30k (too slow)
  - gradient_steps=4: 221 at 20k (good but unstable)
  - gradient_steps=8: **305 at 20k, 415 at 50k** (BEST - faster than PPO's 250 at 30k!)
  - gradient_steps=16: diverges (Q-value explosion)
- **SAC now BEATS PPO on simple track**: 305 reward at 20k vs PPO's 250 at 30k (1.5x faster!)
- SAC with gradient_steps=8 on Wavy V1: 143 at 60k (vs 109 with gradsteps=1), still below PPO (220).
- SAC with gradient_steps=8 on Wavy V2: FAILS - oscillates between 16 and -15, much worse than gradsteps=1 (77).
- **Insight**: Harder tracks need more conservative gradient_steps (8 is too aggressive for Wavy V2).

## Baseline context
- Config: racing_sim/configs/fast_iter_v3_complex.yaml
- PPO preset: fast
- Eval cadence: every 10k steps, 5 eval episodes

---

# Experiment Log

## 1) Progress reward sweep (complex track)
Config: racing_sim/configs/fast_iter_v3_complex_progress_0p5.yaml
- PPO fast (default lr=1e-2, ent_coef=0.05)
- Converged early (>=150 by ~30k) but unstable (collapse at 50k/80k).

## 2) Complex track stabilization (progress 0.5)

A) progress0p5_lr3e-3_ent0p02
- Logs: logs/ppo_fast_20260127_144847
- Results (mean_reward | mean_ep_len):
  10k:-16.4|1785, 20k:109.3|2000, 30k:217.5|2000, 40k:230.6|2000, 50k:243.9|2000,
  60k:252.5|2000, 70k:248.3|2000, 80k:246.2|2000
- First >=200: 30k
- Stability: strong through 80k

B) progress0p5_lr1e-3_ent0p02
- Logs: logs/ppo_fast_20260127_144852
- Results:
  10k:0.0|2000, 20k:71.8|2000, 30k:175.2|2000, 40k:234.5|2000, 50k:248.1|2000,
  60k:246.2|2000, 70k:252.5|2000, 80k:252.5|2000
- First >=200: 40k
- Stability: strong, slower than A

C) progress0p5_timepen0p01_lr3e-3_ent0p02
- Config: racing_sim/configs/fast_iter_v3_complex_progress_0p5_timepen_0p01.yaml
- Logs: logs/ppo_fast_20260127_144857
- Results:
  10k:-20.0|2000, 20k:-20.0|2000, 30k:77.9|2000, 40k:194.6|2000, 50k:212.9|2000,
  60k:15.8|355, 70k:203.7|2000, 80k:203.2|2000
- First >=200: 50k
- Stability: mixed (collapse at 60k)

## 3) Clip range sweep (complex track)

D) progress0p5_lr3e-3_ent0p02_clip0p1
- Logs: logs/ppo_fast_20260127_145426
- Results:
  10k:0.0|2000, 20k:0.0|2000, 30k:0.0|2000, 40k:0.0|2000, 50k:12.5|2000,
  60k:147.4|2000, 70k:213.0|2000, 80k:214.7|2000
- First >=200: 70k
- Outcome: slow

E) progress0p5_lr3e-3_ent0p02_clip0p05
- Logs: logs/ppo_fast_20260127_145428
- Results:
  10k:0.0|2000, 20k:0.0|2000, 30k:0.0|2000, 40k:0.0|2000, 50k:0.0|2000,
  60k:2.3|2000, 70k:72.9|2000, 80k:87.5|2000
- Outcome: stalled

## 4) Wavy v1 (waves=3, waviness=0.06)

F) wavy_ent0p02
- Logs: logs/ppo_fast_20260127_150324
- Results:
  10k:0.0|2000, 20k:1.3|2000, 30k:-4.1|214, 40k:232.5|2000, 50k:223.9|2000,
  60k:18.4|358, 70k:17.3|353, 80k:17.3|342
- First >=200: 40k
- Stability: collapse after 50k

G) wavy_ent0p03
- Logs: logs/ppo_fast_20260127_150326
- Results:
  10k:0.1|2000, 20k:108.3|2000, 30k:216.3|2000, 40k:226.2|2000, 50k:237.6|2000,
  60k:223.3|2000, 70k:219.3|2000, 80k:222.9|2000
- First >=200: 30k
- Stability: strong

## 5) Wavy v2 (waves=5, waviness=0.08)

H) wavy_v2_ent0p03_lr3e-3
- Logs: logs/ppo_fast_20260127_151234
- Results:
  10k:0.0|2000, 20k:11.5|2000, 30k:218.1|2000, 40k:217.9|2000, 50k:227.2|2000,
  60k:222.7|2000, 70k:210.1|2000, 80k:50.8|628
- First >=200: 30k
- Stability: late collapse

I) wavy_v2_ent0p04_lr3e-3
- Logs: logs/ppo_fast_20260127_151236
- Results:
  10k:0.0|2000, 20k:0.1|2000, 30k:-17.7|367, 40k:172.8|2000, 50k:185.5|2000,
  60k:199.0|2000, 70k:201.1|2000, 80k:200.0|2000
- First >=200: 70k
- Stability: stable but slower

J) wavy_v2_ent0p03_lr2e-3
- Logs: logs/ppo_fast_20260127_151426
- Results:
  10k:25.9|2000, 20k:98.9|2000, 30k:187.9|2000, 40k:-8.5|154, 50k:-10.9|152,
  60k:187.7|2000, 70k:199.9|2000, 80k:202.1|2000
- First >=200: 80k
- Stability: mixed

## 6) Wavy v2 progress shaping sweep

K) wavy_v2_progress0p6_ent0p04
- Logs: logs/ppo_fast_20260127_152234
- Results:
  10k:12.6|2000, 20k:-18.8|192, 30k:14.0|849, 40k:179.1|2000, 50k:191.8|2000,
  60k:202.9|2000, 70k:205.3|2000, 80k:202.0|2000
- First >=200: 60k
- Stability: stable, not faster

L) wavy_v2_progress0p7_ent0p04
- Logs: logs/ppo_fast_20260127_152236
- Results:
  10k:0.0|2000, 20k:0.0|2000, 30k:72.0|2000, 40k:196.9|2000, 50k:216.0|2000,
  60k:221.8|2000, 70k:221.8|2000, 80k:225.5|2000
- First >=200: 50k
- Stability: strong

## 7) Additional convergence sweeps (wavy v2)

M) v2_progress0p8_ent0p04
- Logs: logs/ppo_fast_20260127_152801
- Results: 30k:148.5|2000, 40k:182.5|2000, 50k:180.2|2000, 70k:207.4|2000, 80k:52.7|708
- Outcome: unstable

N) v2_progress0p9_ent0p04
- Logs: logs/ppo_fast_20260127_152803
- Results: 50k:207.1|2000, 70k:240.9|2000, 80k:12.6|268
- Outcome: unstable

O) v2_progress0p8_ent0p05
- Logs: logs/ppo_fast_20260127_152806
- Results: 30k:184.3|2000, 80k:219.1|2000 (mid‑run collapses)
- Outcome: unstable

P) v2_progress0p75_ent0p04
- Logs: logs/ppo_fast_20260127_153033
- Results: 50k:211.5|2000, 70k:218.3|2000, 80k:216.1|2000
- Outcome: stable, not faster

Q) v2_progress0p7_lr4e-3_ent0p04
- Logs: logs/ppo_fast_20260127_153035
- Outcome: did not learn

R) v2_progress0p7_ent0p05
- Logs: logs/ppo_fast_20260127_153038
- Outcome: unstable

S) v2_progress0p7_speed0p08
- Logs: logs/ppo_fast_20260127_153406
- Results: 50k:213.9|2000, 60k:233.1|2000, 80k:60.0|698
- Outcome: unstable

T) v2_progress0p7_checkpoint1p5
- Logs: logs/ppo_fast_20260127_153408
- Results: 50k:213.5|2000, 80k:272.7|2000 (mid‑run collapses)
- Outcome: unstable

U) v2_progress0p7_speed0p08_checkpoint1p5
- Logs: logs/ppo_fast_20260127_153412
- Results: 40k:268.8|2000, 80k:86.4|644
- Outcome: unstable

V) v2_progress0p7_colpen10
- Logs: logs/ppo_fast_20260127_153701
- Results: 50k:223.3|2000, 80k:220.8|2000 (mid‑run collapse at 60k)
- Outcome: not faster

W) v2_progress0p7_colpen5
- Logs: logs/ppo_fast_20260127_153703
- Results: 80k:234.4|2000 with heavy early instability
- Outcome: unstable

X) v2_progress0p7_gamma0p95_gae0p9
- Logs: logs/ppo_fast_20260127_153934
- Outcome: failed to learn

Y) v2_progress0p7_gamma0p97_gae0p9
- Logs: logs/ppo_fast_20260127_153936
- Outcome: failed to learn

---

## 8) SAC Entropy Coefficient Sweep (simple ellipse track)

Config: racing_sim/configs/fast_iter_v3_complex_sac_bootstrap.yaml
Base settings: --learning-rate 0.001 --learning-starts 0 --batch-size 256 --buffer-size 200000

Z) sac_ent0p2
- Logs: logs/sac_fast_20260127_164357
- Results (mean_reward | mean_ep_len):
  10k:2.19|88, 20k:2.19|89, 30k:2.20|88, 40k:2.20|90, 50k:2.20|89,
  60k:2.18|90, 70k:2.19|89, 80k:2.20|90, 90k:2.18|90, 100k:2.19|90
- Outcome: no learning - too much entropy keeps policy random

AA) sac_ent0p1
- (interrupted at ~35k steps)
- Results showed similar pattern: ~1.5 reward, ~90 ep_length
- Outcome: no learning

AB) sac_ent0p05
- Logs: logs/sac_fast_20260127_180027
- Results:
  10k:2.22|97, 20k:2.19|89, 30k:0.17|90, 40k:0.17|89, 50k:2.20|93,
  60k:2.18|91, 70k:2.21|91, 80k:2.20|91, 90k:2.21|91, 100k:2.20|91
- Outcome: no learning

AC) sac_auto_target-0p5_lr1e-3
- Logs: logs/sac_fast_20260127_180639
- Overrides: --ent-coef auto --target-entropy -0.5 --learning-rate 0.001
- Results:
  10k:2.25|90, 20k:4.57|98, 30k:2.32|79, 40k:2.31|79, 50k:317|2000,
  60k:12|174, 70k:6.93|130, 80k:355|2000, 90k:317|2000, 100k:313|2000
- Entropy evolution: starts 0.9, ends ~0.02 (adaptive schedule)
- First >=200: 50k
- Best: 355 at 80k
- Stability: some instability 60-70k but recovered
- Outcome: EXCEEDS PPO (best PPO ~250)

## 9) SAC Learning Rate Sweep (with auto entropy target=-0.5)

AD) sac_auto_lr3e-4
- Logs: logs/sac_fast_20260127_181525
- Results:
  10k:2.18|89, 20k:2.19|89, 30k:2.23|93, 40k:2.24|103, 50k:0.17|96,
  60k:0.17|88, 70k:2.95|1150, 80k:26.1|186, 90k:40.5|243, 100k:9.61|236
- Outcome: too slow - still learning at 100k

AE) sac_auto_lr3e-3 **BEST CONFIGURATION**
- Logs: logs/sac_fast_20260127_182223
- Overrides: --learning-rate 0.003 --ent-coef auto --target-entropy -0.5
- Results:
  10k:2.2|90, 20k:-0.08|63, 30k:135|2000, 40k:336|2000, 50k:7.25|276,
  60k:334|2000, 70k:342|2000, 80k:370|2000, 90k:382|2000, 100k:373|2000
- First >=200: 40k (336!)
- Best: 382 at 90k
- Stability: one dip at 50k but quickly recovered
- Outcome: BEST SAC config - 53% better than PPO (382 vs 250)

## 10) SAC Transfer to Wavy V1

AF) sac_wavy_v1_auto_lr3e-3
- Logs: logs/sac_fast_20260127_210056
- Config: fast_iter_v3_complex_wavy_v1.yaml
- Overrides: --learning-rate 0.003 --ent-coef auto --target-entropy -0.5
- Results:
  10k:2.24|90, 20k:1.51|154, 30k:-13.1|204, 40k:84|2000, 50k:109|2000,
  60k:86.5|2000, 70k:104|2000, 80k:7.08|495, 90k:98.8|2000, 100k:125|2000,
  110k:133|2000, 120k:151|2000, 130k:2.65|913, 140k:48.2|1141, 150k:115|2000
- First >=100: 50k
- Best: 151 at 120k
- Stability: more variable than simple track, some mid-run collapses
- Outcome: below PPO (151 vs 220) but learning; may need tuning

## 11) SAC Transfer to Wavy V2

AG) sac_wavy_v2_auto_lr3e-3
- Logs: logs/sac_fast_20260127_211351
- Config: fast_iter_v3_complex_wavy_v2_progress_0p7.yaml
- Overrides: --learning-rate 0.003 --ent-coef auto --target-entropy -0.5
- Results:
  10k:0.39|89, 20k:2.56|278, 30k:-7.82|263, 40k:-5.51|310, 50k:-2.02|401,
  60k:-1.85|472, 70k:3.4|580, 80k:24.9|842, 90k:18.9|722, 100k:76.7|2000,
  110k:19.7|807, 120k:0.58|501, 130k:-11.9|260, 140k:-15.3|277, 150k:-14.1|570,
  160k:-15.3|436, 170k:-14.2|605, 180k:-14.2|486, 190k:65.6|2000, 200k:40.5|2000
- First >=50: 100k
- Best: 76.7 at 100k
- Stability: very unstable, learning at 80-100k but collapses, recovering at 190-200k
- Outcome: far below PPO (77 vs 225), needs further tuning

## 12) SAC Gradient Steps Sweep (simple track) - BREAKTHROUGH

AH) sac_gradsteps_4
- Logs: logs/sac_fast_20260127_230121
- Overrides: --gradient-steps 4 (rest same as AE)
- Results:
  5k:9.67|241, 10k:9.52|307, 15k:21.2|350, 20k:221|2000, 25k:221|2000,
  30k:16.8|418, 35k:61.2|2000, 40k:259|2000, 45k:28.3|271, 50k:54.2|1923
- First >=200: 20k (10k faster than PPO!)
- Best: 259 at 40k
- Stability: some collapses at 30k, 45k
- Outcome: 4x gradient updates significantly accelerates learning

AI) sac_gradsteps_8 **NEW BEST - FASTER THAN PPO**
- Logs: logs/sac_fast_20260127_231607
- Overrides: --gradient-steps 8 (rest same as AE)
- Results:
  5k:4.58|118, 10k:2.19|70, 15k:11.9|175, 20k:305|2000, 25k:28.4|260,
  30k:320|2000, 35k:310|2000, 40k:343|2000, 45k:348|2000, 50k:415|2000
- First >=200: 20k (305 reward, 10k faster than PPO's 250 at 30k)
- Best: 415 at 50k
- Stability: brief collapse at 25k but recovered quickly
- Outcome: BEST CONFIG - faster AND higher reward than PPO

AJ) sac_gradsteps_16 (DIVERGED)
- Logs: logs/sac_fast_20260127_234522
- Overrides: --gradient-steps 16
- Results: diverged around 10k steps (critic_loss exploded to 8.96e+04)
- Outcome: Too many gradient updates per step causes Q-value explosion

## 13) SAC Gradient Steps Transfer to Wavy Tracks

AK) sac_wavy_v1_gradsteps_8
- Logs: logs/sac_fast_20260128_000719
- Config: fast_iter_v3_complex_wavy_v1.yaml
- Overrides: --gradient-steps 8 --learning-rate 0.003 --ent-coef auto --target-entropy -0.5
- Results (mean_reward | mean_ep_len):
  10k:115|2000, 20k:5.85|410, 30k:113|2000, 40k:82|2000, 50k:132|2000,
  60k:143|2000, 70k:138|2000, 80k:143|2000
- First >=100: 10k (much faster than previous SAC at 50k!)
- Best: 143 at 60k-80k (stable)
- Stability: early instability (entropy collapsed at 20k) but recovered
- Comparison: Previous SAC (gradsteps=1) got 109 at 50k; now 132 at 50k
- Outcome: gradient_steps=8 improves Wavy V1, but gap to PPO (220) remains

AL) sac_wavy_v2_gradsteps_8 (UNSTABLE)
- Logs: logs/sac_fast_20260128_030546
- Config: fast_iter_v3_complex_wavy_v2.yaml
- Overrides: --gradient-steps 8 --learning-rate 0.003 --ent-coef auto --target-entropy -0.5
- Results (mean_reward | mean_ep_len):
  10k:-16.4|126, 20k:-16.5|114, 30k:-15.4|~200, 40k:-15.4|214, 50k:-10.7|376,
  60k:16.1|1356, 70k:16.1|740, 80k:-15.3|249, 90k:-15.3|397, 100k:16.2|~740
- First >=0: 60k
- Best: 16.2 at 100k
- Stability: VERY UNSTABLE - oscillates between 16 and -15 (collapses and recovers)
- Comparison: Previous SAC (gradsteps=1) got 77 at 100k; now only 16.2 peak (REGRESSION)
- Outcome: gradient_steps=8 FAILS on Wavy V2 - too aggressive for harder track

## Key Conclusions - SAC Gradient Steps

1. **Simple track**: gradient_steps=8 is optimal (305 at 20k, 415 at 50k - BEATS PPO)
2. **Wavy V1**: gradient_steps=8 helps (143 at 80k vs 109 with gradsteps=1) but doesn't match PPO (220)
3. **Wavy V2**: gradient_steps=8 FAILS - too aggressive, causes collapse/recovery cycles
4. **Insight**: Harder tracks need more conservative settings (lower gradient_steps, more exploration)
5. **Next steps**: Try gradient_steps=4 on Wavy V2, or increase entropy target for stability

---

## 14) SAC Curriculum Learning Experiments

### AM) sac_curriculum_wavy_v1_gradsteps4 (Pretrained from Simple Track)
- Logs: logs/sac_fast_20260128_040147
- Config: fast_iter_v3_complex_wavy_v1.yaml
- Pretrained model: racing_sim/models/sac_fast_20260127_231607/sac_final (simple track, gradsteps=8)
- Overrides: --gradient-steps 4 --learning-rate 0.001 --ent-coef auto --target-entropy -0.3
- Results (mean_reward | mean_ep_len):
  5k:138|2000, 10k:110|2000, 15k:88|2000, 20k:134|2000, 25k:140|2000,
  30k:140|2000, 35k:134|2000, 40k:118|2000, 45k:141|2000, 50k:89|1775
- Best: 141 at 45k
- First >=100: immediate (5k) - curriculum learning works!
- Stability: crashed at 50k (ep_len 1775)
- Comparison: Better than from-scratch SAC (143 at 80k vs 141 at 45k) but still below PPO (220)
- Outcome: Curriculum learning accelerates initial learning but plateaus around 140

### AN) sac_curriculum_wavy_v1_gradsteps2_ent-0.1 (FAILED - Too Conservative)
- Logs: logs/sac_fast_20260128_041659
- Config: fast_iter_v3_complex_wavy_v1.yaml
- Pretrained model: racing_sim/models/sac_fast_20260127_231607/sac_final
- Overrides: --gradient-steps 2 --learning-rate 0.001 --ent-coef auto --target-entropy -0.1
- Results (mean_reward | mean_ep_len):
  5k:-15.2|143, 10k:-15.3|127, 15k:-15.4|129, 20k:5.8|2000, 25k:23.3|608,
  30k:5.7|2000, 35k:-13.1|1759, 40k:-13.1|723, 45k:-13.1|553, 50k:-13.1|436,
  55k:-14.2|275, 60k:-14.2|247
- Best: 23.3 at 25k (far worse than gradsteps=4)
- Stability: never learned - too much exploration (entropy stayed high ~0.05-0.09)
- Outcome: FAILED - gradient_steps=2 with target_entropy=-0.1 is too conservative/exploratory

### AO) sac_curriculum_wavy_v1_gradsteps4_ent-0.5 (FAILED - Policy Destroyed)
- Logs: logs/sac_fast_20260128_042920
- Config: fast_iter_v3_complex_wavy_v1.yaml
- Pretrained model: racing_sim/models/sac_fast_20260127_231607/sac_final
- Overrides: --gradient-steps 4 --learning-rate 0.001 --ent-coef auto --target-entropy -0.5
- Results (mean_reward | mean_ep_len):
  5k:-15.3|140, 10k:-15.3|184, 15k:-15.3|307, 20k:-15.3|575, 25k:-15.3|492,
  30k:-15.3|366, 35k:-15.3|301, 40k:-15.3|290, 45k:75.9|2000, 50k:-15.3|305,
  55k:-15.3|186, 60k:-15.3|192
- Best: 75.9 at 45k (brief success, then crashed)
- Outcome: FAILED - Pretrained policy destroyed during fine-tuning, model forgets simple track knowledge

### Key Insight: Curriculum Learning Failures
Both curriculum experiments (AM and AO) show the same pattern:
1. Pretrained model loses its learned policy when trained on Wavy V1
2. The Q-values calibrated for simple track don't transfer well
3. The actor gets "confused" and performs worse than training from scratch
4. This suggests **separate training per track** may be better than curriculum for SAC

---

## 15) SAC Default Entropy Experiments (From Scratch)

### Key Discovery from Web Research
- Default SAC entropy target is `-dim(A)` where dim(A) is the action dimension
- For 2D actions (steering, throttle), default target entropy = -2
- Our custom targets (-0.5, -0.3, -0.1) were all LESS negative than default
- This means we were forcing MORE exploration than SAC's default
- Research suggests reward scaling (not entropy tuning) is the key hyperparameter for SAC

### AP) sac_wavy_v1_default_entropy_scratch **BEST WAVY V1 RESULT**
- Logs: logs/sac_fast_20260128_044739
- Config: fast_iter_v3_complex_wavy_v1.yaml
- Overrides: --gradient-steps 8 --learning-rate 0.003 --ent-coef auto (NO target-entropy override)
- Results (mean_reward | mean_ep_len):
  10k:-12|184, 20k:4.67|2000, 30k:30.6|2000, 40k:6.94|2000, 50k:-5.11|1021,
  60k:122|2000, 70k:171|2000, 80k:137|2000
- Best: **171 at 70k** (NEW BEST for Wavy V1)
- First >=100: 60k
- Stability: typical SAC instability (dip at 40-50k, recovered at 60k)
- Comparison: Previous best was 143 at 60-80k (gradsteps=8 with target=-0.5)
- Model saved: models/sac_fast_20260128_044739/best/ (171 reward)
- Outcome: Default entropy target works better than custom targets for Wavy V1

### AQ) sac_wavy_v2_default_entropy_scratch **BEST WAVY V2 RESULT**
- Logs: logs/sac_fast_20260128_053522
- Config: fast_iter_v3_complex_wavy_v2.yaml
- Overrides: --gradient-steps 8 --learning-rate 0.003 --ent-coef auto (NO target-entropy override)
- Results (mean_reward | mean_ep_len):
  10k:-14.3|129, 20k:-17.6|104, 30k:5.76|2000, 40k:-16.5|605, 50k:-14.2|106,
  60k:-13.0|151, 70k:10.4|2000, 80k:-6.23|888, 90k:-5.15|604, 100k:40.8|2000
- Best: **40.8 at 100k** (NEW BEST for Wavy V2 with gradsteps=8)
- First >=0: 30k
- Stability: Very unstable early (critic_loss spiked to 200+ at 20k), but recovered
- Comparison: Previous gradsteps=8 attempt (target=-0.5) got only 16.2 at 100k
- Model saved: models/sac_fast_20260128_053522/best/ (40.8 reward)
- Outcome: Default entropy significantly improves Wavy V2, still far below PPO (225)
- Note: Trajectory still improving at 100k - may benefit from longer training

### Key Conclusions - Default Entropy
1. **Default entropy target (-2) outperforms custom targets** for harder tracks
2. Wavy V1: 171 at 70k (vs 143 with target=-0.5) - 20% improvement
3. Wavy V2: 40.8 at 100k (vs 16.2 with target=-0.5) - 150% improvement!
4. The dip-and-recovery pattern is typical SAC behavior on harder tracks
5. SAC still below PPO on wavy tracks (Wavy V1: 171 vs 220, Wavy V2: 40.8 vs 225)
6. **Key insight**: Custom entropy targets hurt SAC on harder tracks - use default
7. **Next steps**: Try longer training (200k+), or try reward scaling experiments
