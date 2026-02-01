# Phase 1 Retrospective: Mistakes, Causes, and Guardrails

## Scope
This note captures why early runs (ellipse + wavy track) underperformed and what we changed to prevent repeat mistakes.

## What Went Wrong
- **Sensor under-coverage**: 5 short LIDAR rays (200px) couldn’t “see” enough of the track, so agents learned to creep into walls.
- **Config drift**: Edits were made in YAMLs that weren’t actually loaded for training or play, masking changes.
- **Reward/scale mismatches**: Comparisons mixed runs with different reward scales or episode lengths, making plots misleading.
- **SAC exploration collapse**: `auto` entropy decayed to near-zero, causing straight-line policies that never turned.
- **Windows subproc slowdown**: Subproc vectorization tanked FPS and made early tuning painfully slow.
- **No action diagnostics**: We didn’t inspect action distributions early, so “no steering” wasn’t caught fast.

## Root Causes
- We optimized for speed before confirming the **sim felt correct by hand**.
- We lacked a single “source-of-truth” config for play/train/eval.
- We compared runs before normalizing **episode length, reward scale, and track settings**.

## Fixes Applied
- Increased LIDAR coverage (9 rays, 400px) across configs.
- Standardized on the active config set for play/train/eval.
- Added longer episode configs for complex tracks and explicit lap checks.
- Forced SAC exploration with fixed entropy when auto collapsed.
- Switched to `dummy` vec env on Windows for predictable FPS.

## Guardrails (Do This Next Time)
- **Always manual-play the current config** before training.
- **Log and inspect action stats** (mean/variance of steering + throttle) early.
- **Compare apples-to-apples**: same config, reward scale, episode length.
- **One active config path** for a run; print it in logs and reports.
- **Keep a baseline PPO run** for each track revision to anchor progress.
