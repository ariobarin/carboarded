# Resume Metrics (Playable Steering)

**Evaluation protocol:** 50 deterministic episodes per agent.

## PPO Highlights
- **fast_iter_v3:** mean reward **79.49**, **32 checkpoints**, **0% collisions**, **400 steps**.
- **default:** mean reward **0.86**, **88 checkpoints**, **0% collisions**, **1000 steps**.

## Baseline Context
- SAC and rule-based baselines currently collide on every run under short training budgets.
- PPO is the only agent consistently completing full episodes.

## Notes for Attribution
- Reward scales differ between `fast_iter_v3` and `default` configs.
- Report both reward and checkpoint counts for clarity.
