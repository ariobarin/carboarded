# What Didn't Work -- Phase 1 Anti-Patterns

Reference guide of failed approaches from Phase 1 experiments. Avoid repeating these.

---

## Reward Shaping Failures

**Progress reward >= 0.8 causes instability.**
Tested progress_reward_scale at 0.8 and 0.9. Both caused training instability with wild reward swings and policy collapse. The sweet spot is 0.5-0.75, with 0.75 being the proven maximum.

**Increasing time penalty hurts convergence.**
Time penalty of -0.1 per step slows learning significantly. Setting it to 0.0 is strictly better. The agent learns to move fast on its own from progress and speed bonuses.

**Changing collision penalty has no effect.**
Tested -5, -10, -20, -25. No measurable difference in convergence speed or final reward. The default -20 is fine; don't waste experiments on this.

**Speed bonus and checkpoint reward boosts don't help.**
Increasing these values beyond defaults produced no improvement. Progress reward shaping dominates the signal.

**Slowdown penalty hurts convergence.**
The original default.yaml had slowdown_distance=0.4 and slowdown_penalty_scale=2.0. Removing both (setting to 0.0) improved convergence. The penalty confuses the agent near walls without teaching it to avoid them.

---

## PPO Failures

**Lowering clip_range (0.1, 0.05) slows learning.**
Tested both values against default 0.2. Both resulted in slower convergence with no stability benefit.

**Gamma and GAE lambda tweaks break learning.**
Tested gamma values other than 0.99 and GAE lambda values other than 0.95. All performed worse than defaults.

**PPO with random starting positions fails on hard tracks.**
PPO achieved only 17.7 reward with --random-start on Wavy V2 (vs 225 without). PPO's on-policy nature cannot handle the variance from random positions.

---

## SAC Failures

**Fixed entropy coefficients prevent learning.**
Tested --ent-coef 0.05, 0.1, 0.2 on SAC. None learned. SAC requires --ent-coef auto with default target entropy. Do not override.

**Custom target entropy (-0.5) is worse than default on hard tracks.**
Partial success on simple tracks but degraded performance on Wavy V1/V2.

**Curriculum learning (fine-tuning on harder tracks) destroys SAC policy.**
Loading a simple-track SAC model and continuing training on Wavy V1/V2 causes the policy to collapse. The replay buffer and critic are mismatched with the new environment dynamics.

**gradient_steps=8 fails on wavy tracks.**
Works well on simple tracks but causes over-fitting and instability on Wavy V1/V2. Use gradient_steps=4 for any wavy track.

**gradient_steps=8 combined with batch_size=512 on Wavy V2 is the worst combination.**
Most unstable configuration tested. Wild reward swings (-8 to 149), never converges.

**tau=0.002 (slower target network updates) prevents learning entirely.**
SAC with tau=0.002 peaked at 14.6 reward. The target network updates too slowly to track the changing policy.

**Combining multiple hyperparameter changes catastrophically interferes.**
Phase 2 combined experiment (lr=0.001, batch_size=512, buffer_size=500000, gradient_steps=2) achieved only 16.66 peak vs 154 baseline. Settings that work individually can destroy performance when combined.

---

## General Anti-Patterns

**Changing multiple parameters at once.**
Every Phase 1 experiment that changed 2+ hyperparameters simultaneously failed. The one-at-a-time approach identified batch_size=512 (peak 209) and lr=0.001 (stable 158) as promising, but combining them with other changes produced 16.66.

**Assuming PPO and SAC respond the same way to changes.**
Random starts help SAC (183.7) but destroy PPO (17.7). gradient_steps tuning is critical for SAC but irrelevant for PPO. Entropy must be auto for SAC but fixed for PPO. Always test algorithm-specific.

**Short training runs for conclusions.**
Several experiments were killed at 5-8k steps before reaching meaningful eval checkpoints. SAC on Wavy V2 needs at least 40-60k steps before showing its true potential due to high initial instability.

**Overwriting existing models during fine-tuning.**
Always save to a new directory. Fine-tuning can collapse, and you need the original model to fall back to.
