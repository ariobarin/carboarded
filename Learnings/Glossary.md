# Glossary

RL and ML terms used throughout this project's documentation.

---

## Algorithms

**PPO (Proximal Policy Optimization)** -- On-policy RL algorithm that updates the policy in small, clipped steps to prevent destructive updates. The default algorithm for this project. Fast per-step but sample-inefficient (can't reuse old experience).

**SAC (Soft Actor-Critic)** -- Off-policy RL algorithm that maximizes both reward and action entropy (randomness). Stores past experience in a replay buffer for reuse. Slower per-step but more sample-efficient. Better with stochastic environments (e.g., random starting positions).

---

## Training Metrics

**approx_kl** -- Approximate KL divergence between the old and new policy after an update. Measures how much the policy changed. Values above 0.1 are a warning sign; above 0.5 usually means the policy is exploding.

**target_kl** -- A threshold for approx_kl. When set, PPO stops early within an epoch if the KL divergence exceeds this value. Acts as a safety brake against destructive updates. Recommended: 0.02 for CNN training.

**entropy** -- Measures how random the policy's actions are. High entropy means the agent is exploring; low entropy means it has settled on a strategy. Too low too early (< 0.5) indicates premature convergence.

**ent_coef (entropy coefficient)** -- Multiplier on the entropy bonus added to the loss. Higher values encourage more exploration. Typical range: 0.01-0.04 for PPO. SAC uses `auto` to tune this automatically.

**explained_variance** -- How well the value network predicts actual returns. 1.0 is perfect prediction; 0.0 is no better than predicting the mean; negative values mean the value network is actively harmful.

**learning_rate** -- Step size for gradient descent. Too high causes instability and collapse; too low converges slowly and may not reach high peaks. Optimal values differ between lidar (0.001) and CNN (0.0003).

---

## Stability Concepts

**PPO collapse** -- A pattern where PPO performance suddenly drops after reaching a peak, often never recovering. Fundamental to PPO on this task. All runs eventually collapse; the strategy is to checkpoint frequently and keep the best snapshot.

**Plasticity loss** -- A neural network's gradual loss of ability to learn new things. Weights grow large or neurons "die" (output zero for all inputs). Related to PPO collapse.

**Dead ReLU** -- A neuron that always outputs zero because its input is always negative. Once dead, it can never activate again. Contributes to plasticity loss over long training runs.

**Weight decay / L2 regularization** -- A technique that penalizes large network weights by adding a small fraction of the weight magnitude to the loss. Keeps weights small, prevents dead neurons, and helps maintain plasticity. In this project, `weight_decay=0.0001` enables higher peak performance.

**Shrink and perturb** -- A technique that periodically shrinks network weights toward zero and adds small random noise. Intended to restore plasticity but in practice destroyed learned policies in this project.

**LayerNorm** -- Normalizes activations within each layer to have zero mean and unit variance. Intended to stabilize training but caused collapse in our experiments.

---

## Observations

**Lidar observation** -- The agent "sees" the track via simulated laser rays cast from the car. Each ray returns a normalized distance [0,1] to the nearest wall. Default: 9 rays spanning a forward arc. Simple, fast, and effective.

**Grid (CNN) observation** -- The agent sees a 36x36 binary grid projected ahead of the car using a perspective (homographic) transform. Each cell is 1 (on track) or 0 (off track). Processed by a convolutional neural network (CNN) feature extractor.

**Homographic grid** -- A grid that simulates a camera looking ahead of the car at a downward angle. Nearby cells are densely spaced (high resolution close to the car) while distant cells are spread out, mimicking natural perspective. Parameters: near distance, far distance, FOV, camera pitch.

**Feature extractor** -- The CNN layers that process raw grid observations into a compact vector representation before it reaches the policy network. Learns to identify track features like curves, edges, and open space.

---

## Reward Components

**Progress reward** -- Per-step reward proportional to how much angular progress the car makes around the track. The primary convergence driver (scale 0.5-0.75). Higher values accelerate learning but can cause instability above 0.8.

**Checkpoint reward** -- +1.0 bonus for crossing the next checkpoint in sequence. The track has 64 evenly-spaced checkpoint lines. Provides coarse directional signal.

**Speed bonus** -- Small per-step reward for going fast (+0.05 * speed/max_speed). Encourages the agent to maintain speed rather than creep.

**Collision penalty** -- -20.0 penalty for hitting a wall, which also terminates the episode. The main negative signal.

---

## Training Infrastructure

**Eval callback** -- Periodically pauses training to run the agent in a separate evaluation environment and record performance. The eval reward is the primary metric for model quality. Models are saved when eval reward reaches a new high.

**VecTransposeImage** -- A Stable-Baselines3 wrapper that transposes image observations from HWC (height, width, channels) to CHW (channels, height, width) format, as required by PyTorch CNNs. Must be applied to both training and evaluation environments.

**Preset** -- A named set of default hyperparameters (fast, balanced, quality) defined in train.py. Provides sensible defaults that can be overridden via CLI flags.
