"""Callbacks for maintaining network plasticity during training.

Plasticity loss (the network's inability to learn new patterns) is a major
cause of PPO policy collapse. These callbacks implement techniques to
restore and maintain plasticity.

Reference: "Understanding Plasticity in Neural Networks" (Nature 2024)
"""

import torch as th
from stable_baselines3.common.callbacks import BaseCallback


class ShrinkPerturbCallback(BaseCallback):
    """Periodically shrink weights toward zero and add noise to restore plasticity.

    The Shrink+Perturb technique:
    1. Shrinks weights toward zero: weight = weight * shrink_factor
    2. Adds small random noise: weight = weight + noise(0, perturb_std)

    This resets neurons that have become "dead" (saturated or stuck) while
    preserving most of the learned policy. The shrinkage prevents weight
    explosion, and the perturbation restores plasticity.

    Args:
        shrink_factor: Multiplicative factor for weight shrinkage (default: 0.8)
        perturb_std: Standard deviation of Gaussian noise to add (default: 0.01)
        interval: Apply shrink+perturb every N training steps (default: 50000)
        verbose: Verbosity level (default: 0)
    """

    def __init__(
        self,
        shrink_factor: float = 0.8,
        perturb_std: float = 0.01,
        interval: int = 50_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.shrink_factor = shrink_factor
        self.perturb_std = perturb_std
        self.interval = interval
        self._last_applied_step = 0

    def _on_step(self) -> bool:
        """Check if we should apply shrink+perturb."""
        # Check if interval has passed since last application
        steps_since_last = self.num_timesteps - self._last_applied_step

        if steps_since_last >= self.interval and self.num_timesteps > 0:
            self._apply_shrink_perturb()
            self._last_applied_step = self.num_timesteps

        return True

    def _apply_shrink_perturb(self) -> None:
        """Apply shrink+perturb to all weight parameters in the policy."""
        policy = self.model.policy

        # Count parameters for logging
        total_params = 0
        affected_params = 0

        with th.no_grad():
            for name, param in policy.named_parameters():
                total_params += 1

                # Only apply to weight matrices, not biases or normalization params
                if "weight" in name and param.dim() >= 2:
                    # Shrink toward zero
                    param.mul_(self.shrink_factor)

                    # Add Gaussian noise
                    noise = th.randn_like(param) * self.perturb_std
                    param.add_(noise)

                    affected_params += 1

        if self.verbose > 0:
            print(
                f"[ShrinkPerturb] Step {self.num_timesteps}: "
                f"Applied to {affected_params}/{total_params} parameters "
                f"(shrink={self.shrink_factor}, perturb_std={self.perturb_std})"
            )

        # Log to TensorBoard if available
        if self.logger is not None:
            self.logger.record("shrink_perturb/step", self.num_timesteps)
            self.logger.record("shrink_perturb/affected_params", affected_params)
