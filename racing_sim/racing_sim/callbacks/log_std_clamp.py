"""Callback to clamp policy log_std during training."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from racing_sim.utils.training_utils import clamp_log_std


class LogStdClampCallback(BaseCallback):
    """Clamp policy log_std parameters to a fixed range."""

    def __init__(self, min_val: float, max_val: float, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.min_val = min_val
        self.max_val = max_val

    def _on_training_start(self) -> None:
        clamp_log_std(self.model.policy, self.min_val, self.max_val)

    def _on_step(self) -> bool:
        clamp_log_std(self.model.policy, self.min_val, self.max_val)
        return True
