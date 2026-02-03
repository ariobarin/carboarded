"""Callback for logging gradient and update norms."""

from __future__ import annotations

from typing import List

import torch
from stable_baselines3.common.callbacks import BaseCallback

from racing_sim.utils.training_utils import compute_grad_norm, compute_update_norm


class GradStatsCallback(BaseCallback):
    """Log gradient and update norms at a fixed timestep interval."""

    def __init__(self, log_freq: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self._last_log_step = 0
        self._snapshot: List[torch.Tensor] = []

    def _snapshot_params(self) -> List[torch.Tensor]:
        params = []
        for param in self.model.policy.parameters():
            params.append(param.detach().float().cpu().clone())
        return params

    def _on_training_start(self) -> None:
        self._snapshot = self._snapshot_params()
        self._last_log_step = self.num_timesteps

    def _on_step(self) -> bool:
        if self.log_freq <= 0:
            return True
        if self.num_timesteps - self._last_log_step < self.log_freq:
            return True

        current = self._snapshot_params()
        update_norm = compute_update_norm(self._snapshot, current)
        grad_norm = compute_grad_norm(self.model.policy)

        self.logger.record("train/update_norm", update_norm)
        if grad_norm > 0.0:
            self.logger.record("train/grad_norm", grad_norm)

        self._snapshot = current
        self._last_log_step = self.num_timesteps
        return True
