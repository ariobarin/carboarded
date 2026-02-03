"""Callback for logging rollout statistics and termination rates."""

from __future__ import annotations

from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from racing_sim.utils.rollout_stats import extract_episode_stats, summarize_rollout_stats


class RolloutStatsCallback(BaseCallback):
    """Logs advantage/return/value stats and episode termination rates."""

    def __init__(self, log_freq: int = 1):
        super().__init__()
        if log_freq < 1:
            raise ValueError("log_freq must be >= 1")
        self.log_freq = log_freq
        self._rollout_count = 0
        self._episode_rewards: List[float] = []
        self._collisions = 0
        self._timeouts = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        rewards, collisions, timeouts = extract_episode_stats(dones, infos)
        if rewards:
            self._episode_rewards.extend(rewards)
        self._collisions += collisions
        self._timeouts += timeouts
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.log_freq != 0:
            return

        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is not None:
            stats = summarize_rollout_stats(
                rollout_buffer.advantages,
                rollout_buffer.returns,
                rollout_buffer.values,
            )
            self.logger.record("rollout/adv_mean", stats.adv_mean)
            self.logger.record("rollout/adv_std", stats.adv_std)
            self.logger.record("rollout/adv_abs_mean", stats.adv_abs_mean)
            self.logger.record("rollout/ret_mean", stats.ret_mean)
            self.logger.record("rollout/ret_std", stats.ret_std)
            self.logger.record("rollout/value_mean", stats.val_mean)
            self.logger.record("rollout/value_std", stats.val_std)

        if self._episode_rewards:
            rewards = np.asarray(self._episode_rewards, dtype=np.float32)
            self.logger.record("rollout/episode_reward_mean", float(np.mean(rewards)))
            self.logger.record("rollout/episode_reward_std", float(np.std(rewards)))
            self.logger.record("rollout/episode_reward_min", float(np.min(rewards)))
            self.logger.record("rollout/episode_reward_max", float(np.max(rewards)))

        total_eps = self._collisions + self._timeouts
        if total_eps > 0:
            self.logger.record("rollout/episodes", total_eps)
            self.logger.record("rollout/term_collision_rate", self._collisions / total_eps)
            self.logger.record("rollout/term_timeout_rate", self._timeouts / total_eps)

        self._episode_rewards = []
        self._collisions = 0
        self._timeouts = 0
