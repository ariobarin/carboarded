"""Utilities for summarizing PPO rollout statistics."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class RolloutStatSummary:
    adv_mean: float
    adv_std: float
    adv_abs_mean: float
    ret_mean: float
    ret_std: float
    val_mean: float
    val_std: float


def _flatten(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array.reshape(-1)


def summarize_rollout_stats(
    advantages: Iterable[float],
    returns: Iterable[float],
    values: Iterable[float],
) -> RolloutStatSummary:
    """Compute basic statistics for PPO rollout buffers."""
    adv = _flatten(advantages)
    ret = _flatten(returns)
    val = _flatten(values)

    return RolloutStatSummary(
        adv_mean=float(np.mean(adv)),
        adv_std=float(np.std(adv)),
        adv_abs_mean=float(np.mean(np.abs(adv))),
        ret_mean=float(np.mean(ret)),
        ret_std=float(np.std(ret)),
        val_mean=float(np.mean(val)),
        val_std=float(np.std(val)),
    )


def extract_episode_stats(
    dones: Iterable[bool],
    infos: Iterable[Mapping[str, object]],
    reward_key: str = "episode_reward",
    collided_key: str = "collided",
) -> Tuple[list[float], int, int]:
    """Extract per-episode rewards and termination counts from step infos."""
    rewards: list[float] = []
    collisions = 0
    timeouts = 0

    for done, info in zip(dones, infos):
        if not done:
            continue
        info = info or {}
        if reward_key in info:
            rewards.append(float(info[reward_key]))
        if info.get(collided_key, False):
            collisions += 1
        else:
            timeouts += 1

    return rewards, collisions, timeouts
