import numpy as np

from racing_sim.utils.rollout_stats import extract_episode_stats, summarize_rollout_stats


def test_summarize_rollout_stats_computes_basic_moments():
    advantages = np.array([[1.0, -1.0], [3.0, -3.0]], dtype=np.float32)
    returns = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    summary = summarize_rollout_stats(advantages, returns, values)

    assert np.isclose(summary.adv_mean, 0.0)
    assert np.isclose(summary.adv_std, np.std(advantages))
    assert np.isclose(summary.adv_abs_mean, np.mean(np.abs(advantages)))
    assert np.isclose(summary.ret_mean, np.mean(returns))
    assert np.isclose(summary.ret_std, np.std(returns))
    assert np.isclose(summary.val_mean, np.mean(values))
    assert np.isclose(summary.val_std, np.std(values))


def test_extract_episode_stats_counts_terminations_and_rewards():
    dones = [False, True, True, True]
    infos = [
        {"episode_reward": 1.0, "collided": False},
        {"episode_reward": 2.0, "collided": True},
        {"episode_reward": 3.0, "collided": False},
        {"collided": False},
    ]

    rewards, collisions, timeouts = extract_episode_stats(dones, infos)

    assert rewards == [2.0, 3.0]
    assert collisions == 1
    assert timeouts == 2
