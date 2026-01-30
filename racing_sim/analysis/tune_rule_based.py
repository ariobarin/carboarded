from __future__ import annotations

import csv
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS_DIR = Path(__file__).resolve().parent / "results"

from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv

from analysis.rule_based import RuleBasedAgent, RuleBasedParams


def run_episode(env: RacingEnv, agent: RuleBasedAgent, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0
    speed_sum = 0.0

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        speed_sum += float(info.get("speed", 0.0))
        done = terminated or truncated

    return {
        "reward": total_reward,
        "steps": steps,
        "checkpoints": int(info.get("checkpoints_passed", 0)),
        "collided": bool(info.get("collided", False)),
        "mean_speed": speed_sum / max(steps, 1),
    }


def evaluate_params(
    config: EnvConfig,
    params: RuleBasedParams,
    episodes: int,
    seed: int,
) -> dict:
    env = RacingEnv(config=config, render_mode=None)
    agent = RuleBasedAgent(params)
    stats = [run_episode(env, agent, seed + i) for i in range(episodes)]
    env.close()

    rewards = np.array([s["reward"] for s in stats], dtype=np.float32)
    checkpoints = np.array([s["checkpoints"] for s in stats], dtype=np.float32)
    collided = np.array([s["collided"] for s in stats], dtype=np.float32)
    mean_speed = np.array([s["mean_speed"] for s in stats], dtype=np.float32)

    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_checkpoints": float(checkpoints.mean()),
        "collision_rate": float(collided.mean()),
        "mean_speed": float(mean_speed.mean()),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune rule-based policy via grid search.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "fast_iter_v3.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="fast_iter_v3")
    args = parser.parse_args()

    config = EnvConfig.from_yaml(args.config)

    grid = {
        "steer_gain": [0.8, 1.2, 1.6, 2.0],
        "throttle_base": [0.1, 0.2, 0.3],
        "throttle_gain": [0.4, 0.6, 0.8],
        "brake_dist": [0.3, 0.4, 0.5],
        "throttle_min": [0.0, 0.05, 0.1],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None

    for values in product(*grid.values()):
        params = RuleBasedParams(**dict(zip(grid.keys(), values)))
        metrics = evaluate_params(config, params, args.episodes, args.seed)
        row = {**asdict(params), **metrics}
        rows.append(row)
        if best is None or row["mean_reward"] > best["mean_reward"]:
            best = row

    results_path = RESULTS_DIR / f"rule_based_grid_{args.tag}.csv"
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best_path = RESULTS_DIR / f"rule_based_best_{args.tag}.yaml"
    with best_path.open("w") as f:
        yaml.safe_dump(best, f, sort_keys=False)

    print(f"Saved grid results to {results_path}")
    print(f"Best params saved to {best_path}")
    print(best)


if __name__ == "__main__":
    main()
