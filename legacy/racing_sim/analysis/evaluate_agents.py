from __future__ import annotations

import csv
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


def run_episode(env: RacingEnv, agent, seed: int, obs_noise: float) -> dict:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    total_reward = 0.0
    steps = 0
    speed_sum = 0.0

    while not done:
        if obs_noise > 0.0:
            obs = np.clip(obs + rng.normal(0.0, obs_noise, size=obs.shape), 0.0, 1.0)
        if hasattr(agent, "predict"):
            action, _ = agent.predict(obs, deterministic=True)
        else:
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


def summarize(episode_stats: list[dict]) -> dict:
    rewards = np.array([s["reward"] for s in episode_stats], dtype=np.float32)
    steps = np.array([s["steps"] for s in episode_stats], dtype=np.float32)
    checkpoints = np.array([s["checkpoints"] for s in episode_stats], dtype=np.float32)
    collided = np.array([s["collided"] for s in episode_stats], dtype=np.float32)
    mean_speed = np.array([s["mean_speed"] for s in episode_stats], dtype=np.float32)

    return {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_steps": float(steps.mean()),
        "mean_checkpoints": float(checkpoints.mean()),
        "collision_rate": float(collided.mean()),
        "mean_speed": float(mean_speed.mean()),
        "best_reward": float(rewards.max()),
        "worst_reward": float(rewards.min()),
    }


def load_agent(spec: dict):
    agent_type = spec["type"]
    if agent_type == "sb3":
        algo = spec["algo"].lower()
        if algo == "ppo":
            from stable_baselines3 import PPO
            return PPO.load(spec["model_path"])
        if algo == "sac":
            from stable_baselines3 import SAC
            return SAC.load(spec["model_path"])
        raise ValueError(f"Unknown algo: {algo}")

    if agent_type == "rule":
        params = RuleBasedParams(**spec.get("params", {}))
        return RuleBasedAgent(params)

    if agent_type == "random":
        class RandomAgent:
            def __init__(self, action_space):
                self.action_space = action_space
            def act(self, _obs):
                return self.action_space.sample()
        return RandomAgent

    raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate agents and baselines.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "fast_iter_v3.yaml"))
    parser.add_argument("--agents", type=str, default=str(Path(__file__).resolve().parent / "agents.yaml"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="fast_iter_v3")
    parser.add_argument("--obs-noise", type=float, default=0.0)
    args = parser.parse_args()

    config = EnvConfig.from_yaml(args.config)

    with open(args.agents, "r") as f:
        manifest = yaml.safe_load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    episode_rows = []

    for spec in manifest["agents"]:
        env = RacingEnv(config=config, render_mode=None)

        if spec["type"] == "random":
            agent = load_agent(spec)(env.action_space)
        else:
            agent = load_agent(spec)

        stats = []
        for i in range(args.episodes):
            stats.append(run_episode(env, agent, args.seed + i, args.obs_noise))

        env.close()

        summary = summarize(stats)
        summary["name"] = spec["name"]
        summary["type"] = spec["type"]
        summary_rows.append(summary)

        for idx, row in enumerate(stats):
            episode_rows.append({
                "name": spec["name"],
                "episode": idx,
                **row,
            })

    summary_path = RESULTS_DIR / f"eval_summary_{args.tag}.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    episodes_path = RESULTS_DIR / f"eval_episodes_{args.tag}.csv"
    with episodes_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    print(f"Saved summary to {summary_path}")
    print(f"Saved episodes to {episodes_path}")


if __name__ == "__main__":
    main()
