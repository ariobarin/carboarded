"""Headless validation script for trained models."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from racing_sim.envs.racing_env import RacingEnv
from racing_sim.config.config import EnvConfig
from racing_sim.utils.model import load_model


def run_validation(
    model_path: str,
    config_path: str,
    num_episodes: int = 100,
    deterministic: bool = False,
    algo: str = "auto",
):
    """Run headless validation."""
    model_path = Path(model_path)

    # Load config
    if config_path:
        config = EnvConfig.from_yaml(config_path)
    else:
        config = EnvConfig()

    config.random_start = False

    # Create environment
    env = RacingEnv(config=config, render_mode=None)

    # Load model using shared utility
    model = load_model(model_path, algo=algo)
    algo_name = "PPO" if type(model).__name__ == "PPO" else "SAC"

    print(f"Validating {algo_name} model: {model_path}")
    print(f"Episodes: {num_episodes}, Deterministic: {deterministic}")
    print("-" * 50)

    episode_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            recent_mean = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode + 1}/{num_episodes}: Recent 10-ep mean = {recent_mean:.2f}"
            )

    env.close()

    print("\n" + "=" * 50)
    print("VALIDATION RESULTS:")
    print(f"  Episodes: {len(episode_rewards)}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Std reward: {np.std(episode_rewards):.2f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    print(
        f"  Success rate (>200): {sum(1 for r in episode_rewards if r > 200) / len(episode_rewards) * 100:.1f}%"
    )
    print("=" * 50)

    return episode_rewards


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument(
        "--algo",
        type=str,
        default="auto",
        choices=["auto", "ppo", "sac"],
        help="Algorithm type (auto-detects if not specified)",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_validation(args.model, args.config, args.episodes, args.deterministic, args.algo)
