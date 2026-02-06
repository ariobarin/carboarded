"""Headless validation script for trained models."""

import argparse
import sys
import zipfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO, SAC
from racing_sim.envs.racing_env import RacingEnv
from racing_sim.config.config import EnvConfig


def detect_algo_from_model(model_path: Path) -> str:
    """Auto-detect algorithm type from saved model file."""
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "data" in zf.namelist():
                import json

                with zf.open("data") as f:
                    data = json.load(f)
                    policy_class = data.get("policy_class", "")
                    if "SAC" in str(policy_class) or "sac" in str(policy_class).lower():
                        return "sac"
                    if "PPO" in str(policy_class) or "ppo" in str(policy_class).lower():
                        return "ppo"
    except Exception:
        pass
    return "ppo"


def run_validation(
    model_path: str,
    config_path: str,
    num_episodes: int = 100,
    deterministic: bool = True,
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

    # Detect and load model
    algo = detect_algo_from_model(model_path)
    if algo == "ppo":
        model = PPO.load(str(model_path))
    else:
        model = SAC.load(str(model_path))

    print(f"Validating {algo.upper()} model: {model_path}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    run_validation(args.model, args.config, args.episodes, args.deterministic)
