"""Play/visualization script for manual control or trained models."""

import argparse
import sys
import zipfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO, SAC

from racing_sim.envs.racing_env import RacingEnv
from racing_sim.config.config import EnvConfig


def detect_algo_from_model(model_path: Path) -> str:
    """Auto-detect algorithm type from saved model file."""
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            if 'data' in zf.namelist():
                import json
                with zf.open('data') as f:
                    data = json.load(f)
                    # Check policy_class or other indicators
                    policy_class = data.get('policy_class', '')
                    if 'SAC' in str(policy_class) or 'sac' in str(policy_class).lower():
                        return 'sac'
                    if 'PPO' in str(policy_class) or 'ppo' in str(policy_class).lower():
                        return 'ppo'
    except Exception:
        pass
    # Default to ppo if can't detect
    return 'ppo'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play racing game or visualize trained agent")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (if not provided, use keyboard control)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="auto",
        choices=["auto", "ppo", "sac"],
        help="Algorithm used for trained model (auto-detects if not specified)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions for trained model"
    )

    return parser.parse_args()


def run_human_control(env: RacingEnv, num_episodes: int):
    """Run the environment with keyboard control."""
    print("\nControls:")
    print("  Arrow keys or WASD: Steer and accelerate")
    print("  ESC: Quit")
    print()

    episode_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # Render first to process events
            env.render()

            # Check for quit
            if not env.renderer.handle_events():
                print("\nQuit requested")
                env.close()
                return episode_rewards

            # Get keyboard input
            steering, throttle = env.renderer.get_keyboard_input()
            action = np.array([steering, throttle], dtype=np.float32)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(f"  Reward: {total_reward:.2f}, Checkpoints: {info['checkpoints_passed']}")

        if info['collided']:
            print("  Crashed!")

    return episode_rewards


def run_trained_model(
    env: RacingEnv,
    model,
    num_episodes: int,
    deterministic: bool = True
):
    """Run the environment with a trained model."""
    print(f"\nRunning trained model ({'deterministic' if deterministic else 'stochastic'} actions)")

    episode_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render
            env.render()

            # Check for quit
            if not env.renderer.handle_events():
                print("\nQuit requested")
                env.close()
                return episode_rewards

        episode_rewards.append(total_reward)
        print(f"  Reward: {total_reward:.2f}, Checkpoints: {info['checkpoints_passed']}")

        if info['collided']:
            print("  Crashed!")

    return episode_rewards


def main():
    """Main play function."""
    args = parse_args()

    # Load config
    if args.config:
        config = EnvConfig.from_yaml(args.config)
    else:
        config = EnvConfig()

    # Create environment with rendering
    env = RacingEnv(config=config, render_mode="human")

    print("Racing Simulation")
    print("=" * 40)

    if args.model:
        # Load trained model
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return

        # Auto-detect algorithm if needed
        algo = args.algo
        if algo == "auto":
            algo = detect_algo_from_model(model_path)
            print(f"Auto-detected algorithm: {algo.upper()}")

        print(f"Loading model from {model_path}")

        if algo == "ppo":
            model = PPO.load(str(model_path))
        else:
            model = SAC.load(str(model_path))

        episode_rewards = run_trained_model(
            env, model, args.episodes, args.deterministic
        )
    else:
        # Human control
        print("Mode: Human control")
        episode_rewards = run_human_control(env, args.episodes)

    # Print statistics
    if episode_rewards:
        print("\n" + "=" * 40)
        print("Statistics:")
        print(f"  Episodes: {len(episode_rewards)}")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"  Std reward: {np.std(episode_rewards):.2f}")
        print(f"  Min reward: {np.min(episode_rewards):.2f}")
        print(f"  Max reward: {np.max(episode_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
