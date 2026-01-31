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
        with zipfile.ZipFile(model_path, "r") as zf:
            if "data" in zf.namelist():
                import json

                with zf.open("data") as f:
                    data = json.load(f)
                    # Check policy_class or other indicators
                    policy_class = data.get("policy_class", "")
                    if "SAC" in str(policy_class) or "sac" in str(policy_class).lower():
                        return "sac"
                    if "PPO" in str(policy_class) or "ppo" in str(policy_class).lower():
                        return "ppo"
    except Exception:
        pass
    # Default to ppo if can't detect
    return "ppo"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Play racing game or visualize trained agent"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (if not provided, use keyboard control)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="auto",
        choices=["auto", "ppo", "sac"],
        help="Algorithm used for trained model (auto-detects if not specified)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML file"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions for trained model",
    )
    parser.add_argument(
        "--race",
        action="store_true",
        help="Race against a trained model (requires --model)",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Show 10x10 CNN grid visualization in front of car",
    )

    return parser.parse_args()


def run_human_control(env: RacingEnv, num_episodes: int):
    """Run the environment with keyboard control."""
    print("\nControls:")
    print("  Arrow keys or WASD: Steer and accelerate")
    print("  R: Reset race")
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

            # Check for reset
            if env.renderer.was_reset_requested():
                obs, info = env.reset()
                done = False
                total_reward = 0.0
                continue

            # Get keyboard input
            steering, throttle = env.renderer.get_keyboard_input()
            action = np.array([steering, throttle], dtype=np.float32)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)
        print(
            f"  Reward: {total_reward:.2f}, Checkpoints: {info['checkpoints_passed']}"
        )

        if info["collided"]:
            print("  Crashed!")

    return episode_rewards


def run_trained_model(
    env: RacingEnv, model, num_episodes: int, deterministic: bool = True
):
    """Run the environment with a trained model."""
    print(
        f"\nRunning trained model ({'deterministic' if deterministic else 'stochastic'} actions)"
    )

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

            # Check for reset
            if env.renderer.was_reset_requested():
                obs, info = env.reset()
                done = False
                total_reward = 0.0
                continue

        episode_rewards.append(total_reward)
        print(
            f"  Reward: {total_reward:.2f}, Checkpoints: {info['checkpoints_passed']}"
        )

        if info["collided"]:
            print("  Crashed!")

    return episode_rewards


def run_race_mode(env: RacingEnv, model, deterministic: bool = True):
    """Race the human against a trained model (debug mode)."""
    import pygame
    import math
    from racing_sim.physics.car import Car
    from racing_sim.sensors.lidar import Lidar
    from racing_sim.utils.progress import progress_delta

    print("\nRace Mode (Debug)")
    print("Controls: Arrow keys or WASD to drive")
    print("R: Reset race")
    print("Blue = You, Red = AI")
    print("ESC to quit.\n")

    # Create a second car for the AI in the same space
    ai_car = Car(
        env.space,
        env.config.car,
        position=(env.track.start_position.x, env.track.start_position.y),
        angle=env.track.start_angle,
    )
    ai_lidar = Lidar(env.space, env.config.lidar)

    # Reset human car to start
    env.car.reset(
        position=(env.track.start_position.x, env.track.start_position.y),
        angle=env.track.start_angle,
    )

    # Track state for reward calculation
    human_checkpoint = env.track.get_checkpoint_index(env.car.position)
    ai_checkpoint = env.track.get_checkpoint_index(ai_car.position)
    human_reward = 0.0
    ai_reward = 0.0

    def get_progress_angle(position):
        delta = position - env.track.center
        angle = math.atan2(delta.y, delta.x)
        return angle if angle >= 0 else angle + math.tau

    human_progress_angle = get_progress_angle(env.car.position)
    ai_progress_angle = get_progress_angle(ai_car.position)

    running = True

    while running:
        # Handle events (must init pygame first)
        env.renderer._init_pygame()
        if not env.renderer.handle_events():
            print("Quit requested")
            break

        # Check for reset
        if env.renderer.was_reset_requested():
            env.car.reset(
                position=(env.track.start_position.x, env.track.start_position.y),
                angle=env.track.start_angle,
            )
            ai_car.reset(
                position=(env.track.start_position.x, env.track.start_position.y),
                angle=env.track.start_angle,
            )
            human_checkpoint = env.track.get_checkpoint_index(env.car.position)
            ai_checkpoint = env.track.get_checkpoint_index(ai_car.position)
            human_progress_angle = get_progress_angle(env.car.position)
            ai_progress_angle = get_progress_angle(ai_car.position)
            human_reward = 0.0
            ai_reward = 0.0
            continue

        # Human input
        steering, throttle = env.renderer.get_keyboard_input()

        # AI input
        ai_obs = ai_lidar.scan(ai_car)
        ai_action, _ = model.predict(ai_obs, deterministic=deterministic)

        # Apply controls
        env.car.apply_control(steering, throttle)
        ai_car.apply_control(float(ai_action[0]), float(ai_action[1]))

        # Update friction
        env.car.update_friction()
        ai_car.update_friction()

        # Step physics
        env.space.step(env.config.physics_dt)

        # Calculate rewards (simplified version of env reward)
        for car, checkpoint, progress_angle, is_human in [
            (env.car, human_checkpoint, human_progress_angle, True),
            (ai_car, ai_checkpoint, ai_progress_angle, False),
        ]:
            reward = env.config.time_penalty
            reward += env.config.speed_bonus_scale * (
                car.speed / env.config.car.max_speed
            )

            new_cp, crossed = env.track.get_progress(car.position, checkpoint)
            if crossed:
                reward += env.config.checkpoint_reward
                if is_human:
                    human_checkpoint = new_cp
                else:
                    ai_checkpoint = new_cp

            if env.config.progress_reward_scale != 0.0:
                new_angle = get_progress_angle(car.position)
                delta = progress_delta(progress_angle, new_angle)
                reward += env.config.progress_reward_scale * delta
                if is_human:
                    human_progress_angle = new_angle
                else:
                    ai_progress_angle = new_angle

            if car.collided:
                reward += env.config.collision_penalty

            if is_human:
                human_reward += reward
            else:
                ai_reward += reward

        # Reset on collision
        if env.car.collided or ai_car.collided:
            env.car.reset(
                position=(env.track.start_position.x, env.track.start_position.y),
                angle=env.track.start_angle,
            )
            ai_car.reset(
                position=(env.track.start_position.x, env.track.start_position.y),
                angle=env.track.start_angle,
            )
            human_checkpoint = env.track.get_checkpoint_index(env.car.position)
            ai_checkpoint = env.track.get_checkpoint_index(ai_car.position)
            human_progress_angle = get_progress_angle(env.car.position)
            ai_progress_angle = get_progress_angle(ai_car.position)

        # Render
        env.renderer.screen.fill(env.renderer.config.background_color)
        env.renderer._render_track(env.track)
        env.renderer._render_checkpoints(env.track)
        env.renderer._render_lidar(ai_lidar)  # Show AI's lidar
        env.renderer._render_car(env.car, color=(100, 150, 255))  # Blue for human
        env.renderer._render_car(ai_car, color=(255, 100, 100))  # Red for AI

        # HUD with rewards
        font = pygame.font.Font(None, 24)
        texts = [
            f"You (Blue): {human_reward:.1f}",
            f"AI (Red): {ai_reward:.1f}",
        ]
        y = 10
        for text in texts:
            surface = font.render(text, True, (255, 255, 255))
            env.renderer.screen.blit(surface, (10, y))
            y += 20

        pygame.display.flip()
        env.renderer.clock.tick(env.renderer.config.fps)

    env.close()


def main():
    """Main play function."""
    args = parse_args()

    # Validate race mode
    if args.race and not args.model:
        print("Error: --race requires --model to specify the AI opponent")
        return

    # Load config
    if args.config:
        config = EnvConfig.from_yaml(args.config)
    else:
        config = EnvConfig()

    # Disable random start for play/race modes
    config.random_start = False

    # Enable grid visualization if requested
    if args.show_grid:
        config.render.show_grid = True

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

        if args.race:
            # Race mode: human vs AI
            run_race_mode(env, model, args.deterministic)
            return
        else:
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
