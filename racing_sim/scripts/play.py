"""Play/visualization script for manual control or trained models."""

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from racing_sim.envs.racing_env import RacingEnv
from racing_sim.config.defaults import resolve_env_config
from racing_sim.physics.car import Car
from racing_sim.sensors.lidar import Lidar
from racing_sim.sensors.grid import compute_grid


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
        "--cnn",
        action="store_true",
        help="Use CNN grid observations (required when loading a CNN-trained model)",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Enable grid visualization overlay at startup",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override render FPS (default: 60)",
    )

    return parser.parse_args()


def print_episode_stats(episode_rewards: List[float]) -> None:
    """Print summary statistics for completed episodes."""
    if not episode_rewards:
        return
    print("\n" + "=" * 40)
    print("Statistics:")
    print(f"  Episodes: {len(episode_rewards)}")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"  Std reward: {np.std(episode_rewards):.2f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")


def run_episodes(
    env: RacingEnv,
    num_episodes: int,
    get_action: Callable[[np.ndarray, RacingEnv], np.ndarray],
    pre_step: Optional[Callable[[RacingEnv], None]] = None,
    action_key: str = "human",
) -> List[float]:
    """Unified episode runner for both human and model control.

    Args:
        env: The racing environment.
        num_episodes: Number of episodes to run.
        get_action: Callable that takes (obs, env) and returns an action.
        pre_step: Optional callable to run before each step (e.g., render for human control).
        action_key: Key for renderer action bars ("human" or "ai").

    Returns:
        List of total rewards for each completed episode.
    """
    episode_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # Optional pre-step hook (render before action for human control)
            if pre_step:
                pre_step(env)

            # Handle quit/reset events
            if not env.renderer.handle_events():
                print("\nQuit requested")
                env.close()
                return episode_rewards

            if env.renderer.was_random_start_toggle_requested():
                env.config.random_start = not env.config.random_start
                state = "enabled" if env.config.random_start else "disabled"
                print(f"Random starts {state} (apply on next reset)")

            if env.renderer.was_reset_requested():
                obs, info = env.reset()
                done = False
                total_reward = 0.0
                continue

            # Get and execute action
            action = get_action(obs, env)

            # Update action bars on renderer
            env.renderer.last_actions = {
                action_key: (float(action[0]), float(action[1]))
            }

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render after step (for model control the action is visualized)
            if not pre_step:
                env.render()

        episode_rewards.append(total_reward)
        print(
            f"  Reward: {total_reward:.2f}, Checkpoints: {info['checkpoints_passed']}"
        )

        if info["collided"]:
            print("  Crashed!")

    return episode_rewards


def get_human_action(obs: np.ndarray, env: RacingEnv) -> np.ndarray:
    """Get action from keyboard input."""
    steering, throttle = env.renderer.get_keyboard_input()
    return np.array([steering, throttle], dtype=np.float32)


def make_model_action_fn(model, deterministic: bool = True):
    """Create an action function for a trained model."""
    def get_action(obs: np.ndarray, env: RacingEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    return get_action


def run_human_control(env: RacingEnv, num_episodes: int) -> List[float]:
    """Run the environment with keyboard control."""
    print("\nControls:")
    print("  Arrow keys or WASD: Steer and accelerate")
    print("  R: Reset race")
    print("  T: Toggle random starts")
    print("  L: Toggle lidar visualization")
    print("  G: Toggle grid visualization")
    print("  C: Toggle car POV mode (see what AI sees)")
    print("  ESC: Quit")
    print()

    def pre_step(env):
        env.render()

    return run_episodes(env, num_episodes, get_human_action, pre_step=pre_step)


def run_trained_model(
    env: RacingEnv, model, num_episodes: int, deterministic: bool = True
) -> List[float]:
    """Run the environment with a trained model."""
    print(
        f"\nRunning trained model ({'deterministic' if deterministic else 'stochastic'} actions)"
    )
    action_fn = make_model_action_fn(model, deterministic)
    return run_episodes(env, num_episodes, action_fn, action_key="ai")


def run_race_mode(env: RacingEnv, model, deterministic: bool = True) -> None:
    """Race mode: human vs AI for debugging/visualization.

    Simple visualization showing both cars. When human crashes, immediate reset.
    AI car is "ghost" - visible but doesn't affect human's physics.
    """
    print("\nRace Mode (Human vs AI)")
    print("Controls: Arrow keys or WASD to drive")
    print("R: Reset race")
    print("T: Toggle random starts")
    print("L: Toggle lidar visualization")
    print("G: Toggle grid visualization")
    print("Blue = You, Red = AI")
    print("ESC to quit.\n")

    # Create a ghost AI car (no collision with human, just for visualization)
    # Don't set up collision handler - it would override human car's handler
    ai_car = Car(
        env.space,
        env.config.car,
        position=(env.track.start_position.x, env.track.start_position.y),
        angle=env.track.start_angle,
        setup_collision_handler=False,
    )
    ai_lidar = Lidar(env.space, env.config.lidar)

    # Register collision handler for AI car using a separate collision type
    # (collision_type=3 avoids overwriting the human car's (1, 2) handler)
    ai_car.shape.collision_type = 3
    ai_car.sensor_only = True
    env.space.on_collision(
        collision_type_a=3,
        collision_type_b=2,
        begin=ai_car._on_collision_begin,
        data={"car": ai_car},
    )

    # Initialize pygame
    env.renderer._init_pygame()

    def reset_race():
        env.reset()
        env.car.sensor_only = False
        ai_car.reset(
            position=(env.car.position.x, env.car.position.y),
            angle=env.car.angle,
        )

    reset_race()

    running = True
    while running:
        # Handle events first
        if not env.renderer.handle_events():
            print("\nQuit requested")
            break

        if env.renderer.was_random_start_toggle_requested():
            env.config.random_start = not env.config.random_start
            state = "enabled" if env.config.random_start else "disabled"
            print(f"Random starts {state} (apply on next reset)")

        # Handle reset request
        if env.renderer.was_reset_requested():
            reset_race()
            continue

        # Get human input
        steering, throttle = env.renderer.get_keyboard_input()
        action = np.array([steering, throttle], dtype=np.float32)

        # Step environment (sensor_only mode prevents car from stopping on collision)
        obs, reward, terminated, truncated, info = env.step(action)

        # If human crashed, reset immediately (before rendering the crash)
        if info["collided"] and env.config.terminate_on_collision:
            print(f"Crashed! (Checkpoints: {info['checkpoints_passed']})")
            reset_race()
            continue

        # If AI crashed, reset both cars
        if ai_car.collided:
            print(f"AI Crashed! (Checkpoints: {info['checkpoints_passed']})")
            reset_race()
            continue

        # Get AI observation based on config (lidar or grid)
        if env.config.obs_type == "grid":
            grid = compute_grid(
                ai_car.position, ai_car.angle, env.track, env.config.grid
            )
            ai_obs = grid[:, :, np.newaxis]  # Shape: (36, 36, 1) for CNN
        else:
            ai_obs = ai_lidar.scan(ai_car)  # Shape: (5,) for lidar

        # Get AI action and apply to ghost car
        ai_action, _ = model.predict(ai_obs, deterministic=deterministic)
        ai_car.apply_control(float(ai_action[0]), float(ai_action[1]))
        ai_car.update_friction(env.config.physics_dt)

        # Update action bars for both players
        env.renderer.last_actions = {
            "human": (float(action[0]), float(action[1])),
            "ai": (float(ai_action[0]), float(ai_action[1])),
        }

        # Render with AI's sensors visualized
        env.renderer.render(
            car=env.car,
            track=env.track,
            lidar=ai_lidar,
            info=info,
            ai_car=ai_car,
            sensor_car=ai_car,
        )

    env.close()


def main():
    """Main play function."""
    args = parse_args()

    if args.race and not args.model:
        print("Error: --race requires --model to specify the AI opponent")
        return

    # Load config
    config, _ = resolve_env_config(args.config)

    # Disable random start by default for play/race modes (toggle with T)
    config.random_start = False

    # Enable CNN grid observations if requested
    if args.cnn:
        config.obs_type = "grid"

    # Load model early so we can auto-detect obs_type before creating the env
    model = None
    algo = args.algo
    if args.model:
        from racing_sim.utils.model import detect_algo_from_model, infer_obs_type, load_model

        model_path = Path(args.model)
        if algo == "auto":
            algo = detect_algo_from_model(model_path)

        model = load_model(model_path, algo=algo)

        # Reconcile obs_type: model's obs space wins unless --cnn was explicit
        if not args.cnn:
            detected = infer_obs_type(model)
            if detected and detected != config.obs_type:
                print(f"Note: model expects '{detected}' observations, overriding config obs_type='{config.obs_type}'")
                config.obs_type = detected

    # Enable grid visualization if requested
    if args.show_grid or config.obs_type == "grid":
        config.render.show_grid = True

    # Override FPS if requested
    if args.fps is not None:
        config.render.fps = args.fps

    # Create environment with rendering
    env = RacingEnv(config=config, render_mode="human")

    print("Racing Simulation")
    print("=" * 40)

    if model is not None:
        print(f"Algorithm: {algo.upper()}")
        print(f"Model: {args.model}")

        if args.race:
            run_race_mode(env, model, args.deterministic)
            return
        else:
            episode_rewards = run_trained_model(
                env, model, args.episodes, args.deterministic
            )
    else:
        print("Mode: Human control")
        episode_rewards = run_human_control(env, args.episodes)

    # Print statistics
    print_episode_stats(episode_rewards)

    env.close()


if __name__ == "__main__":
    main()
