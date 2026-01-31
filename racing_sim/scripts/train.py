"""Training script for PPO/SAC agents."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from racing_sim.config.config import EnvConfig

PRESETS = {
    "ppo": {
        "fast": dict(
            learning_rate=1e-2,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
        ),
        "balanced": dict(
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=8,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        ),
        "quality": dict(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        ),
    },
    "sac": {
        "fast": dict(
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            learning_starts=1_000,
        ),
        "balanced": dict(
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            learning_starts=1_000,
        ),
        "quality": dict(
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            learning_starts=1_000,
        ),
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train RL agents for racing")

    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="Algorithm to use (ppo or sac)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["fast", "balanced", "quality"],
        help="Hyperparameter preset"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="Total timesteps to train"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        default="dummy",
        choices=["auto", "dummy", "subproc"],
        help="Vectorized environment backend"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20000,
        help="Evaluation frequency (in timesteps)"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=2,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100000,
        help="Checkpoint save frequency (in timesteps)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for saved models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override preset learning rate"
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=None,
        help="Override PPO clip range"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Override discount factor (gamma)"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="Override PPO GAE lambda"
    )
    parser.add_argument(
        "--ent-coef",
        type=str,
        default=None,
        help="Override entropy coefficient (float for PPO, float or 'auto'/'auto_0.1' for SAC)"
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="Override SAC target entropy"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (SAC only)"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Override replay buffer size (SAC only)"
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="Override SAC learning_starts"
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=None,
        help="Override SAC train_freq"
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Override SAC gradient_steps"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override SAC tau (target network soft update coefficient)"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training"
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saves during training"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to pretrained model to load (for curriculum learning / fine-tuning)"
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Start car at random checkpoint each episode (improves exploration)"
    )
    parser.add_argument(
        "--cohort-spawn",
        action="store_true",
        help="PPO cohort spawning: all envs share the same random start per rollout. "
             "Gives PPO coherent gradients with random positions. Ignored for SAC."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Wrap env in VecNormalize for reward normalization (recommended for SAC stability)"
    )

    return parser


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = build_arg_parser()
    return parser.parse_args(argv)


def make_env(config: EnvConfig, rank: int, seed: int = 0):
    """Create a function that returns an environment instance."""
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from racing_sim.envs.racing_env import RacingEnv

        env = RacingEnv(config=config, render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def resolve_env_config(path_str: Optional[str]) -> Tuple[EnvConfig, str]:
    """Load config from path or default config file."""
    if path_str:
        return EnvConfig.from_yaml(path_str), path_str

    # Try fast_iter_v3_complex_progress_0p5.yaml as the default (simple track, proven config)
    default_path = Path(__file__).parent.parent / "configs" / "fast_iter_v3_complex_progress_0p5.yaml"
    if default_path.exists():
        return EnvConfig.from_yaml(str(default_path)), str(default_path)

    return EnvConfig(), "EnvConfig() defaults"


def make_vec_env(env_fns, vec_env_type: str):
    """Create a vectorized environment based on the requested backend."""
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns)
    if vec_env_type == "dummy":
        return DummyVecEnv(env_fns)
    if len(env_fns) > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def make_save_vec_normalize_callback(save_path: str, env):
    """Create a callback to save VecNormalize stats when a new best model is saved."""
    from stable_baselines3.common.callbacks import BaseCallback

    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.save_path = save_path
            self.env = env

        def _on_step(self):
            """Called on each step. Save VecNormalize stats."""
            vec_norm_path = Path(self.save_path) / "vecnormalize.pkl"
            self.env.save(str(vec_norm_path))
            return True

    return SaveVecNormalizeCallback()


def make_cohort_spawn_callback(num_checkpoints: int):
    """Create a callback that synchronizes spawn checkpoints across all envs.

    At the start of each PPO rollout, picks a random checkpoint and sets it
    on all vectorized envs. This ensures all episodes in a batch start from
    the same position, giving PPO coherent gradients while still training
    across all track positions over time.
    """
    import random as _random
    from stable_baselines3.common.callbacks import BaseCallback

    class CohortSpawnCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.num_checkpoints = num_checkpoints

        def _on_rollout_start(self):
            checkpoint = _random.randint(0, self.num_checkpoints - 1)
            self.training_env.env_method("set_spawn_checkpoint", checkpoint)
            if self.verbose > 0:
                self.logger.record("cohort/spawn_checkpoint", checkpoint)

        def _on_step(self):
            return True

    return CohortSpawnCallback()


def build_preset_kwargs(args) -> dict:
    """Build preset kwargs with CLI overrides applied."""
    preset_kwargs = PRESETS[args.algo][args.preset].copy()
    if args.learning_rate is not None:
        preset_kwargs["learning_rate"] = args.learning_rate
    if args.clip_range is not None and args.algo == "ppo":
        preset_kwargs["clip_range"] = args.clip_range
    if args.gamma is not None:
        preset_kwargs["gamma"] = args.gamma
    if args.gae_lambda is not None and args.algo == "ppo":
        preset_kwargs["gae_lambda"] = args.gae_lambda
    if args.ent_coef is not None:
        if args.algo == "ppo":
            preset_kwargs["ent_coef"] = float(args.ent_coef)
        else:
            try:
                preset_kwargs["ent_coef"] = float(args.ent_coef)
            except ValueError:
                preset_kwargs["ent_coef"] = args.ent_coef
    if args.algo == "sac" and args.target_entropy is not None:
        preset_kwargs["target_entropy"] = args.target_entropy
    if args.algo == "sac":
        if args.batch_size is not None:
            preset_kwargs["batch_size"] = args.batch_size
        if args.buffer_size is not None:
            preset_kwargs["buffer_size"] = args.buffer_size
        if args.learning_starts is not None:
            preset_kwargs["learning_starts"] = args.learning_starts
        if args.train_freq is not None:
            preset_kwargs["train_freq"] = args.train_freq
        if args.gradient_steps is not None:
            preset_kwargs["gradient_steps"] = args.gradient_steps
        if args.tau is not None:
            preset_kwargs["tau"] = args.tau
    return preset_kwargs


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config, config_source = resolve_env_config(args.config)

    # Apply CLI overrides to config
    if args.random_start:
        config.random_start = True

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_{args.preset}_{timestamp}"

    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Cohort spawn: override random_start for PPO
    use_cohort_spawn = False
    if args.cohort_spawn:
        if args.algo == "sac":
            print("Warning: --cohort-spawn is ignored for SAC (SAC handles random starts natively)")
        else:
            use_cohort_spawn = True
            # Disable per-env random_start since the callback handles it
            config.random_start = False

    print(f"Training {args.algo.upper()} for {args.total_timesteps} timesteps")
    print(f"Using {args.n_envs} parallel environments")
    print(f"Preset: {args.preset}")
    print(f"Config: {config_source}")
    print(f"Random start: {config.random_start}")
    if use_cohort_spawn:
        print(f"Cohort spawn: enabled (synchronized random starts per rollout)")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    # Create vectorized environment
    env_fns = [make_env(config, i, args.seed) for i in range(args.n_envs)]
    env = make_vec_env(env_fns, args.vec_env)

    # Wrap with VecNormalize for reward normalization (helps SAC stability)
    if args.normalize:
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
        print("VecNormalize enabled (norm_reward=True, norm_obs=False)")

    # Create evaluation environment
    eval_env = None
    if not args.no_eval:
        eval_env = make_vec_env([make_env(config, 0, args.seed + 1000)], "dummy")
        # If training env uses VecNormalize, eval env must also be wrapped for compatibility
        if args.normalize:
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99, training=False)

    # Create callbacks
    callbacks = []
    if not args.no_checkpoint:
        from stable_baselines3.common.callbacks import CheckpointCallback

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=str(model_dir),
            name_prefix=args.algo,
        )
        callbacks.append(checkpoint_callback)
    if eval_env is not None:
        from stable_baselines3.common.callbacks import EvalCallback

        # Create callback to save VecNormalize stats with best model
        callback_on_new_best = None
        if args.normalize:
            callback_on_new_best = make_save_vec_normalize_callback(str(model_dir / "best"), env)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            callback_on_new_best=callback_on_new_best,
        )
        callbacks.append(eval_callback)
    # Add cohort spawn callback for PPO random starts
    if use_cohort_spawn:
        cohort_callback = make_cohort_spawn_callback(num_checkpoints=64)
        callbacks.append(cohort_callback)

    callback = None
    if callbacks:
        from stable_baselines3.common.callbacks import CallbackList

        callback = CallbackList(callbacks)

    # Create model
    tensorboard_log = None if args.no_tensorboard else str(log_dir)
    preset_kwargs = build_preset_kwargs(args)

    if args.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            policy="MlpPolicy",
            env=env,
            **preset_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            device=args.device,
        )
    else:  # SAC
        from stable_baselines3 import SAC

        model = SAC(
            policy="MlpPolicy",
            env=env,
            **preset_kwargs,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            device=args.device,
        )

    # Load pretrained model if specified (for curriculum learning)
    if args.load_model:
        print(f"\nLoading pretrained model from: {args.load_model}")
        model.set_parameters(args.load_model)
        print("Model parameters loaded successfully")

    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=not args.no_progress,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = model_dir / f"{args.algo}_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")

    # Save VecNormalize stats if used
    if args.normalize:
        vec_norm_path = model_dir / "vecnormalize.pkl"
        env.save(str(vec_norm_path))
        print(f"VecNormalize stats saved to {vec_norm_path}")

    # Cleanup
    env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
