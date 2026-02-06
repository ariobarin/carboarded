"""Training script for PPO/SAC agents."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from racing_sim.config.config import EnvConfig
from racing_sim.config.defaults import load_training_presets, resolve_env_config


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
        "--config-list",
        type=str,
        default=None,
        help="Comma-separated list of config YAML files (multi-track training)"
    )
    parser.add_argument(
        "--multi-track-mode",
        type=str,
        default="round_robin",
        choices=["round_robin", "random"],
        help="Track selection mode when using --config-list"
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
        default=1,
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
        help="Override learning rate"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["linear", "cosine"],
        help="LR schedule: linear decay to 10%% of initial, cosine decay to 10%% of initial"
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
        "--ppo-batch-size",
        type=int,
        default=None,
        help="Override PPO batch size"
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
        "--target-kl",
        type=float,
        default=None,
        help="PPO target KL divergence for early stopping updates (e.g. 0.01-0.03)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override PPO n_epochs (number of passes through rollout data per update)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Override PPO n_steps (rollout length per env)"
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
        help="(Deprecated) PPO cohort spawning is enabled by default. "
             "Use --no-cohort-spawn to disable. Ignored for SAC."
    )
    parser.add_argument(
        "--no-cohort-spawn",
        action="store_true",
        help="Disable PPO cohort spawning (use config.random_start or fixed start instead)"
    )
    parser.add_argument(
        "--cohort-checkpoint",
        type=int,
        default=None,
        help="Fixed checkpoint index for PPO cohort spawning. "
             "When set, all rollouts start at this checkpoint (ignores randomness)."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Wrap env in VecNormalize for reward normalization (recommended for SAC stability)"
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Deprecated (forces obs_type='grid'); kept for backward compatibility"
    )

    # Phase 3: Plasticity loss prevention arguments
    parser.add_argument(
        "--layernorm",
        action="store_true",
        help="Use LayerNorm policy (prevents dead ReLUs and weight explosion)"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=None,
        help="L2 regularization weight (weight_decay for optimizer, e.g. 0.0001)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability for policy/value MLPs (0 disables)"
    )
    parser.add_argument(
        "--adam-betas",
        type=float,
        nargs=2,
        default=None,
        metavar=("BETA1", "BETA2"),
        help="Adam optimizer betas (e.g. 0.99 0.99 for equal momentum/variance decay)"
    )
    parser.add_argument(
        "--shrink-perturb",
        action="store_true",
        help="Enable Shrink+Perturb callback (periodically resets plasticity)"
    )
    parser.add_argument(
        "--shrink-interval",
        type=int,
        default=50000,
        help="Steps between shrink+perturb applications (default: 50000)"
    )
    parser.add_argument(
        "--freeze-cnn-layers",
        type=int,
        default=0,
        help="Freeze the first N CNN layers (grid obs only). Default: 0"
    )
    parser.add_argument(
        "--log-std-min",
        type=float,
        default=None,
        help="Clamp policy log_std minimum (PPO only)"
    )
    parser.add_argument(
        "--log-std-max",
        type=float,
        default=None,
        help="Clamp policy log_std maximum (PPO only)"
    )
    parser.add_argument(
        "--grad-log-freq",
        type=int,
        default=0,
        help="Log grad/update norms every N timesteps (0 disables)"
    )
    parser.add_argument(
        "--rollout-log-freq",
        type=int,
        default=0,
        help="Log rollout stats every N rollouts (0 disables)"
    )

    return parser


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = build_arg_parser()
    return parser.parse_args(argv)


def parse_config_list_arg(config_list: Optional[str]) -> list[str]:
    """Parse comma-separated config list CLI argument."""
    if not config_list:
        return []
    paths = [p.strip() for p in config_list.split(",") if p.strip()]
    if not paths:
        raise ValueError("--config-list must contain at least one path.")
    return paths


def load_config_list(paths: list[str]) -> list[EnvConfig]:
    """Load EnvConfig objects from a list of paths."""
    return [EnvConfig.from_yaml(path) for path in paths]


def get_num_checkpoints(config: EnvConfig) -> int:
    """Resolve the checkpoint count for a config."""
    if config.track.track_type == "custom" and config.track.custom is not None:
        return config.track.custom.num_checkpoints
    return 64


def make_env(
    configs: list[EnvConfig],
    rank: int,
    seed: int = 0,
    multi_track: bool = False,
    multi_track_mode: str = "round_robin",
):
    """Create a function that returns an environment instance."""
    def _init():
        from stable_baselines3.common.monitor import Monitor

        if multi_track:
            from racing_sim.envs.multi_track_env import MultiTrackEnv
            env = MultiTrackEnv(configs=configs, render_mode=None, mode=multi_track_mode)
        else:
            from racing_sim.envs.racing_env import RacingEnv
            env = RacingEnv(config=configs[0], render_mode=None)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


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


def make_cohort_spawn_callback(num_checkpoints: int, fixed_checkpoint: int | None = None):
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
            self.fixed_checkpoint = fixed_checkpoint

        def _on_rollout_start(self):
            if self.fixed_checkpoint is not None:
                checkpoint = int(self.fixed_checkpoint) % max(self.num_checkpoints, 1)
            else:
                checkpoint = _random.randint(0, self.num_checkpoints - 1)
            self.training_env.env_method("set_spawn_checkpoint", checkpoint)
            if self.verbose > 0:
                self.logger.record("cohort/spawn_checkpoint", checkpoint)

        def _on_step(self):
            return True

    return CohortSpawnCallback()


def make_lr_schedule(initial_lr: float, schedule_type: str):
    """Create a learning rate schedule callable for SB3.

    Args:
        initial_lr: Starting learning rate.
        schedule_type: 'linear' or 'cosine'. Both decay to 10% of initial.

    Returns:
        Callable that takes progress_remaining (1.0 -> 0.0) and returns LR.
    """
    import math as _math
    min_lr = initial_lr * 0.1

    if schedule_type == "linear":
        def schedule(progress_remaining: float) -> float:
            return min_lr + (initial_lr - min_lr) * progress_remaining
        return schedule
    elif schedule_type == "cosine":
        def schedule(progress_remaining: float) -> float:
            return min_lr + (initial_lr - min_lr) * 0.5 * (1 + _math.cos(_math.pi * (1 - progress_remaining)))
        return schedule
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def build_training_kwargs(args, defaults: dict) -> Tuple[dict, dict]:
    """Build training kwargs with CLI overrides applied.

    Returns:
        (training_kwargs, extra_defaults) where extra_defaults holds
        non-SB3 values like l2_reg.
    """
    training_kwargs = defaults[args.algo].copy()
    extra_defaults = {}
    if "l2_reg" in training_kwargs:
        extra_defaults["l2_reg"] = training_kwargs.pop("l2_reg")
    if args.learning_rate is not None:
        training_kwargs["learning_rate"] = args.learning_rate
    if args.lr_schedule is not None:
        base_lr = training_kwargs.get("learning_rate", 3e-4)
        if not isinstance(base_lr, (int, float)):
            base_lr = 3e-4
        training_kwargs["learning_rate"] = make_lr_schedule(float(base_lr), args.lr_schedule)
    if args.clip_range is not None and args.algo == "ppo":
        training_kwargs["clip_range"] = args.clip_range
    if args.gamma is not None:
        training_kwargs["gamma"] = args.gamma
    if args.gae_lambda is not None and args.algo == "ppo":
        training_kwargs["gae_lambda"] = args.gae_lambda
    if args.ent_coef is not None:
        if args.algo == "ppo":
            training_kwargs["ent_coef"] = float(args.ent_coef)
        else:
            try:
                training_kwargs["ent_coef"] = float(args.ent_coef)
            except ValueError:
                training_kwargs["ent_coef"] = args.ent_coef
    if args.algo == "ppo" and args.target_kl is not None:
        training_kwargs["target_kl"] = args.target_kl
    if args.algo == "ppo" and args.n_epochs is not None:
        training_kwargs["n_epochs"] = args.n_epochs
    if args.algo == "ppo" and args.n_steps is not None:
        training_kwargs["n_steps"] = args.n_steps
    if args.algo == "ppo" and args.ppo_batch_size is not None:
        training_kwargs["batch_size"] = args.ppo_batch_size
    if args.algo == "sac" and args.target_entropy is not None:
        training_kwargs["target_entropy"] = args.target_entropy
    if args.algo == "sac":
        if args.batch_size is not None:
            training_kwargs["batch_size"] = args.batch_size
        if args.buffer_size is not None:
            training_kwargs["buffer_size"] = args.buffer_size
        if args.learning_starts is not None:
            training_kwargs["learning_starts"] = args.learning_starts
        if args.train_freq is not None:
            training_kwargs["train_freq"] = args.train_freq
        if args.gradient_steps is not None:
            training_kwargs["gradient_steps"] = args.gradient_steps
        if args.tau is not None:
            training_kwargs["tau"] = args.tau
    if args.l2_reg is not None:
        extra_defaults["l2_reg"] = args.l2_reg
    return training_kwargs, extra_defaults


def normalize_dropout(dropout: Optional[float]) -> Optional[float]:
    """Normalize and validate dropout probability."""
    if dropout is None:
        return None
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("--dropout must be in the range [0, 1).")
    if dropout == 0.0:
        return None
    return float(dropout)


def main():
    """Main training function."""
    args = parse_args()

    if args.config_list and args.config:
        raise ValueError("--config and --config-list are mutually exclusive.")

    config_list_paths = parse_config_list_arg(args.config_list)
    if config_list_paths:
        configs = load_config_list(config_list_paths)
        config_source = "config-list: " + ", ".join(config_list_paths)
    else:
        config, config_source = resolve_env_config(args.config)
        configs = [config]

    # Apply CLI overrides to configs
    for cfg in configs:
        # CNN flag (deprecated): force grid if requested
        if args.cnn and cfg.obs_type != "grid":
            print("Note: forcing obs_type='grid' due to --cnn.")
            cfg.obs_type = "grid"

    config = configs[0]

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_{timestamp}"

    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Cohort spawn: override random_start for PPO
    use_cohort_spawn = args.algo == "ppo" and not args.no_cohort_spawn
    if args.algo == "sac":
        if args.cohort_spawn or args.no_cohort_spawn:
            print("Warning: cohort-spawn flags are ignored for SAC (SAC handles random starts natively)")
        use_cohort_spawn = False

    if use_cohort_spawn:
        checkpoint_counts = {get_num_checkpoints(cfg) for cfg in configs}
        if len(checkpoint_counts) > 1:
            print("Warning: cohort spawn disabled (configs have different num_checkpoints).")
            use_cohort_spawn = False

    if use_cohort_spawn:
        # Disable per-env random_start since the callback handles it
        for cfg in configs:
            cfg.random_start = False
    elif args.random_start:
        for cfg in configs:
            cfg.random_start = True

    multi_track = len(configs) > 1

    print(f"Training {args.algo.upper()} for {args.total_timesteps} timesteps")
    print(f"Using {args.n_envs} parallel environments")
    print(f"Config: {config_source}")
    if multi_track:
        print(f"Multi-track: enabled ({args.multi_track_mode}, {len(configs)} configs)")
    print(f"Obs type: {config.obs_type}")
    print(f"Random start: {config.random_start}")
    if use_cohort_spawn:
        if args.cohort_checkpoint is None:
            print("Cohort spawn: enabled (synchronized random starts per rollout)")
        else:
            print(f"Cohort spawn: enabled (fixed checkpoint={args.cohort_checkpoint})")
    elif args.algo == "ppo":
        print("Cohort spawn: disabled (using config.random_start)")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")

    # Create vectorized environment
    env_fns = [
        make_env(
            configs,
            i,
            args.seed,
            multi_track=multi_track,
            multi_track_mode=args.multi_track_mode,
        )
        for i in range(args.n_envs)
    ]
    env = make_vec_env(env_fns, args.vec_env)

    # Wrap with VecNormalize for reward normalization (helps SAC stability)
    if args.normalize:
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
        print("VecNormalize enabled (norm_reward=True, norm_obs=False)")

    # Create evaluation environment
    eval_env = None
    if not args.no_eval:
        eval_env = make_vec_env([
            make_env(
                configs,
                0,
                args.seed + 1000,
                multi_track=multi_track,
                multi_track_mode=args.multi_track_mode,
            )
        ], "dummy")
        # If training env uses VecNormalize, eval env must also be wrapped for compatibility
        if args.normalize:
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99, training=False)
        # If using CNN (image obs), wrap eval env in VecTransposeImage to match training env
        if config.obs_type == "grid":
            from stable_baselines3.common.vec_env import VecTransposeImage
            eval_env = VecTransposeImage(eval_env)

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
        num_checkpoints = get_num_checkpoints(configs[0])
        cohort_callback = make_cohort_spawn_callback(
            num_checkpoints=num_checkpoints,
            fixed_checkpoint=args.cohort_checkpoint,
        )
        callbacks.append(cohort_callback)

    # Add Shrink+Perturb callback for plasticity restoration
    if args.shrink_perturb:
        from racing_sim.callbacks.plasticity import ShrinkPerturbCallback
        shrink_callback = ShrinkPerturbCallback(
            interval=args.shrink_interval,
            verbose=1,
        )
        callbacks.append(shrink_callback)
        print(f"Shrink+Perturb enabled (interval={args.shrink_interval})")

    # Log_std clamp callback (PPO only)
    if args.algo == "ppo" and (args.log_std_min is not None or args.log_std_max is not None):
        if args.log_std_min is None or args.log_std_max is None:
            raise ValueError("--log-std-min and --log-std-max must be set together")
        from racing_sim.callbacks.log_std_clamp import LogStdClampCallback
        clamp_callback = LogStdClampCallback(args.log_std_min, args.log_std_max)
        callbacks.append(clamp_callback)
        print(f"log_std clamped to [{args.log_std_min}, {args.log_std_max}]")

    # Grad/update norm logging
    if args.grad_log_freq and args.grad_log_freq > 0:
        from racing_sim.callbacks.grad_stats import GradStatsCallback
        callbacks.append(GradStatsCallback(log_freq=args.grad_log_freq))
        print(f"Grad/update norm logging enabled (freq={args.grad_log_freq})")

    if args.rollout_log_freq and args.rollout_log_freq > 0:
        from racing_sim.callbacks.rollout_stats import RolloutStatsCallback
        callbacks.append(RolloutStatsCallback(log_freq=args.rollout_log_freq))
        print(f"Rollout stats logging enabled (freq={args.rollout_log_freq})")

    callback = None
    if callbacks:
        from stable_baselines3.common.callbacks import CallbackList

        callback = CallbackList(callbacks)

    # Create model
    tensorboard_log = None if args.no_tensorboard else str(log_dir)
    training_defaults, _ = load_training_presets()
    training_kwargs, extra_defaults = build_training_kwargs(args, training_defaults)
    if "l2_reg" in extra_defaults:
        args.l2_reg = extra_defaults["l2_reg"]

    args.dropout = normalize_dropout(args.dropout)

    # Determine policy class
    if args.layernorm and config.obs_type == "grid":
        raise ValueError("--layernorm is not compatible with grid observations")

    # Build policy_kwargs for optimizer customization
    policy_kwargs = {}
    optimizer_kwargs = {}

    if args.l2_reg is not None:
        optimizer_kwargs["weight_decay"] = args.l2_reg
        print(f"L2 regularization enabled (weight_decay={args.l2_reg})")

    if args.adam_betas is not None:
        optimizer_kwargs["betas"] = tuple(args.adam_betas)
        print(f"Adam betas set to {tuple(args.adam_betas)}")

    if optimizer_kwargs:
        policy_kwargs["optimizer_kwargs"] = optimizer_kwargs

    if args.dropout is not None:
        policy_kwargs["dropout"] = args.dropout
        print(f"Dropout enabled (p={args.dropout})")

    policy = "CnnPolicy" if config.obs_type == "grid" else "MlpPolicy"
    if args.dropout is not None:
        from racing_sim.policies.dropout_policy import DropoutActorCriticPolicy, DropoutSACPolicy

        policy = DropoutActorCriticPolicy if args.algo == "ppo" else DropoutSACPolicy
        if config.obs_type == "grid":
            from stable_baselines3.common.torch_layers import NatureCNN

            policy_kwargs["features_extractor_class"] = NatureCNN

    if args.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            policy=policy,
            env=env,
            **training_kwargs,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            device=args.device,
        )
    else:  # SAC
        from stable_baselines3 import SAC

        model = SAC(
            policy=policy,
            env=env,
            **training_kwargs,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=args.seed,
            device=args.device,
        )

    # Freeze early CNN layers for transfer learning (grid obs only)
    if args.freeze_cnn_layers and args.freeze_cnn_layers > 0:
        if config.obs_type != "grid":
            raise ValueError("--freeze-cnn-layers requires grid observations (use --cnn or grid config)")
        from racing_sim.utils.training_utils import freeze_cnn_layers
        frozen = freeze_cnn_layers(model.policy, args.freeze_cnn_layers)
        print(f"Frozen {len(frozen)} CNN params across extractors (first {args.freeze_cnn_layers} layers)")

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
