import importlib


def import_train_module():
    return importlib.import_module("scripts.train")


def test_train_module_imports_without_optional_deps():
    train = import_train_module()
    assert hasattr(train, "build_arg_parser")


def test_train_arg_parser_defaults():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args([])

    assert args.algo == "ppo"
    assert args.preset == "fast"
    assert args.total_timesteps == 200000
    assert args.n_envs == 1
    assert args.vec_env == "dummy"
    assert args.eval_episodes == 2
    assert args.device == "auto"
    assert args.learning_rate is None
    assert args.clip_range is None
    assert args.gamma is None
    assert args.gae_lambda is None
    assert args.ent_coef is None
    assert args.target_entropy is None
    assert args.batch_size is None
    assert args.buffer_size is None
    assert args.learning_starts is None
    assert args.train_freq is None
    assert args.gradient_steps is None
    assert args.no_eval is False
    assert args.no_checkpoint is False
    assert args.no_tensorboard is False
    assert args.no_progress is False


def test_train_presets_structure():
    train = import_train_module()

    assert set(train.PRESETS.keys()) == {"ppo", "sac"}
    assert set(train.PRESETS["ppo"].keys()) == {"fast", "balanced", "quality"}
    assert set(train.PRESETS["sac"].keys()) == {"fast", "balanced", "quality"}

    assert train.PRESETS["ppo"]["fast"]["n_steps"] < train.PRESETS["ppo"]["quality"]["n_steps"]
    assert train.PRESETS["sac"]["fast"]["buffer_size"] < train.PRESETS["sac"]["quality"]["buffer_size"]


def test_resolve_env_config_defaults_to_yaml():
    train = import_train_module()
    config, source = train.resolve_env_config(None)

    assert "configs" in source
    assert config.car.max_speed == 1000.0


def test_train_arg_parser_overrides_entropy_settings():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--ent-coef", "0.2", "--target-entropy", "-1.0"])

    assert args.ent_coef == "0.2"
    assert args.target_entropy == -1.0


def test_build_preset_kwargs_overrides_clip_range_for_ppo():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--algo", "ppo", "--preset", "fast", "--clip-range", "0.1"])

    preset_kwargs = train.build_preset_kwargs(args)

    assert preset_kwargs["clip_range"] == 0.1


def test_build_preset_kwargs_overrides_gamma_and_gae_lambda_for_ppo():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--algo", "ppo", "--preset", "fast", "--gamma", "0.95", "--gae-lambda", "0.9"])

    preset_kwargs = train.build_preset_kwargs(args)

    assert preset_kwargs["gamma"] == 0.95
    assert preset_kwargs["gae_lambda"] == 0.9


def test_build_preset_kwargs_overrides_sac_training_params():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args([
        "--algo", "sac",
        "--preset", "fast",
        "--batch-size", "256",
        "--buffer-size", "200000",
        "--learning-starts", "500",
        "--train-freq", "4",
        "--gradient-steps", "2",
    ])

    preset_kwargs = train.build_preset_kwargs(args)

    assert preset_kwargs["batch_size"] == 256
    assert preset_kwargs["buffer_size"] == 200000
    assert preset_kwargs["learning_starts"] == 500
    assert preset_kwargs["train_freq"] == 4
    assert preset_kwargs["gradient_steps"] == 2
