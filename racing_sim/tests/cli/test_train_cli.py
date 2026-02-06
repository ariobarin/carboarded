import importlib

import pytest


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
    assert args.total_timesteps == 200000
    assert args.n_envs == 1
    assert args.vec_env == "dummy"
    assert args.eval_episodes == 1
    assert args.device == "auto"
    assert args.learning_rate is None
    assert args.clip_range is None
    assert args.gamma is None
    assert args.gae_lambda is None
    assert args.ent_coef is None
    assert args.target_entropy is None
    assert args.batch_size is None
    assert args.ppo_batch_size is None
    assert args.buffer_size is None
    assert args.learning_starts is None
    assert args.train_freq is None
    assert args.gradient_steps is None
    assert args.no_eval is False
    assert args.no_checkpoint is False
    assert args.no_tensorboard is False
    assert args.no_progress is False
    assert args.cohort_spawn is False
    assert args.no_cohort_spawn is False
    assert args.cohort_checkpoint is None
    assert args.freeze_cnn_layers == 0
    assert args.grad_log_freq == 0
    assert args.rollout_log_freq == 0
    assert args.log_std_min is None
    assert args.log_std_max is None
    assert args.n_steps is None
    assert args.dropout is None


def test_training_defaults_structure():
    from racing_sim.config.defaults import load_training_presets

    presets, _ = load_training_presets()

    assert set(presets.keys()) == {"ppo", "sac"}
    assert "n_steps" in presets["ppo"]
    assert "buffer_size" in presets["sac"]


def test_resolve_env_config_defaults_to_yaml():
    from racing_sim.config.defaults import resolve_env_config
    config, source = resolve_env_config(None)

    assert "configs" in source
    assert "default.yaml" in source
    assert config.car.max_speed == 360.0


def test_train_arg_parser_overrides_entropy_settings():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--ent-coef", "0.2", "--target-entropy", "-1.0"])

    assert args.ent_coef == "0.2"
    assert args.target_entropy == -1.0


def test_train_arg_parser_disables_cohort_spawn():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--no-cohort-spawn"])

    assert args.no_cohort_spawn is True


def test_train_arg_parser_sets_cohort_checkpoint():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--cohort-checkpoint", "3"])

    assert args.cohort_checkpoint == 3


def test_build_training_kwargs_overrides_clip_range_for_ppo():
    train = import_train_module()
    from racing_sim.config.defaults import load_training_presets
    parser = train.build_arg_parser()
    args = parser.parse_args(["--algo", "ppo", "--clip-range", "0.1"])

    presets, _ = load_training_presets()
    training_kwargs, _ = train.build_training_kwargs(args, presets)

    assert training_kwargs["clip_range"] == 0.1


def test_build_training_kwargs_overrides_gamma_and_gae_lambda_for_ppo():
    train = import_train_module()
    from racing_sim.config.defaults import load_training_presets
    parser = train.build_arg_parser()
    args = parser.parse_args(["--algo", "ppo", "--gamma", "0.95", "--gae-lambda", "0.9"])

    presets, _ = load_training_presets()
    training_kwargs, _ = train.build_training_kwargs(args, presets)

    assert training_kwargs["gamma"] == 0.95
    assert training_kwargs["gae_lambda"] == 0.9


def test_build_training_kwargs_overrides_ppo_rollout_settings():
    train = import_train_module()
    from racing_sim.config.defaults import load_training_presets
    parser = train.build_arg_parser()
    args = parser.parse_args([
        "--algo", "ppo",
        "--n-steps", "256",
        "--ppo-batch-size", "128",
    ])

    presets, _ = load_training_presets()
    training_kwargs, _ = train.build_training_kwargs(args, presets)

    assert training_kwargs["n_steps"] == 256
    assert training_kwargs["batch_size"] == 128


def test_build_training_kwargs_overrides_sac_training_params():
    train = import_train_module()
    from racing_sim.config.defaults import load_training_presets
    parser = train.build_arg_parser()
    args = parser.parse_args([
        "--algo", "sac",
        "--batch-size", "256",
        "--buffer-size", "200000",
        "--learning-starts", "500",
        "--train-freq", "4",
        "--gradient-steps", "2",
    ])

    presets, _ = load_training_presets()
    training_kwargs, _ = train.build_training_kwargs(args, presets)

    assert training_kwargs["batch_size"] == 256
    assert training_kwargs["buffer_size"] == 200000
    assert training_kwargs["learning_starts"] == 500
    assert training_kwargs["train_freq"] == 4
    assert training_kwargs["gradient_steps"] == 2


def test_train_arg_parser_accepts_dropout():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--dropout", "0.2"])

    assert args.dropout == 0.2


def test_train_arg_parser_accepts_config_list():
    train = import_train_module()
    parser = train.build_arg_parser()
    args = parser.parse_args(["--config-list", "a.yaml,b.yaml", "--multi-track-mode", "random"])

    assert args.config_list == "a.yaml,b.yaml"
    assert args.multi_track_mode == "random"


def test_normalize_dropout_validates_bounds():
    train = import_train_module()

    assert train.normalize_dropout(None) is None
    assert train.normalize_dropout(0.0) is None
    assert train.normalize_dropout(0.2) == 0.2

    with pytest.raises(ValueError):
        train.normalize_dropout(-0.1)

    with pytest.raises(ValueError):
        train.normalize_dropout(1.0)


