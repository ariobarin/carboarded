import importlib


def import_tuner_module():
    return importlib.import_module("scripts.physics_tuner")


def test_physics_tuner_module_imports():
    tuner = import_tuner_module()
    assert hasattr(tuner, "build_arg_parser")


def test_physics_tuner_arg_parser_defaults():
    tuner = import_tuner_module()
    parser = tuner.build_arg_parser()
    args = parser.parse_args([])

    assert args.config is None
    assert args.fps is None
