import importlib


def import_sweep_module():
    return importlib.import_module("scripts.physics_sweep")


def test_physics_sweep_module_imports():
    sweep = import_sweep_module()
    assert hasattr(sweep, "build_arg_parser")


def test_physics_sweep_arg_parser_defaults():
    sweep = import_sweep_module()
    parser = sweep.build_arg_parser()
    args = parser.parse_args([])

    assert args.configs is None
    assert args.output is None
    assert args.steps == 600
