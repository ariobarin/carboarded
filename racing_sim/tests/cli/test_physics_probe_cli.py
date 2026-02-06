import importlib


def import_probe_module():
    return importlib.import_module("scripts.physics_probe")


def test_physics_probe_module_imports():
    probe = import_probe_module()
    assert hasattr(probe, "build_arg_parser")


def test_physics_probe_arg_parser_defaults():
    probe = import_probe_module()
    parser = probe.build_arg_parser()
    args = parser.parse_args([])

    assert args.config is None
    assert args.output is None
    assert args.scenario == "straight_full_throttle"
    assert args.steps == 600
