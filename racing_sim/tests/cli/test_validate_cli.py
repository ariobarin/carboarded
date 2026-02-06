import importlib


def import_validate_module():
    return importlib.import_module("scripts.validate")


def test_validate_arg_parser_defaults():
    validate = import_validate_module()
    parser = validate.build_arg_parser()
    args = parser.parse_args(["--model", "dummy.zip"])

    assert args.deterministic is False


def test_validate_arg_parser_deterministic_flag():
    validate = import_validate_module()
    parser = validate.build_arg_parser()
    args = parser.parse_args(["--model", "dummy.zip", "--deterministic"])

    assert args.deterministic is True
