import importlib

import pytest


def test_package_import_does_not_require_optional_deps():
    try:
        module = importlib.import_module("racing_sim")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Importing racing_sim should not require optional deps: {exc}")

    assert "RacingEnv" in getattr(module, "__all__", [])
