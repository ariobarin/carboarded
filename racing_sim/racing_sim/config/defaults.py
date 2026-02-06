"""Shared default config loaders to avoid duplicated sources of truth."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from racing_sim.config.config import EnvConfig


def _repo_root() -> Path:
    # defaults.py lives at racing_sim/racing_sim/config/defaults.py
    # repo root is two parents up from racing_sim/racing_sim
    return Path(__file__).resolve().parents[2]


def default_env_config_path() -> Path:
    """Return the canonical default env config path."""
    return _repo_root() / "configs" / "default.yaml"


def default_training_presets_path() -> Path:
    """Return the canonical training defaults path."""
    return _repo_root() / "configs" / "training_presets.yaml"


def load_default_env_config() -> EnvConfig:
    """Load the canonical default env config from YAML."""
    path = default_env_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Default env config not found: {path}")
    return EnvConfig.from_yaml(str(path))


def resolve_env_config(path_str: Optional[str]) -> Tuple[EnvConfig, str]:
    """Load env config from path or canonical default YAML."""
    if path_str:
        return EnvConfig.from_yaml(path_str), path_str
    default_path = default_env_config_path()
    return load_default_env_config(), str(default_path)


def load_training_presets(path_str: Optional[str] = None) -> Tuple[Dict, str]:
    """Load training defaults from YAML."""
    path = Path(path_str) if path_str else default_training_presets_path()
    if not path.exists():
        raise FileNotFoundError(f"Training presets not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data, str(path)
