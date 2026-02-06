"""Model loading utilities shared across scripts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from stable_baselines3 import PPO, SAC


def detect_algo_from_model(model_path: Path) -> str:
    """Auto-detect algorithm type from saved model file.

    Inspects the model's zip file to determine if it was trained with PPO or SAC.

    Args:
        model_path: Path to the saved model .zip file.

    Returns:
        Algorithm name: 'ppo' or 'sac'. Defaults to 'ppo' if detection fails.
    """
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "data" in zf.namelist():
                with zf.open("data") as f:
                    data = json.load(f)
                    policy_class = data.get("policy_class", "")
                    if "SAC" in str(policy_class) or "sac" in str(policy_class).lower():
                        return "sac"
                    if "PPO" in str(policy_class) or "ppo" in str(policy_class).lower():
                        return "ppo"
    except Exception:
        pass
    return "ppo"


def infer_obs_type(model: Union[PPO, SAC]) -> Optional[str]:
    """Infer observation type from a loaded model's observation space.

    Returns:
        'lidar' for flat vector obs, 'grid' for image obs, or None on failure.
    """
    try:
        shape = model.observation_space.shape
        if len(shape) == 1:
            return "lidar"
        if len(shape) >= 2:
            return "grid"
    except Exception:
        pass
    return None


def load_model(
    model_path: Union[str, Path],
    algo: str = "auto",
    device: str = "auto",
) -> Union[PPO, SAC]:
    """Load a trained model from disk.

    Args:
        model_path: Path to the saved model .zip file.
        algo: Algorithm type ('ppo', 'sac', or 'auto' to detect).
        device: PyTorch device to load model on.

    Returns:
        Loaded PPO or SAC model instance.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If algo is invalid.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    if algo == "auto":
        algo = detect_algo_from_model(model_path)

    from stable_baselines3 import PPO, SAC

    if algo == "ppo":
        return PPO.load(str(model_path), device=device)
    elif algo == "sac":
        return SAC.load(str(model_path), device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'ppo', 'sac', or 'auto'.")
