"""Training utilities for policy manipulation and diagnostics."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


def clamp_log_std(policy: nn.Module, min_val: float, max_val: float) -> None:
    """Clamp policy log_std in-place if present."""
    if not hasattr(policy, "log_std"):
        return
    with torch.no_grad():
        policy.log_std.data.clamp_(min_val, max_val)


def freeze_cnn_layers(policy: nn.Module, num_layers: int) -> List[str]:
    """Freeze the first N parameterized CNN layers across extractors."""
    if num_layers <= 0:
        return []

    frozen_names: List[str] = []
    for extractor_name in ("features_extractor", "pi_features_extractor", "vf_features_extractor"):
        extractor = getattr(policy, extractor_name, None)
        if extractor is None or not hasattr(extractor, "cnn"):
            continue
        cnn = extractor.cnn
        frozen = 0
        for layer_idx, layer in enumerate(cnn):
            params = list(layer.named_parameters(recurse=False))
            if not params:
                continue
            if frozen >= num_layers:
                break
            for name, param in params:
                param.requires_grad = False
                frozen_names.append(f"{extractor_name}.cnn.{layer_idx}.{name}")
            frozen += 1

    return frozen_names


def compute_grad_norm(module: nn.Module) -> float:
    """Compute global L2 norm of gradients for a module."""
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def compute_update_norm(prev_params: Iterable[torch.Tensor], curr_params: Iterable[torch.Tensor]) -> float:
    """Compute L2 norm of parameter updates between two snapshots."""
    total = 0.0
    for prev, curr in zip(prev_params, curr_params):
        diff = (curr - prev).detach().float()
        total += float(diff.pow(2).sum().item())
    return total ** 0.5
