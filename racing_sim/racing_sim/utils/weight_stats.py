"""Utilities for summarizing model weight tensors."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def summarize_tensors(tensors: Dict[str, torch.Tensor]) -> List[Dict[str, object]]:
    """Summarize tensors with basic distribution statistics.

    Args:
        tensors: Mapping of parameter name -> tensor.

    Returns:
        List of per-parameter stats dicts.
    """
    rows: List[Dict[str, object]] = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        flat = tensor.detach().float().reshape(-1)
        numel = int(flat.numel())
        if numel == 0:
            stats = {
                "name": name,
                "shape": tuple(tensor.shape),
                "numel": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "max_abs": 0.0,
                "l2_norm": 0.0,
                "l1_norm": 0.0,
                "zeros_frac": 0.0,
                "p95_abs": 0.0,
                "p99_abs": 0.0,
                "is_bias": name.endswith("bias") or name.endswith(".bias"),
            }
            rows.append(stats)
            continue

        abs_flat = flat.abs()
        mean = _safe_float(flat.mean())
        std = _safe_float(flat.std(unbiased=False))
        min_v = _safe_float(flat.min())
        max_v = _safe_float(flat.max())
        max_abs = _safe_float(abs_flat.max())
        l2_norm = _safe_float(flat.norm(2))
        l1_norm = _safe_float(abs_flat.sum())
        zeros_frac = _safe_float((flat == 0).float().mean())
        p95_abs = _safe_float(torch.quantile(abs_flat, 0.95))
        p99_abs = _safe_float(torch.quantile(abs_flat, 0.99))

        stats = {
            "name": name,
            "shape": tuple(tensor.shape),
            "numel": numel,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "max_abs": max_abs,
            "l2_norm": l2_norm,
            "l1_norm": l1_norm,
            "zeros_frac": zeros_frac,
            "p95_abs": p95_abs,
            "p99_abs": p99_abs,
            "is_bias": name.endswith("bias") or name.endswith(".bias"),
        }
        rows.append(stats)

    return rows


def aggregate_stats(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate stats across rows (weights or biases)."""
    total_numel = 0
    sum_l2_sq = 0.0
    max_abs = 0.0
    mean_acc = 0.0
    std_acc = 0.0
    zeros_acc = 0.0
    p95_acc = 0.0
    p99_acc = 0.0
    count = 0

    for row in rows:
        numel = int(row["numel"])
        if numel == 0:
            continue
        total_numel += numel
        l2_norm = float(row["l2_norm"])
        sum_l2_sq += l2_norm * l2_norm
        max_abs = max(max_abs, float(row["max_abs"]))
        mean_acc += float(row["mean"])
        std_acc += float(row["std"])
        zeros_acc += float(row["zeros_frac"])
        p95_acc += float(row["p95_abs"])
        p99_acc += float(row["p99_abs"])
        count += 1

    if count == 0:
        return {
            "numel": 0,
            "l2_norm": 0.0,
            "max_abs": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "zeros_frac": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
        }

    return {
        "numel": total_numel,
        "l2_norm": sum_l2_sq ** 0.5,
        "max_abs": max_abs,
        "mean": mean_acc / count,
        "std": std_acc / count,
        "zeros_frac": zeros_acc / count,
        "p95_abs": p95_acc / count,
        "p99_abs": p99_acc / count,
    }
