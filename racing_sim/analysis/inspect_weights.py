"""Inspect and plot weight statistics for SB3 models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from racing_sim.utils.model import load_model
from racing_sim.utils.weight_stats import aggregate_stats, summarize_tensors


def parse_models(raw_models: List[str]) -> Dict[str, Path]:
    models: Dict[str, Path] = {}
    for raw in raw_models:
        if "=" not in raw:
            raise ValueError(f"Model must be in name=path format: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = Path(path.strip())
        models[name] = path
    return models


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histograms(out_dir: Path, name: str, weights: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"Plotting skipped ({exc})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(weights, bins=120, color="#2e6f9e", alpha=0.85)
    ax.set_title(f"{name} weight histogram")
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_hist.png", dpi=140)
    plt.close(fig)

    # Log-abs histogram for tails
    log_abs = np.log10(np.abs(weights) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(log_abs, bins=120, color="#7a9a01", alpha=0.85)
    ax.set_title(f"{name} log10(abs(weight))")
    ax.set_xlabel("log10(abs(weight))")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_logabs_hist.png", dpi=140)
    plt.close(fig)


def plot_layer_bars(out_dir: Path, name: str, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"Plotting skipped ({exc})")
        return

    layer_names = [r["name"] for r in rows]
    l2 = [r["l2_norm"] for r in rows]
    max_abs = [r["max_abs"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(layer_names, l2, color="#1b9e77")
    ax.set_title(f"{name} layer L2 norms")
    ax.set_xlabel("L2 norm")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_layer_l2.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(layer_names, max_abs, color="#d95f02")
    ax.set_title(f"{name} layer max abs")
    ax.set_xlabel("Max abs")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_layer_maxabs.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SB3 model weights")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model in name=path format. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Learnings/figures/custom_track_weights",
        help="Output directory for plots and CSVs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    models = parse_models(args.model)

    summary_rows: List[Dict[str, object]] = []
    for name, path in models.items():
        if not path.exists():
            print(f"Missing model: {path}")
            continue

        model = load_model(path, algo="auto", device="cpu")
        state = model.policy.state_dict()
        rows = summarize_tensors(state)

        # Split weight/bias for aggregation
        weight_rows = [r for r in rows if not r["is_bias"]]
        bias_rows = [r for r in rows if r["is_bias"]]

        weight_agg = aggregate_stats(weight_rows)
        bias_agg = aggregate_stats(bias_rows)

        summary_rows.append({
            "model": name,
            "path": str(path),
            "weights_numel": weight_agg["numel"],
            "weights_l2_norm": weight_agg["l2_norm"],
            "weights_max_abs": weight_agg["max_abs"],
            "weights_mean": weight_agg["mean"],
            "weights_std": weight_agg["std"],
            "weights_zeros_frac": weight_agg["zeros_frac"],
            "weights_p95_abs": weight_agg["p95_abs"],
            "weights_p99_abs": weight_agg["p99_abs"],
            "bias_numel": bias_agg["numel"],
            "bias_l2_norm": bias_agg["l2_norm"],
            "bias_max_abs": bias_agg["max_abs"],
        })

        # Save per-model parameter stats
        save_csv(out_dir / f"{name}_params.csv", rows)

        # Plot per-model histograms (weights only)
        weight_values = []
        for r in weight_rows:
            tensor = state[r["name"]].detach().float().reshape(-1)
            weight_values.append(tensor.cpu().numpy())
        if weight_values:
            weights = np.concatenate(weight_values, axis=0)
            plot_histograms(out_dir, name, weights)

        # Plot layer bars for top 30 params by max_abs
        top_rows = sorted(weight_rows, key=lambda r: r["max_abs"], reverse=True)[:30]
        plot_layer_bars(out_dir, f"{name}_top30", top_rows)

    # Save summary CSV
    if summary_rows:
        save_csv(out_dir / "summary.csv", summary_rows)

        # Comparative plot: overall max_abs and l2_norm
        try:
            import matplotlib.pyplot as plt
            names = [r["model"] for r in summary_rows]
            l2 = [r["weights_l2_norm"] for r in summary_rows]
            max_abs = [r["weights_max_abs"] for r in summary_rows]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(names, l2, color="#4c78a8")
            ax.set_title("Overall weight L2 norm by model")
            ax.set_ylabel("L2 norm")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(out_dir / "overall_l2.png", dpi=140)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(names, max_abs, color="#f58518")
            ax.set_title("Overall max abs weight by model")
            ax.set_ylabel("Max abs")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(out_dir / "overall_max_abs.png", dpi=140)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - optional plotting
            print(f"Plotting skipped ({exc})")


if __name__ == "__main__":
    main()
