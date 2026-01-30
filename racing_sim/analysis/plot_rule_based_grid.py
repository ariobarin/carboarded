from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def heatmap_for_tag(tag: str):
    df = pd.read_csv(RESULTS_DIR / f"rule_based_grid_{tag}.csv")
    pivot = df.pivot_table(
        index="steer_gain",
        columns="throttle_gain",
        values="mean_reward",
        aggfunc="max",
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="viridis")
    ax.set_title(f"Rule-Based Grid (max reward) - {tag}")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("throttle_gain")
    ax.set_ylabel("steer_gain")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean Reward (max)")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"rule_based_heatmap_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot rule-based grid heatmaps.")
    parser.add_argument("--tags", nargs="+", default=["fast_iter_v3", "default"])
    args = parser.parse_args()

    for tag in args.tags:
        heatmap_for_tag(tag)


if __name__ == "__main__":
    main()
