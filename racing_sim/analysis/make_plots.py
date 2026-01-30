from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def bar_plot(df, x, y, err=None, title="", ylabel="", filename="plot.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(df))
    ax.bar(positions, df[y], yerr=df[err] if err else None, capsize=4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(df[x], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def box_plot(episodes: pd.DataFrame, value: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    data = []
    for name, group in episodes.groupby("name"):
        labels.append(name)
        data.append(group[value].values)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(value)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def scatter_plot(episodes: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, group in episodes.groupby("name"):
        ax.scatter(group["checkpoints"], group["reward"], alpha=0.5, label=name, s=20)
    ax.set_title(title)
    ax.set_xlabel("Checkpoints Passed")
    ax.set_ylabel("Episode Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation plots.")
    parser.add_argument("--tag", type=str, default="fast_iter_v3")
    args = parser.parse_args()

    summary = pd.read_csv(RESULTS_DIR / f"eval_summary_{args.tag}.csv")
    episodes = pd.read_csv(RESULTS_DIR / f"eval_episodes_{args.tag}.csv")

    bar_plot(
        summary,
        x="name",
        y="mean_reward",
        err="std_reward",
        title="Mean Reward (50 Episodes)",
        ylabel="Mean Reward",
        filename=f"eval_mean_reward_bar_{args.tag}.png",
    )
    bar_plot(
        summary,
        x="name",
        y="mean_checkpoints",
        title="Mean Checkpoints",
        ylabel="Checkpoints",
        filename=f"eval_checkpoints_bar_{args.tag}.png",
    )
    bar_plot(
        summary,
        x="name",
        y="collision_rate",
        title="Collision Rate",
        ylabel="Collision Rate",
        filename=f"eval_collision_rate_bar_{args.tag}.png",
    )
    bar_plot(
        summary,
        x="name",
        y="mean_steps",
        title="Mean Episode Length",
        ylabel="Steps",
        filename=f"eval_steps_bar_{args.tag}.png",
    )
    bar_plot(
        summary,
        x="name",
        y="mean_speed",
        title="Mean Speed",
        ylabel="Speed",
        filename=f"eval_speed_bar_{args.tag}.png",
    )

    box_plot(
        episodes,
        value="reward",
        title="Reward Distribution",
        filename=f"eval_reward_boxplot_{args.tag}.png",
    )
    scatter_plot(
        episodes,
        title="Reward vs Checkpoints",
        filename=f"eval_reward_vs_checkpoints_{args.tag}.png",
    )


if __name__ == "__main__":
    main()
