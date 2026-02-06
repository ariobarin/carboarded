from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PLOTS = Path(__file__).resolve().parent / "plots"


def load_eval(log_dir: Path):
    data = np.load(log_dir / "evaluations.npz")
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    return timesteps, mean_rewards


def plot_series(ax, label, log_dir):
    timesteps, mean_rewards = load_eval(log_dir)
    ax.plot(timesteps, mean_rewards, marker="o", linewidth=2, label=label)


def make_plot(title, series, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, log_dir in series:
        plot_series(ax, label, log_dir)
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Eval Reward")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    default_series = [
        ("ppo_fast_default", LOGS / "ppo_fast_20260126_222844"),
        ("ppo_balanced_default", LOGS / "ppo_balanced_20260126_223017"),
        ("sac_fast_default", LOGS / "sac_fast_20260126_223152"),
    ]
    fast_iter_series = [
        ("ppo_fast_iter_v1", LOGS / "ppo_fast_20260126_224338"),
        ("ppo_fast_iter_v2", LOGS / "ppo_fast_20260126_224446"),
        ("ppo_fast_iter_v3", LOGS / "ppo_fast_20260126_225213"),
        ("ppo_fast_iter_v4", LOGS / "ppo_fast_20260126_225035"),
    ]
    v3_preset_series = [
        ("ppo_fast_v3", LOGS / "ppo_fast_20260126_225213"),
        ("ppo_balanced_v3", LOGS / "ppo_balanced_20260126_225340"),
    ]
    v3_long_series = [
        ("ppo_fast_v3_200k", LOGS / "ppo_fast_20260126_225942"),
        ("ppo_balanced_v3_200k", LOGS / "ppo_balanced_20260126_230311"),
        ("ppo_fast_v3_lr1e-4_partial", LOGS / "ppo_fast_20260126_230822"),
    ]
    v3_fast_long_series = [
        ("ppo_fast_v3_200k", LOGS / "ppo_fast_20260126_225942"),
        ("ppo_fast_v3_400k", LOGS / "ppo_fast_20260126_231105"),
    ]

    make_plot(
        "Default Config: PPO Presets",
        default_series,
        PLOTS / "default_config_eval.png",
    )
    make_plot(
        "Fast-Iter Config Sweep (Reward-Scale Not Comparable to Default)",
        fast_iter_series,
        PLOTS / "fast_iter_eval.png",
    )
    make_plot(
        "Fast-Iter v3: PPO Preset Comparison",
        v3_preset_series,
        PLOTS / "fast_iter_v3_presets.png",
    )
    make_plot(
        "Fast-Iter v3: Long-Run Comparison",
        v3_long_series,
        PLOTS / "fast_iter_v3_long.png",
    )
    make_plot(
        "Fast-Iter v3: PPO Fast 200k vs 400k",
        v3_fast_long_series,
        PLOTS / "fast_iter_v3_fast_200k_400k.png",
    )


if __name__ == "__main__":
    main()
