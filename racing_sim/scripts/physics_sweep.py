"""Run physics probes across multiple configs and summarize results."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import physics_probe
from racing_sim.config.config import EnvConfig
from racing_sim.config.defaults import default_env_config_path


@dataclass
class SweepResult:
    config_path: str
    scenario: str
    max_speed_target: float
    max_speed_actual: float
    time_to_25: Optional[float]
    time_to_50: Optional[float]
    time_to_75: Optional[float]
    avg_forward_accel_1s: Optional[float]
    avg_speed_last_1s: Optional[float]
    avg_speed_pre_turn: Optional[float]
    avg_speed_post_turn: Optional[float]


TURN_SCENARIOS = {"turn_after_full", "turn_constant"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run physics probe sweeps")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Config YAML paths to sweep (defaults to configs/default.yaml)",
    )
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    parser.add_argument("--scenario", type=str, default="straight_full_throttle",
                        choices=sorted(physics_probe.SCENARIOS),
                        help="Scenario to run (use 'all' to run key scenarios)")
    parser.add_argument("--steps", type=int, default=600, help="Steps per scenario")
    parser.add_argument("--warmup-steps", type=int, default=120, help="Warmup steps")
    parser.add_argument("--turn-steer", type=float, default=0.5, help="Steer for turns")
    parser.add_argument("--turn-throttle", type=float, default=1.0, help="Throttle for turns")
    return parser


def _resolve_configs(configs: Optional[Sequence[str]]) -> List[str]:
    if configs:
        return list(configs)
    return [str(default_env_config_path())]


def _time_to_threshold(samples: Sequence[physics_probe.ProbeSample], threshold: float) -> Optional[float]:
    for sample in samples:
        if sample.speed >= threshold:
            return sample.time_s
    return None


def _avg_window(values: Sequence[float], window: int, start: Optional[int] = None, end: Optional[int] = None) -> Optional[float]:
    if not values:
        return None
    start_idx = 0 if start is None else max(0, start)
    end_idx = len(values) if end is None else min(len(values), end)
    if end_idx <= start_idx:
        return None
    window = min(window, end_idx - start_idx)
    segment = values[end_idx - window:end_idx]
    return float(np.mean(segment)) if segment else None


def _summarize(
    samples: Sequence[physics_probe.ProbeSample],
    config: EnvConfig,
    scenario: str,
    warmup_steps: int,
) -> SweepResult:
    speeds = [s.speed for s in samples]
    forward_accel = [s.forward_accel for s in samples]
    max_speed_actual = max(speeds) if speeds else 0.0

    max_speed_target = config.car.max_speed
    t_25 = _time_to_threshold(samples, max_speed_target * 0.25)
    t_50 = _time_to_threshold(samples, max_speed_target * 0.50)
    t_75 = _time_to_threshold(samples, max_speed_target * 0.75)

    dt = samples[1].time_s - samples[0].time_s if len(samples) > 1 else config.physics_dt
    window = max(1, int(1.0 / max(dt, 1e-6)))
    avg_forward_accel_1s = _avg_window(forward_accel, window, start=0, end=window)
    avg_speed_last_1s = _avg_window(speeds, window)

    avg_speed_pre_turn = None
    avg_speed_post_turn = None
    if scenario in TURN_SCENARIOS and warmup_steps > 0:
        avg_speed_pre_turn = _avg_window(speeds, window, end=warmup_steps)
        avg_speed_post_turn = _avg_window(speeds, window)

    return SweepResult(
        config_path=str(config.source_path) if hasattr(config, "source_path") else "",
        scenario=scenario,
        max_speed_target=max_speed_target,
        max_speed_actual=max_speed_actual,
        time_to_25=t_25,
        time_to_50=t_50,
        time_to_75=t_75,
        avg_forward_accel_1s=avg_forward_accel_1s,
        avg_speed_last_1s=avg_speed_last_1s,
        avg_speed_pre_turn=avg_speed_pre_turn,
        avg_speed_post_turn=avg_speed_post_turn,
    )


def _write_csv(path: Path, results: Iterable[SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config_path",
            "scenario",
            "max_speed_target",
            "max_speed_actual",
            "time_to_25",
            "time_to_50",
            "time_to_75",
            "avg_forward_accel_1s",
            "avg_speed_last_1s",
            "avg_speed_pre_turn",
            "avg_speed_post_turn",
        ])
        for r in results:
            writer.writerow([
                r.config_path,
                r.scenario,
                f"{r.max_speed_target:.3f}",
                f"{r.max_speed_actual:.3f}",
                "" if r.time_to_25 is None else f"{r.time_to_25:.3f}",
                "" if r.time_to_50 is None else f"{r.time_to_50:.3f}",
                "" if r.time_to_75 is None else f"{r.time_to_75:.3f}",
                "" if r.avg_forward_accel_1s is None else f"{r.avg_forward_accel_1s:.3f}",
                "" if r.avg_speed_last_1s is None else f"{r.avg_speed_last_1s:.3f}",
                "" if r.avg_speed_pre_turn is None else f"{r.avg_speed_pre_turn:.3f}",
                "" if r.avg_speed_post_turn is None else f"{r.avg_speed_post_turn:.3f}",
            ])


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    config_paths = _resolve_configs(args.configs)

    scenarios = [args.scenario]
    if args.scenario == "all":
        scenarios = [
            "straight_full_throttle",
            "coast_after_full",
            "turn_after_full",
        ]

    results: List[SweepResult] = []
    for config_path in config_paths:
        config = physics_probe._load_config(config_path)
        setattr(config, "source_path", config_path)
        for scenario in scenarios:
            samples = physics_probe._run_probe(
                config=config,
                scenario=scenario,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                turn_steer=args.turn_steer,
                turn_throttle=args.turn_throttle,
            )
            results.append(
                _summarize(samples, config, scenario, args.warmup_steps)
            )

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("logs") / f"physics_sweep_{stamp}.csv"

    _write_csv(output_path, results)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
