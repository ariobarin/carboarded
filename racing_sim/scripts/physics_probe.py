"""Headless physics probe for logging speed/accel/position to CSV."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from racing_sim.config.config import EnvConfig
from racing_sim.envs.racing_env import RacingEnv


@dataclass
class ProbeSample:
    step: int
    time_s: float
    x: float
    y: float
    speed: float
    forward_speed: float
    lateral_speed: float
    accel: float
    forward_accel: float
    steering: float
    throttle: float
    distance: float


SCENARIOS = {
    "straight_full_throttle",
    "straight_half_throttle",
    "coast_after_full",
    "turn_after_full",
    "turn_constant",
    "all",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless physics probe")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--output", type=str, default=None, help="CSV output path (defaults to logs/)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="straight_full_throttle",
        choices=sorted(SCENARIOS),
        help="Probe scenario",
    )
    parser.add_argument("--steps", type=int, default=600, help="Number of steps to run")
    parser.add_argument("--warmup-steps", type=int, default=120, help="Steps before turn/coast")
    parser.add_argument("--turn-steer", type=float, default=0.5, help="Steering for turn scenarios")
    parser.add_argument("--turn-throttle", type=float, default=1.0, help="Throttle during turn")
    parser.add_argument("--plot", action="store_true", help="Plot speed/accel (requires matplotlib)")
    return parser


def _load_config(path: Optional[str]) -> EnvConfig:
    if path:
        config = EnvConfig.from_yaml(path)
    else:
        default_path = Path(__file__).parent.parent / "configs" / "physics_v2.yaml"
        config = EnvConfig.from_yaml(str(default_path))

    # Force simple ellipse for probe.
    config.track.waviness = 0.0
    config.track.waves = 0
    config.track.wave_phase = 0.0
    config.random_start = False
    config.obs_type = "lidar"
    return config


def _scenario_action(
    scenario: str,
    step: int,
    warmup_steps: int,
    turn_steer: float,
    turn_throttle: float,
) -> Tuple[float, float]:
    if scenario == "straight_full_throttle":
        return 0.0, 1.0
    if scenario == "straight_half_throttle":
        return 0.0, 0.5
    if scenario == "coast_after_full":
        return (0.0, 1.0) if step < warmup_steps else (0.0, 0.0)
    if scenario == "turn_after_full":
        return (0.0, 1.0) if step < warmup_steps else (turn_steer, turn_throttle)
    if scenario == "turn_constant":
        return turn_steer, turn_throttle
    raise ValueError(f"Unknown scenario: {scenario}")


def _run_probe(
    config: EnvConfig,
    scenario: str,
    steps: int,
    warmup_steps: int,
    turn_steer: float,
    turn_throttle: float,
) -> List[ProbeSample]:
    env = RacingEnv(config=config, render_mode=None)
    env.reset()
    env.car.sensor_only = True

    dt = config.physics_dt
    prev_velocity = np.array([0.0, 0.0], dtype=np.float32)
    prev_position = np.array([env.car.position.x, env.car.position.y], dtype=np.float32)
    distance = 0.0
    samples: List[ProbeSample] = []

    for step in range(steps):
        steering, throttle = _scenario_action(
            scenario, step, warmup_steps, turn_steer, turn_throttle
        )
        action = np.array([steering, throttle], dtype=np.float32)
        env.step(action)

        velocity = np.array([env.car.velocity.x, env.car.velocity.y], dtype=np.float32)
        position = np.array([env.car.position.x, env.car.position.y], dtype=np.float32)

        accel_vec = (velocity - prev_velocity) / max(dt, 1e-6)
        accel = float(np.linalg.norm(accel_vec))

        forward = env.car.body.rotation_vector
        lateral = np.array([-forward.y, forward.x], dtype=np.float32)
        forward_vec = np.array([forward.x, forward.y], dtype=np.float32)

        forward_speed = float(np.dot(forward_vec, velocity))
        lateral_speed = float(np.dot(lateral, velocity))
        forward_accel = float(np.dot(forward_vec, accel_vec))

        step_dist = float(np.linalg.norm(position - prev_position))
        distance += step_dist

        samples.append(
            ProbeSample(
                step=step,
                time_s=step * dt,
                x=float(position[0]),
                y=float(position[1]),
                speed=float(np.linalg.norm(velocity)),
                forward_speed=forward_speed,
                lateral_speed=lateral_speed,
                accel=accel,
                forward_accel=forward_accel,
                steering=steering,
                throttle=throttle,
                distance=distance,
            )
        )

        prev_velocity = velocity
        prev_position = position

    env.close()
    return samples


def _write_csv(path: Path, samples: List[ProbeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "time_s",
            "x",
            "y",
            "speed",
            "forward_speed",
            "lateral_speed",
            "accel",
            "forward_accel",
            "steering",
            "throttle",
            "distance",
        ])
        for s in samples:
            writer.writerow([
                s.step,
                f"{s.time_s:.6f}",
                f"{s.x:.6f}",
                f"{s.y:.6f}",
                f"{s.speed:.6f}",
                f"{s.forward_speed:.6f}",
                f"{s.lateral_speed:.6f}",
                f"{s.accel:.6f}",
                f"{s.forward_accel:.6f}",
                f"{s.steering:.3f}",
                f"{s.throttle:.3f}",
                f"{s.distance:.6f}",
            ])


def _summarize(samples: List[ProbeSample], max_speed: float) -> None:
    speeds = [s.speed for s in samples]
    accel = [s.forward_accel for s in samples]
    max_v = max(speeds) if speeds else 0.0

    def time_to_threshold(threshold: float) -> Optional[float]:
        for s in samples:
            if s.speed >= threshold:
                return s.time_s
        return None

    t_25 = time_to_threshold(max_speed * 0.25)
    t_50 = time_to_threshold(max_speed * 0.50)
    t_75 = time_to_threshold(max_speed * 0.75)

    avg_accel_1s = None
    if samples:
        cutoff = max(1, int(1.0 / (samples[1].time_s - samples[0].time_s))) if len(samples) > 1 else 1
        avg_accel_1s = float(np.mean(accel[:cutoff]))

    print("Probe summary")
    print(f"  Max speed: {max_v:.2f}")
    if t_25 is not None:
        print(f"  Time to 25% max: {t_25:.2f}s")
    if t_50 is not None:
        print(f"  Time to 50% max: {t_50:.2f}s")
    if t_75 is not None:
        print(f"  Time to 75% max: {t_75:.2f}s")
    if avg_accel_1s is not None:
        print(f"  Avg forward accel (first 1s): {avg_accel_1s:.2f}")


def _maybe_plot(samples: List[ProbeSample]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Plot skipped: matplotlib not available.")
        return

    t = [s.time_s for s in samples]
    speed = [s.speed for s in samples]
    accel = [s.forward_accel for s in samples]
    distance = [s.distance for s in samples]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, speed, label="speed")
    axes[0].set_ylabel("Speed")
    axes[0].grid(True)
    axes[1].plot(t, accel, label="forward accel", color="orange")
    axes[1].set_ylabel("Forward Accel")
    axes[1].grid(True)
    axes[2].plot(t, distance, label="distance", color="green")
    axes[2].set_ylabel("Distance")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)
    plt.tight_layout()
    plt.show()


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = _load_config(args.config)

    scenarios = [args.scenario] if args.scenario != "all" else [
        "straight_full_throttle",
        "coast_after_full",
        "turn_after_full",
    ]

    for scenario in scenarios:
        samples = _run_probe(
            config=config,
            scenario=scenario,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            turn_steer=args.turn_steer,
            turn_throttle=args.turn_throttle,
        )

        if args.output:
            output_path = Path(args.output)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path("logs") / f"physics_probe_{scenario}_{stamp}.csv"

        _write_csv(output_path, samples)
        print(f"Saved: {output_path}")
        _summarize(samples, max_speed=config.car.max_speed)
        if args.plot:
            _maybe_plot(samples)


if __name__ == "__main__":
    main()
