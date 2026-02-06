"""Simple physics tuner for live parameter adjustments."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pygame
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from racing_sim.config.config import EnvConfig
from racing_sim.config.defaults import resolve_env_config
from racing_sim.envs.racing_env import RacingEnv


@dataclass
class ParamSpec:
    name: str
    attr: str
    step: float
    fine_step: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def clamp(self, value: float) -> float:
        if self.min_value is not None:
            value = max(self.min_value, value)
        if self.max_value is not None:
            value = min(self.max_value, value)
        return value


PARAMS: List[ParamSpec] = [
    ParamSpec("engine_power", "engine_power", 25.0, 5.0, min_value=0.0),
    ParamSpec("max_speed", "max_speed", 50.0, 10.0, min_value=0.0),
    ParamSpec("accel_boost", "accel_boost", 0.05, 0.01, min_value=0.0),
    ParamSpec("throttle_response", "throttle_response", 0.05, 0.01, min_value=0.0, max_value=1.0),
    ParamSpec("lateral_friction", "lateral_friction", 0.05, 0.01, min_value=0.0),
    ParamSpec("rolling_friction", "rolling_friction", 0.01, 0.002, min_value=0.0),
    ParamSpec("air_drag", "air_drag", 0.0005, 0.0001, min_value=0.0),
    ParamSpec("angular_damping", "angular_damping", 0.05, 0.01, min_value=0.0),
    ParamSpec("steering_power", "steering_power", 25.0, 5.0, min_value=0.0),
    ParamSpec("steering_speed_ref", "steering_speed_ref", 10.0, 2.0, min_value=1.0),
    ParamSpec("steering_min_factor", "steering_min_factor", 0.05, 0.01, min_value=0.0, max_value=1.0),
    ParamSpec("turning_drag", "turning_drag", 0.005, 0.001, min_value=0.0),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Physics tuner (live config tweaks)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--fps", type=int, default=None, help="Override render FPS (default: 60)")
    return parser


def _format_value(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _apply_delta(config: EnvConfig, spec: ParamSpec, delta: float) -> float:
    current = getattr(config.car, spec.attr)
    updated = spec.clamp(current + delta)
    setattr(config.car, spec.attr, updated)
    return updated


def _set_caption(selected: ParamSpec, value: float) -> None:
    pygame.display.set_caption(
        f"Physics Tuner - [{selected.name}] = {_format_value(value)}"
    )


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    config, _ = resolve_env_config(args.config)

    # Force a simple ellipse track for tuning.
    config.track.waviness = 0.0
    config.track.waves = 0
    config.track.wave_phase = 0.0

    # Tuning focus: lidar + manual control.
    config.obs_type = "lidar"
    config.random_start = False

    if args.fps is not None:
        config.render.fps = args.fps

    env = RacingEnv(config=config, render_mode="human")
    obs, info = env.reset()

    print("\nPhysics Tuner Controls")
    print("  Arrow keys / WASD: drive")
    print("  [ / ] : select parameter")
    print("  - / = : decrease / increase (hold SHIFT for fine step)")
    print("  R: reset car")
    print("  ESC: quit\n")

    selected_idx = 0
    selected = PARAMS[selected_idx]

    # Initialize pygame display before reading input.
    env.render()
    _set_caption(selected, getattr(config.car, selected.attr))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_LEFTBRACKET:
                    selected_idx = (selected_idx - 1) % len(PARAMS)
                    selected = PARAMS[selected_idx]
                    _set_caption(selected, getattr(config.car, selected.attr))
                if event.key == pygame.K_RIGHTBRACKET:
                    selected_idx = (selected_idx + 1) % len(PARAMS)
                    selected = PARAMS[selected_idx]
                    _set_caption(selected, getattr(config.car, selected.attr))

                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    is_fine = (event.mod & pygame.KMOD_SHIFT) != 0
                    step = selected.fine_step if is_fine else selected.step
                    if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        step = -step
                    updated = _apply_delta(config, selected, step)
                    _set_caption(selected, updated)
                    print(f"{selected.name}: {_format_value(updated)}")

        steering, throttle = env.renderer.get_keyboard_input()
        env.renderer.last_actions = {"human": (float(steering), float(throttle))}
        action = np.array([steering, throttle], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

        env.render()

    env.close()


if __name__ == "__main__":
    main()
