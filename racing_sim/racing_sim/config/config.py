"""Configuration dataclasses for the racing simulation."""

from dataclasses import dataclass, field
from typing import List
import yaml
from pathlib import Path


@dataclass
class CarConfig:
    """Car physics configuration."""
    mass: float = 1.0
    width: float = 40.0
    height: float = 20.0
    max_speed: float = 1000.0
    engine_power: float = 500.0
    lateral_friction: float = 0.5
    rolling_friction: float = 0.05
    angular_damping: float = 0.7
    steering_power: float = 500.0


@dataclass
class LidarConfig:
    """Lidar sensor configuration."""
    num_rays: int = 5
    ray_angles: List[float] = field(default_factory=lambda: [-60.0, -30.0, 0.0, 30.0, 60.0])
    max_distance: float = 200.0


@dataclass
class TrackConfig:
    """Track geometry configuration."""
    width: float = 100.0
    outer_radius_x: float = 350.0
    outer_radius_y: float = 250.0
    center_x: float = 400.0
    center_y: float = 300.0
    waviness: float = 0.0
    waves: int = 0
    wave_phase: float = 0.0


@dataclass
class RenderConfig:
    """Rendering configuration."""
    screen_width: int = 800
    screen_height: int = 600
    fps: int = 60
    show_lidar: bool = True
    car_color: tuple = (0, 100, 255)
    wall_color: tuple = (100, 100, 100)
    lidar_clear_color: tuple = (0, 255, 0)
    lidar_hit_color: tuple = (255, 0, 0)
    background_color: tuple = (30, 30, 30)


@dataclass
class EnvConfig:
    """Combined environment configuration."""
    car: CarConfig = field(default_factory=CarConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    physics_dt: float = 1.0 / 60.0
    max_episode_steps: int = 1000

    # Reward configuration
    checkpoint_reward: float = 1.0
    speed_bonus_scale: float = 0.1
    progress_reward_scale: float = 0.0
    slowdown_distance: float = 0.4
    slowdown_penalty_scale: float = 2.0
    collision_penalty: float = -10.0
    time_penalty: float = -0.1

    @classmethod
    def from_yaml(cls, path: str) -> "EnvConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        if "car" in data:
            config.car = CarConfig(**data["car"])
        if "lidar" in data:
            config.lidar = LidarConfig(**data["lidar"])
        if "track" in data:
            config.track = TrackConfig(**data["track"])
        if "render" in data:
            # Convert color lists to tuples
            render_data = data["render"]
            for key in ["car_color", "wall_color", "lidar_clear_color",
                        "lidar_hit_color", "background_color"]:
                if key in render_data:
                    render_data[key] = tuple(render_data[key])
            config.render = RenderConfig(**render_data)
        if "physics_dt" in data:
            config.physics_dt = data["physics_dt"]
        if "max_episode_steps" in data:
            config.max_episode_steps = data["max_episode_steps"]
        if "checkpoint_reward" in data:
            config.checkpoint_reward = data["checkpoint_reward"]
        if "speed_bonus_scale" in data:
            config.speed_bonus_scale = data["speed_bonus_scale"]
        if "collision_penalty" in data:
            config.collision_penalty = data["collision_penalty"]
        if "progress_reward_scale" in data:
            config.progress_reward_scale = data["progress_reward_scale"]
        if "slowdown_distance" in data:
            config.slowdown_distance = data["slowdown_distance"]
        if "slowdown_penalty_scale" in data:
            config.slowdown_penalty_scale = data["slowdown_penalty_scale"]
        if "time_penalty" in data:
            config.time_penalty = data["time_penalty"]

        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            "car": {
                "mass": self.car.mass,
                "width": self.car.width,
                "height": self.car.height,
                "max_speed": self.car.max_speed,
                "engine_power": self.car.engine_power,
                "lateral_friction": self.car.lateral_friction,
                "rolling_friction": self.car.rolling_friction,
                "angular_damping": self.car.angular_damping,
                "steering_power": self.car.steering_power,
            },
            "lidar": {
                "num_rays": self.lidar.num_rays,
                "ray_angles": self.lidar.ray_angles,
                "max_distance": self.lidar.max_distance,
            },
            "track": {
                "width": self.track.width,
                "outer_radius_x": self.track.outer_radius_x,
                "outer_radius_y": self.track.outer_radius_y,
                "center_x": self.track.center_x,
                "center_y": self.track.center_y,
                "waviness": self.track.waviness,
                "waves": self.track.waves,
                "wave_phase": self.track.wave_phase,
            },
            "render": {
                "screen_width": self.render.screen_width,
                "screen_height": self.render.screen_height,
                "fps": self.render.fps,
                "show_lidar": self.render.show_lidar,
                "car_color": list(self.render.car_color),
                "wall_color": list(self.render.wall_color),
                "lidar_clear_color": list(self.render.lidar_clear_color),
                "lidar_hit_color": list(self.render.lidar_hit_color),
                "background_color": list(self.render.background_color),
            },
            "physics_dt": self.physics_dt,
            "max_episode_steps": self.max_episode_steps,
            "checkpoint_reward": self.checkpoint_reward,
            "speed_bonus_scale": self.speed_bonus_scale,
            "progress_reward_scale": self.progress_reward_scale,
            "slowdown_distance": self.slowdown_distance,
            "slowdown_penalty_scale": self.slowdown_penalty_scale,
            "collision_penalty": self.collision_penalty,
            "time_penalty": self.time_penalty,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
