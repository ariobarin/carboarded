"""Configuration dataclasses for the racing simulation."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import yaml
from pathlib import Path


@dataclass
class NodeConfig:
    """Configuration for a single track node."""
    x: float
    y: float
    radius: float = 0.0


@dataclass
class NodeTrackConfig:
    """Configuration for a node-based custom track."""
    nodes: List[NodeConfig] = field(default_factory=list)
    width: float = 100.0
    start_node_index: int = 0
    start_offset: float = 0.0
    num_checkpoints: int = 64


@dataclass
class CarConfig:
    """Car physics configuration."""
    mass: float = 1.0
    width: float = 40.0
    height: float = 20.0
    max_speed: float = 360.0
    engine_power: float = 200.0
    accel_boost: float = 0.05
    throttle_response: float = 0.05
    lateral_friction: float = 1.3
    rolling_friction: float = 0.03
    air_drag: float = 0.004
    angular_damping: float = 0.6
    steering_power: float = 900.0
    steering_speed_ref: float = 450.0
    steering_min_factor: float = 0.02
    turning_drag: float = 0.7


@dataclass
class LidarConfig:
    """Lidar sensor configuration."""
    num_rays: int = 9
    ray_angles: List[float] = field(
        default_factory=lambda: [-90.0, -67.5, -45.0, -22.5, 0.0, 22.5, 45.0, 67.5, 90.0]
    )
    max_distance: float = 400.0


@dataclass
class TrackConfig:
    """Track geometry configuration."""
    width: float = 200.0
    outer_radius_x: float = 350.0
    outer_radius_y: float = 250.0
    center_x: float = 400.0
    center_y: float = 300.0
    waviness: float = 0.0
    waves: int = 0
    wave_phase: float = 0.0
    # Track type discriminator: "elliptical" (default) or "custom" (node-based)
    track_type: str = "elliptical"
    # Custom track configuration (used when track_type="custom")
    custom: Optional[NodeTrackConfig] = None


@dataclass
class GridConfig:
    """Grid sensor configuration for CNN observations.

    Uses perspective (homographic) projection to simulate a camera mounted on
    the car looking down at the ground. This creates a trapezoidal sampling
    pattern: narrow at far distances, wide at near distances.
    """
    grid_size: int = 36           # NxN grid (minimum for NatureCNN)
    camera_height: float = 50.0   # Height above ground (world units)
    camera_pitch: float = 45.0    # Downward tilt from horizontal (degrees)
    fov_horizontal: float = 60.0  # Horizontal field of view (degrees)
    near_distance: float = 30.0   # Min visible distance ahead of car
    far_distance: float = 200.0   # Max visible distance ahead of car


@dataclass
class RenderConfig:
    """Rendering configuration."""
    screen_width: int = 800
    screen_height: int = 600
    fps: int = 60
    show_lidar: bool = True
    show_grid: bool = True
    grid_size: int = 10
    grid_cell_size: float = 20.0
    grid_samples: int = 3
    car_color: tuple = (0, 100, 255)
    wall_color: tuple = (100, 100, 100)
    lidar_clear_color: tuple = (0, 255, 0)
    lidar_hit_color: tuple = (255, 0, 0)
    background_color: tuple = (30, 30, 30)
    grid_on_color: tuple = (0, 200, 0)
    grid_off_color: tuple = (200, 0, 0)
    
    # POV mode settings
    pov_car_offset_y: int = 80  # Distance from bottom of screen to car center
    pov_lidar_max_length: int = 400  # Max lidar ray length in pixels
    pov_grid_max_size: int = 300  # Max grid overlay size in pixels
    pov_worldspace_scale: float = 2.0  # Pixels per world unit for worldspace grid


@dataclass
class EnvConfig:
    """Combined environment configuration."""
    car: CarConfig = field(default_factory=CarConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    physics_dt: float = 1.0 / 60.0
    max_episode_steps: int = 2000

    # Observation type: "lidar" (MLP) or "grid" (CNN occupancy grid)
    obs_type: str = "grid"

    # Reward configuration
    checkpoint_reward: float = 1.0
    max_checkpoint_skip: int = 10
    speed_bonus_scale: float = 0.05
    progress_reward_scale: float = 0.75
    slowdown_distance: float = 0.0
    slowdown_penalty_scale: float = 0.0
    collision_penalty: float = -20.0
    time_penalty: float = 0.0
    off_track_penalty: float = 0.0
    max_off_track_steps: int = 0
    terminate_on_collision: bool = True
    wall_contact_penalty: float = 0.0
    max_wall_contact_steps: int = 0

    # Randomization
    random_start: bool = True  # Start car at random checkpoint each episode
    random_start_lateral_fraction: float = 1.0  # Fraction of safe half-width for lateral spawn offset

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
            track_data = data["track"].copy()
            # Handle custom track configuration
            if "custom" in track_data and track_data.get("track_type") == "custom":
                custom_data = track_data.pop("custom")
                nodes = [NodeConfig(**node) for node in custom_data.get("nodes", [])]
                custom_config = NodeTrackConfig(
                    nodes=nodes,
                    width=custom_data.get("width", 100.0),
                    start_node_index=custom_data.get("start_node_index", 0),
                    start_offset=custom_data.get("start_offset", 0.0),
                    num_checkpoints=custom_data.get("num_checkpoints", 64),
                )
                track_data["custom"] = custom_config
            elif "custom" in track_data:
                # Remove custom if track_type is not "custom"
                track_data.pop("custom", None)
            config.track = TrackConfig(**track_data)
        if "grid" in data:
            config.grid = GridConfig(**data["grid"])
        if "obs_type" in data:
            config.obs_type = data["obs_type"]
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
        if "max_checkpoint_skip" in data:
            config.max_checkpoint_skip = data["max_checkpoint_skip"]
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
        if "off_track_penalty" in data:
            config.off_track_penalty = data["off_track_penalty"]
        if "max_off_track_steps" in data:
            config.max_off_track_steps = data["max_off_track_steps"]
        if "terminate_on_collision" in data:
            config.terminate_on_collision = data["terminate_on_collision"]
        if "wall_contact_penalty" in data:
            config.wall_contact_penalty = data["wall_contact_penalty"]
        if "max_wall_contact_steps" in data:
            config.max_wall_contact_steps = data["max_wall_contact_steps"]
        if "random_start" in data:
            config.random_start = data["random_start"]
        if "random_start_lateral_fraction" in data:
            config.random_start_lateral_fraction = data["random_start_lateral_fraction"]

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
                "accel_boost": self.car.accel_boost,
                "throttle_response": self.car.throttle_response,
                "lateral_friction": self.car.lateral_friction,
                "rolling_friction": self.car.rolling_friction,
                "air_drag": self.car.air_drag,
                "angular_damping": self.car.angular_damping,
                "steering_power": self.car.steering_power,
                "steering_speed_ref": self.car.steering_speed_ref,
                "steering_min_factor": self.car.steering_min_factor,
                "turning_drag": self.car.turning_drag,
            },
            "lidar": {
                "num_rays": self.lidar.num_rays,
                "ray_angles": self.lidar.ray_angles,
                "max_distance": self.lidar.max_distance,
            },
            "track": self._track_to_dict(),
            "grid": {
                "grid_size": self.grid.grid_size,
                "camera_height": self.grid.camera_height,
                "camera_pitch": self.grid.camera_pitch,
                "fov_horizontal": self.grid.fov_horizontal,
                "near_distance": self.grid.near_distance,
                "far_distance": self.grid.far_distance,
            },
            "obs_type": self.obs_type,
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
            "max_checkpoint_skip": self.max_checkpoint_skip,
            "speed_bonus_scale": self.speed_bonus_scale,
            "progress_reward_scale": self.progress_reward_scale,
            "slowdown_distance": self.slowdown_distance,
            "slowdown_penalty_scale": self.slowdown_penalty_scale,
            "collision_penalty": self.collision_penalty,
            "time_penalty": self.time_penalty,
            "off_track_penalty": self.off_track_penalty,
            "max_off_track_steps": self.max_off_track_steps,
            "terminate_on_collision": self.terminate_on_collision,
            "wall_contact_penalty": self.wall_contact_penalty,
            "max_wall_contact_steps": self.max_wall_contact_steps,
            "random_start": self.random_start,
            "random_start_lateral_fraction": self.random_start_lateral_fraction,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _track_to_dict(self) -> dict:
        """Convert track config to dictionary for YAML serialization."""
        if self.track.track_type == "custom" and self.track.custom is not None:
            return {
                "track_type": "custom",
                "custom": {
                    "width": self.track.custom.width,
                    "num_checkpoints": self.track.custom.num_checkpoints,
                    "start_node_index": self.track.custom.start_node_index,
                    "start_offset": self.track.custom.start_offset,
                    "nodes": [
                        {"x": node.x, "y": node.y, "radius": node.radius}
                        for node in self.track.custom.nodes
                    ],
                },
            }
        else:
            return {
                "width": self.track.width,
                "outer_radius_x": self.track.outer_radius_x,
                "outer_radius_y": self.track.outer_radius_y,
                "center_x": self.track.center_x,
                "center_y": self.track.center_y,
                "waviness": self.track.waviness,
                "waves": self.track.waves,
                "wave_phase": self.track.wave_phase,
            }
