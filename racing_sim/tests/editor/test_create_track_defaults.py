import importlib

from racing_sim.config.config import EnvConfig


def import_create_track_module():
    return importlib.import_module("scripts.create_track")


def test_track_export_uses_good_model_defaults(tmp_path):
    create_track = import_create_track_module()
    editor = create_track.TrackEditor(800, 600)

    editor.state.track.nodes = [
        create_track.NodeData(x=0.0, y=0.0, radius=0.0),
        create_track.NodeData(x=10.0, y=0.0, radius=0.0),
        create_track.NodeData(x=10.0, y=10.0, radius=0.0),
    ]
    editor.state.track.width = 4.0
    editor.state.track.start_node_index = 0
    editor.state.track.start_offset = 0.0

    output_path = tmp_path / "track.yaml"
    editor._save_track_to_path(str(output_path))

    config = EnvConfig.from_yaml(str(output_path))
    assert config.obs_type == "grid"
    assert config.progress_reward_scale == 0.5
    assert config.speed_bonus_scale == 0.01
    assert config.collision_penalty == -2.0
    assert config.time_penalty == 0.0
    assert config.max_episode_steps == 512
