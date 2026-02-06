import math

import torch
from torch import nn

from racing_sim.utils.training_utils import clamp_log_std, compute_grad_norm, freeze_cnn_layers


class DummyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(2, 2, kernel_size=3),
        )


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = DummyExtractor()
        self.pi_features_extractor = DummyExtractor()
        self.vf_features_extractor = DummyExtractor()
        self.log_std = nn.Parameter(torch.tensor([0.0, 1.0, -2.0]))


def test_freeze_cnn_layers_freezes_first_param_layers():
    policy = DummyPolicy()
    frozen = freeze_cnn_layers(policy, num_layers=1)

    # First conv layer should be frozen, second should remain trainable.
    for name, param in policy.features_extractor.cnn[0].named_parameters():
        assert param.requires_grad is False
    for name, param in policy.features_extractor.cnn[2].named_parameters():
        assert param.requires_grad is True

    # Same for pi/vf extractors
    for name, param in policy.pi_features_extractor.cnn[0].named_parameters():
        assert param.requires_grad is False
    for name, param in policy.vf_features_extractor.cnn[0].named_parameters():
        assert param.requires_grad is False

    assert frozen


def test_clamp_log_std_applies_bounds():
    policy = DummyPolicy()
    clamp_log_std(policy, min_val=-1.0, max_val=0.5)
    vals = policy.log_std.detach().cpu().numpy().tolist()
    assert vals == [0.0, 0.5, -1.0]


def test_compute_grad_norm():
    model = nn.Linear(2, 2, bias=False)
    for param in model.parameters():
        param.grad = torch.ones_like(param) * 2.0

    norm = compute_grad_norm(model)
    # 4 params * 2.0^2 => sum=16, sqrt=4
    assert math.isclose(norm, 4.0, rel_tol=1e-6)
