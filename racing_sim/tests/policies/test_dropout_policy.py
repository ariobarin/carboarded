from torch import nn

from racing_sim.policies.dropout_policy import DropoutMlpExtractor


def test_dropout_mlp_extractor_inserts_dropout_layers():
    extractor = DropoutMlpExtractor(
        feature_dim=8,
        net_arch=[16, 8],
        activation_fn=nn.ReLU,
        dropout=0.25,
    )

    policy_dropouts = [module for module in extractor.policy_net if isinstance(module, nn.Dropout)]
    value_dropouts = [module for module in extractor.value_net if isinstance(module, nn.Dropout)]

    assert len(policy_dropouts) == 2
    assert len(value_dropouts) == 2
    assert all(module.p == 0.25 for module in policy_dropouts)
    assert all(module.p == 0.25 for module in value_dropouts)
