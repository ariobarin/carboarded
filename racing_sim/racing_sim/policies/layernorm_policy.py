"""LayerNorm policy to prevent plasticity loss in PPO.

LayerNorm before each activation helps prevent dead ReLU units and weight
magnitude explosion, which are root causes of PPO policy collapse.

Reference: "Understanding Plasticity in Neural Networks" (Nature 2024)
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class LayerNormMlpExtractor(nn.Module):
    """MLP feature extractor with LayerNorm before each activation.

    Architecture per network (policy and value):
        Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU

    This prevents:
    - Dead ReLU units (LayerNorm normalizes inputs to reasonable range)
    - Weight magnitude explosion (LayerNorm keeps activations bounded)
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()

        if net_arch is None:
            net_arch = [64, 64]

        # Parse net_arch - handle both simple list and dict formats
        policy_arch = []
        value_arch = []

        for layer in net_arch:
            if isinstance(layer, int):
                policy_arch.append(layer)
                value_arch.append(layer)
            elif isinstance(layer, dict):
                if "pi" in layer:
                    policy_arch = layer["pi"]
                if "vf" in layer:
                    value_arch = layer["vf"]

        # Build policy network with LayerNorm
        self.policy_net = self._build_network(feature_dim, policy_arch, activation_fn)
        self.value_net = self._build_network(feature_dim, value_arch, activation_fn)

        # Store output dimensions
        self.latent_dim_pi = policy_arch[-1] if policy_arch else feature_dim
        self.latent_dim_vf = value_arch[-1] if value_arch else feature_dim

    def _build_network(
        self,
        input_dim: int,
        arch: List[int],
        activation_fn: Type[nn.Module],
    ) -> nn.Module:
        """Build a network with LayerNorm before each activation."""
        if not arch:
            return nn.Identity()

        layers = []
        last_dim = input_dim

        for hidden_dim in arch:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn())
            last_dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass through both networks."""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through policy network."""
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Forward pass through value network."""
        return self.value_net(features)


class LayerNormActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with LayerNorm in the MLP extractor.

    Use this policy with PPO to help prevent policy collapse caused by
    plasticity loss. The LayerNorm layers keep activations bounded and
    prevent dead ReLU units from accumulating.

    Usage:
        model = PPO(
            policy=LayerNormActorCriticPolicy,
            env=env,
            ...
        )
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        # Store net_arch before calling super().__init__
        # because _build_mlp_extractor needs it
        self._custom_net_arch = net_arch if net_arch is not None else [64, 64]

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor with LayerNorm."""
        self.mlp_extractor = LayerNormMlpExtractor(
            feature_dim=self.features_dim,
            net_arch=self._custom_net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
