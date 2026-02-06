"""Dropout-enabled policies for PPO and SAC."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic


def _build_mlp_with_dropout(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module],
    dropout: float,
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    last_dim = input_dim

    for hidden_dim in net_arch:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_fn())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim

    if output_dim > 0:
        if not net_arch and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(last_dim, output_dim))

    return layers


class DropoutMlpExtractor(nn.Module):
    """MLP extractor with optional dropout after each activation.

    If no hidden layers are specified, dropout (when enabled) is applied
    directly to the input features.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if net_arch is None:
            net_arch = [64, 64]

        self.dropout = float(dropout or 0.0)

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

        self.policy_net = self._build_network(feature_dim, policy_arch, activation_fn)
        self.value_net = self._build_network(feature_dim, value_arch, activation_fn)

        self.latent_dim_pi = policy_arch[-1] if policy_arch else feature_dim
        self.latent_dim_vf = value_arch[-1] if value_arch else feature_dim

    def _build_network(
        self,
        input_dim: int,
        arch: List[int],
        activation_fn: Type[nn.Module],
    ) -> nn.Module:
        if not arch:
            if self.dropout > 0.0:
                return nn.Sequential(nn.Dropout(self.dropout))
            return nn.Identity()

        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in arch:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation_fn())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            last_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class DropoutActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with dropout in the MLP extractor."""

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
        dropout: float = 0.0,
    ) -> None:
        self.dropout = float(dropout or 0.0)

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
        self.mlp_extractor = DropoutMlpExtractor(
            feature_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            dropout=self.dropout,
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({"dropout": self.dropout})
        return data


class DropoutActor(Actor):
    """SAC Actor with dropout in the latent MLP."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        dropout: float = 0.0,
    ) -> None:
        self.dropout = float(dropout or 0.0)

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )

        if self.dropout > 0.0:
            if self.net_arch:
                latent_pi_net = _build_mlp_with_dropout(
                    self.features_dim,
                    -1,
                    self.net_arch,
                    self.activation_fn,
                    self.dropout,
                )
                self.latent_pi = nn.Sequential(*latent_pi_net)
            else:
                self.latent_pi = nn.Sequential(nn.Dropout(self.dropout))

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({"dropout": self.dropout})
        return data


class DropoutContinuousCritic(ContinuousCritic):
    """SAC critic with dropout in each Q network."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        dropout: float = 0.0,
    ) -> None:
        self.dropout = float(dropout or 0.0)
        self._dropout_net_arch = net_arch
        self._dropout_activation_fn = activation_fn
        self._dropout_features_dim = features_dim

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

        if self.dropout > 0.0:
            self._rebuild_q_networks()

    def _rebuild_q_networks(self) -> None:
        action_dim = get_action_dim(self.action_space)

        self.q_networks = []
        for idx in range(self.n_critics):
            q_net_list = _build_mlp_with_dropout(
                self._dropout_features_dim + action_dim,
                1,
                self._dropout_net_arch,
                self._dropout_activation_fn,
                self.dropout,
            )
            q_net = nn.Sequential(*q_net_list)
            setattr(self, f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class DropoutSACPolicy(SACPolicy):
    """SAC policy that applies dropout in actor/critic MLPs."""

    def __init__(self, *args, dropout: float = 0.0, **kwargs) -> None:
        self.dropout = float(dropout or 0.0)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        actor_kwargs["dropout"] = self.dropout
        return DropoutActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs["dropout"] = self.dropout
        return DropoutContinuousCritic(**critic_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update({"dropout": self.dropout})
        return data
