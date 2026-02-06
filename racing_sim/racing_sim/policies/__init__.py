"""Custom policies for RL training."""

__all__ = ["LayerNormActorCriticPolicy", "DropoutActorCriticPolicy", "DropoutSACPolicy"]


def __getattr__(name):
    if name == "LayerNormActorCriticPolicy":
        from racing_sim.policies.layernorm_policy import LayerNormActorCriticPolicy
        return LayerNormActorCriticPolicy
    if name in ("DropoutActorCriticPolicy", "DropoutSACPolicy"):
        from racing_sim.policies import dropout_policy
        return getattr(dropout_policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
