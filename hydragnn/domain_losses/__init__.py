import copy

from .interatomic import InteratomicPotentialDomainLoss
from .constraint_optimizers import create_constraint_optimizer


def create_domain_loss(model, config):
    if not config or not config.get("enabled", False):
        return model
    provider = config.get("provider")
    if provider == "interatomic_potential":
        return InteratomicPotentialDomainLoss(model, config)
    if provider == "optimal_power_flow":
        raise ValueError(
            "The optimal_power_flow provider requires dataset metadata and cannot "
            "be attached by create_model_config. Use the OPF training entry point, "
            "which constructs OPFDomainLoss after model creation."
        )
    raise ValueError(f"Unknown training-loss provider: {provider!r}.")


def uses_interatomic_potential(config):
    domain = config.get("NeuralNetwork", {}).get("Training", {}).get("loss", {})
    return bool(
        domain.get("enabled") and domain.get("provider") == "interatomic_potential"
    )


def defer_domain_loss(config, provider):
    """Disable a metadata-dependent provider during base-model construction."""
    model_config = copy.deepcopy(config)
    loss = model_config.get("Training", {}).get("loss")
    if loss is None or not loss.get("enabled", False):
        return model_config
    if loss.get("provider") != provider:
        raise ValueError(
            f"Cannot defer provider {loss.get('provider')!r}; expected {provider!r}."
        )
    loss["enabled"] = False
    return model_config


__all__ = [
    "InteratomicPotentialDomainLoss",
    "create_domain_loss",
    "defer_domain_loss",
    "uses_interatomic_potential",
    "create_constraint_optimizer",
]
