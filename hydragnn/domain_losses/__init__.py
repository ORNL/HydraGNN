from .interatomic import InteratomicPotentialDomainLoss


def create_domain_loss(model, config):
    if not config or not config.get("enabled", False):
        return model
    provider = config.get("provider")
    if provider == "interatomic_potential":
        return InteratomicPotentialDomainLoss(model, config)
    if provider == "optimal_power_flow":
        # OPF attaches its provider after dataset metadata is available.
        return model
    raise ValueError(f"Unknown DomainLoss provider: {provider!r}.")


def uses_interatomic_potential(config):
    domain = config.get("NeuralNetwork", {}).get("Training", {}).get("DomainLoss", {})
    return bool(
        domain.get("enabled") and domain.get("provider") == "interatomic_potential"
    )


__all__ = [
    "InteratomicPotentialDomainLoss",
    "create_domain_loss",
    "uses_interatomic_potential",
]
