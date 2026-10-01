import torch
import torch.distributed as dist


class ConstraintOptimizer(torch.nn.Module):
    """Turn named non-negative constraint residuals into scalar loss terms."""

    @staticmethod
    def _validate_scale(scale):
        scale = float(scale)
        if scale < 0:
            raise ValueError("Constraint scales must be non-negative.")
        return scale

    def penalty(self, name, residual, scale=1.0, update_state=False):
        raise NotImplementedError

    def update(self, completed_epoch):
        return None


class FixedPenalty(ConstraintOptimizer):
    def penalty(self, name, residual, scale=1.0, update_state=False):
        return self._validate_scale(scale) * residual.square()


class AugmentedLagrangian(ConstraintOptimizer):
    def __init__(self, names, config):
        super().__init__()
        self.names = tuple(names)
        self.register_buffer("rho", torch.tensor(float(config.get("rho", 1.0))))
        self.rho_growth = float(config.get("rho_growth", 2.0))
        self.rho_max = float(config.get("rho_max", 1.0e4))
        self.required_reduction = float(config.get("required_reduction", 0.9))
        self.update_every = int(config.get("update_every", 1))
        self.register_buffer("previous_residual_norm", torch.tensor(float("inf")))
        for name in self.names:
            self.register_buffer(f"dual_{name}", torch.tensor(0.0))
        self._sums = dict.fromkeys(self.names, 0.0)
        self._counts = dict.fromkeys(self.names, 0)
        if self.rho <= 0 or self.rho_growth < 1 or self.rho_max < self.rho:
            raise ValueError("augmented_lagrangian requires 0 < rho <= rho_max.")
        if not 0 < self.required_reduction <= 1 or self.update_every <= 0:
            raise ValueError("Invalid augmented_lagrangian update configuration.")

    def penalty(self, name, residual, scale=1.0, update_state=False):
        value = self._validate_scale(scale) * residual
        if update_state:
            self._sums[name] += float(value.detach())
            self._counts[name] += 1
        dual = getattr(self, f"dual_{name}").to(value)
        return dual * value + 0.5 * self.rho * value.square()

    @torch.no_grad()
    def update(self, completed_epoch):
        if completed_epoch < 0 or (completed_epoch + 1) % self.update_every:
            return
        active = self.rho.new_tensor(float(any(self._counts.values())))
        if dist.is_initialized():
            dist.all_reduce(active)
        if active == 0:
            return
        means = []
        for name in self.names:
            stats = self.rho.new_tensor([self._sums[name], self._counts[name]])
            if dist.is_initialized():
                dist.all_reduce(stats)
            mean = stats[0] / stats[1].clamp_min(1)
            means.append(mean)
            if stats[1] > 0:
                dual = getattr(self, f"dual_{name}")
                dual.copy_(torch.clamp_min(dual + self.rho * mean, 0))
            self._sums[name], self._counts[name] = 0.0, 0
        norm = torch.linalg.vector_norm(torch.stack(means))
        if (
            torch.isfinite(self.previous_residual_norm)
            and norm > self.required_reduction * self.previous_residual_norm
        ):
            self.rho.copy_(torch.clamp(self.rho * self.rho_growth, max=self.rho_max))
        self.previous_residual_norm.copy_(norm)


def create_constraint_optimizer(names, config):
    kind = config.get("type", "fixed_penalty")
    if kind == "fixed_penalty":
        return FixedPenalty()
    if kind == "augmented_lagrangian":
        return AugmentedLagrangian(names, config)
    raise ValueError(f"Unknown constraint_optimizer type: {kind!r}.")
