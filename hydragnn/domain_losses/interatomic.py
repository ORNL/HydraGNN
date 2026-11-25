import torch
import torch_scatter


class InteratomicPotentialDomainLoss(torch.nn.Module):
    """Declarative energy/force training objective for interatomic potentials."""

    _SUPPORTED_TERMS = {"energy", "energy_per_atom", "forces", "hessian", "stress"}
    requires_find_unused_parameters = True

    @staticmethod
    def _validate_energy_derivative(
        name, prediction, operator, with_respect_to="positions"
    ):
        expected = {
            "operator": operator,
            "of": "energy",
            "with_respect_to": with_respect_to,
        }
        for field, value in expected.items():
            if prediction.get(field) != value:
                raise ValueError(
                    f"The {name} term requires prediction.{field}={value!r}."
                )

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        term_list = config.get("supervised", {}).get("terms", [])
        term_names = [term["variable"] for term in term_list]
        duplicates = sorted({name for name in term_names if term_names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate interatomic training-loss terms: {duplicates}")
        self.terms = {term["variable"]: term for term in term_list}
        unknown = set(self.terms) - self._SUPPORTED_TERMS
        if unknown:
            raise ValueError(
                f"Unsupported interatomic training-loss terms: {sorted(unknown)}"
            )
        self.active_terms = [
            name
            for name, term in self.terms.items()
            if float(term.get("weight", 0.0)) > 0
        ]
        if not self.active_terms:
            raise ValueError(
                "Interatomic training loss requires at least one active term."
            )
        if "forces" in self.active_terms:
            prediction = self.terms["forces"].get("prediction", {})
            self._validate_energy_derivative("forces", prediction, "negative_gradient")
        if "hessian" in self.active_terms:
            prediction = self.terms["hessian"].get("prediction", {})
            self._validate_energy_derivative("hessian", prediction, "hessian")
        if "stress" in self.active_terms:
            prediction = self.terms["stress"].get("prediction", {})
            self._validate_energy_derivative(
                "stress", prediction, "gradient_per_volume", "strain"
            )
        expected = {
            "energy": ("energy", "per_structure"),
            "energy_per_atom": ("energy_per_atom", "per_atom"),
        }
        for name, (target, normalization) in expected.items():
            if name not in self.active_terms:
                continue
            term = self.terms[name]
            if (
                term.get("variable") != target
                or term.get("normalization", normalization) != normalization
            ):
                raise ValueError(
                    f"Term {name!r} requires target={target!r} and "
                    f"normalization={normalization!r}."
                )
        if (
            "forces" in self.active_terms
            and self.terms["forces"].get("variable") != "forces"
        ):
            raise ValueError("The forces term requires target='forces'.")
        if self.model.num_heads != 1:
            raise ValueError(
                "Interatomic training loss requires exactly one energy head."
            )
        if list(self.model.head_dims) != [1]:
            raise ValueError(
                "Interatomic training loss requires a scalar energy head with dim=1."
            )
        self.atomistic_mode_enabled = True
        self.task_names = list(self.active_terms)
        self.task_weights = [
            float(self.terms[name]["weight"]) for name in self.task_names
        ]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, data):
        self._stress_displacement = None
        self._stress_cell = None
        if "stress" not in self.active_terms:
            return self.model(data)

        # Imported lazily: hydragnn.models imports hydragnn.domain_losses.
        from hydragnn.models.allscaip_stack import AllScAIPStack

        # AllScAIP owns its differentiable graph construction and applies
        # strain internally. Other backbones receive the same symmetric
        # infinitesimal strain here.
        if isinstance(self.model, AllScAIPStack):
            prediction = self.model(data)
            self._stress_displacement = getattr(
                self.model, "_hydragnn_stress_displacement", None
            )
            self._stress_cell = getattr(self.model, "_hydragnn_stress_cell", None)
            return prediction

        if getattr(data, "cell", None) is None:
            raise ValueError(
                "The enabled stress term requires data.cell for every graph."
            )
        num_graphs = int(data.batch.max().item()) + 1
        cell = data.cell
        if cell.dim() == 2 and cell.shape == (3, 3):
            cell = cell.unsqueeze(0).expand(num_graphs, 3, 3).contiguous()
        elif cell.dim() == 2 and cell.shape == (3 * num_graphs, 3):
            cell = cell.view(num_graphs, 3, 3)
        elif cell.dim() == 3 and cell.shape[1:] == (3, 3):
            if cell.shape[0] == 1 and num_graphs > 1:
                cell = cell.expand(num_graphs, 3, 3).contiguous()
            elif cell.shape[0] != num_graphs:
                raise ValueError(
                    f"Expected one cell or {num_graphs} cells for stress "
                    f"loss, got {cell.shape[0]}."
                )
        else:
            raise ValueError(
                f"Unexpected cell shape {tuple(cell.shape)} for stress loss."
            )
        displacement = torch.zeros_like(cell, requires_grad=True)
        strain = 0.5 * (displacement + displacement.transpose(-1, -2))
        original_pos = data.pos
        original_cell = data.cell
        original_edge_shifts = getattr(data, "edge_shifts", None)
        data.pos = original_pos + torch.bmm(
            original_pos.unsqueeze(1), strain[data.batch]
        ).squeeze(1)
        strained_cell = cell + torch.bmm(cell, strain)
        data.cell = strained_cell
        if original_edge_shifts is not None and original_edge_shifts.numel() > 0:
            # Periodic-image offsets must deform with the cell, or the strain
            # derivative misses the cross-boundary contributions to the virial.
            edge_graph = data.batch[data.edge_index[0]]
            integer_shifts = getattr(data, "integer_edge_shifts", None)
            if integer_shifts is not None:
                data.edge_shifts = torch.bmm(
                    integer_shifts.to(strained_cell.dtype).unsqueeze(1),
                    strained_cell[edge_graph],
                ).squeeze(1)
            else:
                data.edge_shifts = original_edge_shifts + torch.bmm(
                    original_edge_shifts.unsqueeze(1), strain[edge_graph]
                ).squeeze(1)
        try:
            prediction = self.model(data)
        finally:
            data.pos = original_pos
            data.cell = original_cell
            if original_edge_shifts is not None:
                data.edge_shifts = original_edge_shifts
        self._stress_displacement = displacement
        self._stress_cell = cell
        return prediction

    def _loss(self, name, prediction, target):
        # Imported lazily: hydragnn.models imports hydragnn.domain_losses.
        from hydragnn.models.utils import loss_function_selection

        if prediction.numel() == 0:
            return prediction.sum()
        loss_name = self.terms[name].get(
            "metric",
            self.config.get("supervised", {}).get(
                "default_metric", self.model.loss_function_type
            ),
        )
        loss_function = loss_function_selection(loss_name)
        if loss_function is None:
            raise ValueError(f"Unknown loss function {loss_name!r} for term {name!r}.")
        return loss_function(prediction, target)

    def _graph_energy(self, pred, data):
        if self.head_type[0] == "node":
            return torch_scatter.scatter_add(pred[0], data.batch, dim=0).squeeze(-1)
        if self.head_type[0] == "graph":
            if getattr(self.model, "graph_pooling", "mean") != "add":
                raise ValueError("Graph energy heads require graph_pooling='add'.")
            if isinstance(pred, dict):
                return pred["graph"][0].squeeze(-1)
            if isinstance(pred, (list, tuple)):
                return pred[0].squeeze(-1)
            return pred.squeeze(-1)
        raise ValueError(
            "Interatomic energy must be predicted by a node or graph head."
        )

    def prediction_target_pairs(self, pred, data, create_graph=True):
        """Build predictions and targets for the configured active terms."""
        if getattr(data, "pos", None) is None:
            raise ValueError("Interatomic training loss requires data.pos.")
        needs_energy_target = bool(
            {"energy", "energy_per_atom"}.intersection(self.active_terms)
        )
        if needs_energy_target and getattr(data, "energy", None) is None:
            raise ValueError(
                "Enabled energy terms require an energy target in data.energy."
            )

        energy_pred = self._graph_energy(pred, data)
        values = {}
        if needs_energy_target:
            energy_true = data.energy.to(energy_pred).reshape_as(energy_pred)
            if "energy" in self.active_terms:
                values["energy"] = (energy_pred, energy_true)
            if "energy_per_atom" in self.active_terms:
                atom_counts = torch.bincount(data.batch).to(energy_pred.dtype)
                values["energy_per_atom"] = (
                    energy_pred / atom_counts,
                    energy_true / atom_counts,
                )

        needs_derivatives = bool({"forces", "hessian"}.intersection(self.active_terms))
        if needs_derivatives:
            if not data.pos.requires_grad:
                raise ValueError(
                    "data.pos must require gradients for force or Hessian training."
                )
            if "forces" in self.active_terms and getattr(data, "forces", None) is None:
                raise ValueError("The enabled forces term requires data.forces.")
            if (
                "hessian" in self.active_terms
                and getattr(data, "hessian", None) is None
            ):
                raise ValueError("The enabled hessian term requires data.hessian.")
            if "hessian" in self.active_terms and data.num_graphs != 1:
                raise ValueError("Hessian loss currently requires batch size 1.")
            if not energy_pred.requires_grad:
                raise ValueError(
                    "Predicted energy is not differentiable with respect to positions."
                )
            gradient = torch.autograd.grad(
                energy_pred,
                data.pos,
                grad_outputs=torch.ones_like(energy_pred),
                retain_graph=energy_pred.requires_grad,
                create_graph=create_graph or "hessian" in self.active_terms,
                allow_unused=True,
            )[0]
            if gradient is None:
                raise ValueError(
                    "Predicted energy is not differentiable with respect to positions."
                )
            force_prediction = -gradient
            if "forces" in self.active_terms:
                force_target = data.forces.to(force_prediction)
                finite = torch.isfinite(force_target)
                values["forces"] = (
                    force_prediction[finite],
                    force_target[finite],
                )
            if "hessian" in self.active_terms:
                rows = []
                for component in force_prediction.reshape(-1):
                    force_gradient = torch.autograd.grad(
                        component,
                        data.pos,
                        retain_graph=True,
                        create_graph=create_graph,
                    )[0]
                    rows.append(-force_gradient.reshape(-1))
                hessian_prediction = torch.stack(rows)
                hessian_target = data.hessian.to(hessian_prediction).reshape_as(
                    hessian_prediction
                )
                finite = torch.isfinite(hessian_target)
                values["hessian"] = (
                    hessian_prediction[finite],
                    hessian_target[finite],
                )
        if "stress" in self.active_terms:
            if getattr(data, "stress", None) is None:
                raise ValueError("The enabled stress term requires data.stress.")
            displacement = getattr(self, "_stress_displacement", None)
            cell = getattr(self, "_stress_cell", None)
            if displacement is None or cell is None:
                raise RuntimeError(
                    "The selected MLIP backbone did not expose a differentiable "
                    "cell displacement for stress prediction."
                )
            virial = torch.autograd.grad(
                energy_pred,
                displacement,
                grad_outputs=torch.ones_like(energy_pred),
                retain_graph=energy_pred.requires_grad,
                create_graph=create_graph,
            )[0]
            volume = torch.det(cell).abs().view(-1, 1, 1)
            min_volume = torch.finfo(volume.dtype).eps
            if not torch.isfinite(volume).all() or torch.any(volume <= min_volume):
                raise ValueError(
                    "stress loss requires finite, non-singular simulation cells."
                )
            stress_prediction = virial / volume
            stress_target = data.stress.to(stress_prediction).reshape_as(
                stress_prediction
            )
            values["stress"] = (stress_prediction, stress_target)
        return values

    def energy_force_loss(self, pred, data, create_graph=True):
        values = self.prediction_target_pairs(pred, data, create_graph=create_graph)
        reference_prediction = next(iter(values.values()))[0]

        total = reference_prediction.new_zeros(())
        component_losses = []
        report = {}
        for name in self.active_terms:
            component = self._loss(name, *values[name])
            component_losses.append(component)
            weight = float(self.terms[name]["weight"])
            weighted = weight * component
            total = total + weighted
            report[f"supervised.{name}"] = {
                "raw": component.detach(),
                "weight": weight,
                "weighted": weighted.detach(),
            }
        self.last_loss_components = report
        return total, component_losses
