"""Shared utilities for OPF solution workflows (heterogeneous and homogeneous)."""

import copy
import logging
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import degree


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


class OPFDomainLoss:
    """Domain-informed regularization for OPF bus-level targets.

    Feasibility penalties (all zero on any strictly feasible OPF solution):
      - voltage_bound_weight           : Penalty for Vm (bus_pred[:, vm_output_index]) outside [v_min, v_max].
      - angle_diff_weight              : Penalty for predicted Va angle-difference outside line [theta_min, theta_max].
      - line_flow_weight               : Penalty for DC-approximate branch flow (DeltaVa / x_ij) exceeding rate_a.
      - line_flow_slack               : Tolerance subtracted from rate_a before penalising, absorbing the
                                        small linearisation error of the DC approximation on AC-feasible
                                        solutions.  Default 1e-4 (one decade above the ~1.3e-5 residual
                                        observed on pglib_opf_case10000_goc ground-truth data).

    Each raw penalty is normalized by a per-term exponential moving average (EMA)
    before the weight is applied.  This keeps every term near unit scale and makes
    the weights directly comparable to the task loss, regardless of the raw
    physical magnitudes (radians, per-unit power, etc.).
      - ema_momentum  (default 0.1): EMA decay.  Smaller = slower adaptation.

    Curriculum scheduling: domain-loss weights are ramped up gradually so the
    model first converges on the task loss before physics constraints are enforced.
      - warmup_epochs  (default 0): epochs with zero domain-loss weight.
      - ramp_epochs    (default 0): epochs over which weights linearly increase
                                    from 0 to their configured values.
    Example: warmup_epochs=3, ramp_epochs=3 with num_epoch=10 means:
      epochs 0-2: no domain loss, epochs 3-5: linear ramp, epochs 6-9: full weight.

    Feature-index conventions (derived from the gridopt/PyG OPFDataset schema):
      bus targets  : [Va (0), Vm (1)]
      ac_line attrs: [theta_min(0), theta_max(1), r_from(2), r_to(3), b_sh(4), x(5), rate_a(6), ...]
      transformer  : [theta_min(0), theta_max(1), r(2), x(3), rate_a(4), ...]
    """

    def __init__(self, config: dict | None = None, node_target_type: str = "bus"):
        cfg = copy.deepcopy(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.node_target_type = node_target_type
        self.voltage_bound_weight = float(cfg.get("voltage_bound_weight", 0.0))
        self.voltage_bound_feature_indices = cfg.get(
            "voltage_bound_feature_indices", None
        )
        # vm_output_index: index in bus_pred corresponding to voltage magnitude (Vm).
        # Default is 1 — bus targets are [Va, Vm] in the OPFDataset schema.
        self.voltage_output_index = int(cfg.get("voltage_output_index", 1))
        # va_output_index: index in bus_pred corresponding to voltage angle (Va).
        self.va_output_index = int(cfg.get("va_output_index", 0))
        self.angle_diff_weight = float(cfg.get("angle_diff_weight", 0.0))
        self.line_flow_weight = float(cfg.get("line_flow_weight", 0.0))
        # line_flow_slack: a small tolerance subtracted from rate_a before the DC thermal-limit
        # penalty is evaluated.  It exists because the DC power-flow formula
        #   P_ij = (Va_i - Va_j) / x_ij
        # is a linearisation of the full AC power-flow equations.  Even when the OPF solver
        # produces a strictly AC-feasible solution, the DC approximation introduces a residual
        # of ~1e-5 p.u. (empirically measured on pglib_opf_case10000_goc ground-truth data:
        # mean ~1.3e-5, max ~1.7e-5).  Without a slack the penalty is non-zero on ground truth,
        # which means the gradient incorrectly penalises physically correct predictions.
        # The default 1e-4 is one decade above the observed noise floor — large enough to zero
        # out the DC-approximation artefact but small enough to still penalise real violations.
        self.line_flow_slack = float(cfg.get("line_flow_slack", 1e-4))
        # EMA state for per-term scale normalization.
        self._ema_momentum = float(cfg.get("ema_momentum", 0.1))
        self._penalty_ema: dict[str, float] = {}
        # Curriculum scheduling.
        self.warmup_epochs = int(cfg.get("warmup_epochs", 0))
        self.ramp_epochs = int(cfg.get("ramp_epochs", 0))

        if self.voltage_bound_feature_indices is not None:
            if len(self.voltage_bound_feature_indices) != 2:
                raise RuntimeError(
                    "DomainLoss.voltage_bound_feature_indices must be [vmin_idx, vmax_idx]."
                )
            self.voltage_bound_feature_indices = tuple(
                int(v) for v in self.voltage_bound_feature_indices
            )


    def _curriculum_scale(self) -> float:
        """Return a [0, 1] multiplier for domain-loss weights based on current epoch.

        Reads os.environ["HYDRAGNN_EPOCH"] set by the HydraGNN training loop each
        epoch — no changes to shared training code are needed.
          - epoch < warmup_epochs          -> 0.0  (task-loss only)
          - warmup_epochs <= epoch < warmup + ramp -> linear ramp 0.0 -> 1.0
          - epoch >= warmup + ramp_epochs  -> 1.0  (full weight)
        """
        if self.warmup_epochs == 0 and self.ramp_epochs == 0:
            return 1.0
        try:
            epoch = int(os.environ.get("HYDRAGNN_EPOCH", "0"))
        except (ValueError, TypeError):
            return 1.0
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        progress = (epoch - self.warmup_epochs) / self.ramp_epochs
        return float(min(progress, 1.0))

    def _normalize(self, name: str, raw: torch.Tensor) -> torch.Tensor:
        """Normalize *raw* by its EMA so that the effective scale ≈ 1.0 on average.

        On the first call the EMA is seeded with the raw value, returning 1.0
        (or near-1.0 for non-zero values).  Subsequent calls use the smoothed
        estimate so the normalization adapts gradually as training progresses.
        """
        val = float(raw.detach())
        if name not in self._penalty_ema:
            # Seed: ema = raw value, normalized output = 1.0 on first step.
            self._penalty_ema[name] = max(val, 1e-8)
        else:
            m = self._ema_momentum
            self._penalty_ema[name] = max(
                m * val + (1.0 - m) * self._penalty_ema[name], 1e-8
            )
        # Floor at 1e-8 prevents division by zero when a penalty term is exactly zero
        # (e.g. the constraint is already satisfied for all samples in a batch).
        return raw / self._penalty_ema[name]

    def __call__(self, pred, value, head_index, data):
        if not self.enabled or data is None:
            return value.new_zeros(()), {}

        if self.node_target_type != "bus":
            return value.new_zeros(()), {}
        if not hasattr(data, "node_types") or "bus" not in data.node_types:
            return value.new_zeros(()), {}
        if len(pred) == 0:
            return value.new_zeros(()), {}

        bus_pred = pred[0]
        if bus_pred.dim() == 1:
            bus_pred = bus_pred.unsqueeze(-1)
        bus_true = value[head_index[0]]
        if bus_true.shape != bus_pred.shape:
            bus_true = bus_true.reshape_as(bus_pred)
        bus_true = bus_true.to(bus_pred.device)

        total_penalty = bus_pred.new_zeros(())
        metrics = {}
        curriculum = self._curriculum_scale()
        metrics["opf_curriculum_scale"] = torch.tensor(curriculum)

        if curriculum == 0.0:
            metrics["opf_domain_total"] = total_penalty.detach()
            return total_penalty, metrics

        if (
            self.voltage_bound_weight > 0.0
            and self.voltage_bound_feature_indices is not None
            and hasattr(data["bus"], "x")
        ):
            vmin_idx, vmax_idx = self.voltage_bound_feature_indices
            bus_x = data["bus"].x
            if bus_x.dim() >= 2 and bus_x.shape[1] > max(vmin_idx, vmax_idx):
                lower = bus_x[:, vmin_idx].reshape(-1)
                upper = bus_x[:, vmax_idx].reshape(-1)
                voltage = bus_pred[:, self.voltage_output_index].reshape(-1)
                # F.relu zeros out values that already satisfy the bound, so the gradient
                # is zero for feasible predictions and proportional to the violation otherwise.
                # Squaring gives a smooth (C1) penalty with growing gradient for larger violations.
                bound_penalty = torch.mean(
                    F.relu(lower - voltage).pow(2)
                    + F.relu(voltage - upper).pow(2)
                )
                total_penalty = (
                    total_penalty + curriculum * self.voltage_bound_weight * self._normalize("voltage_bound", bound_penalty)
                )
                metrics["opf_voltage_bound"] = bound_penalty.detach()

        # ── Angle difference limit penalty ──────────────────────────────────
        # Penalise predicted Va angle-differences that violate per-line bounds.
        #   ac_line  edge_attr: [theta_min(0), theta_max(1), ...]
        #   transformer edge_attr: [theta_min(0), theta_max(1), ...]
        if self.angle_diff_weight > 0.0 and bus_pred.shape[-1] > self.va_output_index:
            Va = bus_pred[:, self.va_output_index].reshape(-1)
            for rel, rel_tag in [
                (("bus", "ac_line", "bus"), "ac"),
                (("bus", "transformer", "bus"), "tr"),
            ]:
                if rel not in data.edge_types:
                    continue
                ea = getattr(data[rel], "edge_attr", None)
                ei = getattr(data[rel], "edge_index", None)
                if ea is None or ei is None or ea.numel() == 0 or ea.shape[1] < 2:
                    continue
                theta_min = ea[:, 0].to(Va.device)
                theta_max = ea[:, 1].to(Va.device)
                src, dst = ei
                delta_theta = Va[src] - Va[dst]
                # Same relu-squared form as voltage_bound: zero gradient inside the
                # feasible region [theta_min, theta_max], growing penalty outside it.
                # No slack is needed here: verified empirically that this term is exactly
                # zero on OPFDataset ground-truth solutions (Va and theta bounds share units).
                angdiff_p = torch.mean(
                    F.relu(delta_theta - theta_max).pow(2)
                    + F.relu(theta_min - delta_theta).pow(2)
                )
                total_penalty = total_penalty + curriculum * self.angle_diff_weight * self._normalize(f"{rel_tag}_angle_diff", angdiff_p)
                metrics[f"opf_{rel_tag}_angle_diff"] = angdiff_p.detach()

        # ── DC thermal limit penalty ─────────────────────────────────────────
        # Penalise approximate DC branch flows that exceed the thermal limit.
        #   P_ij = (Va_i - Va_j) / x_ij   (DC power flow approximation)
        #   ac_line:     x = edge_attr[:,5], rate_a = edge_attr[:,6]
        #   transformer: x = edge_attr[:,3], rate_a = edge_attr[:,4]
        if self.line_flow_weight > 0.0 and bus_pred.shape[-1] > self.va_output_index:
            Va = bus_pred[:, self.va_output_index].reshape(-1)
            for rel, x_idx, ra_idx, rel_tag in [
                (("bus", "ac_line", "bus"),    5, 6, "ac"),
                (("bus", "transformer", "bus"), 3, 4, "tr"),
            ]:
                if rel not in data.edge_types:
                    continue
                ea = getattr(data[rel], "edge_attr", None)
                ei = getattr(data[rel], "edge_index", None)
                if ea is None or ei is None or ea.numel() == 0 or ea.shape[1] <= max(x_idx, ra_idx):
                    continue
                # clamp x_ij away from zero to avoid division-by-zero in the DC formula;
                # 1e-6 p.u. is several orders of magnitude below any physical reactance.
                x_ij   = ea[:, x_idx].to(Va.device).clamp(min=1e-6)
                # clamp rate_a to be non-negative; negative thermal limits are nonsensical
                # and could arise from edge cases in dataset normalisation.
                rate_a = ea[:, ra_idx].to(Va.device).clamp(min=0.0)
                src, dst = ei
                # DC power-flow approximation: P_ij ≈ (Va_i - Va_j) / x_ij  [per unit].
                # This linearises the full AC formula sin(Va_i - Va_j) / x_ij and is only
                # exact in the flat-voltage, small-angle limit.
                P_ij = (Va[src] - Va[dst]) / x_ij
                # line_flow_slack is subtracted from rate_a to absorb the residual introduced
                # by the DC linearisation on AC-feasible solutions (see __init__ for details).
                # Without it, ground-truth predictions would incur a spurious non-zero penalty.
                flow_p = torch.mean(F.relu(P_ij.abs() - rate_a - self.line_flow_slack).pow(2))
                total_penalty = total_penalty + curriculum * self.line_flow_weight * self._normalize(f"{rel_tag}_line_flow", flow_p)
                metrics[f"opf_{rel_tag}_line_flow"] = flow_p.detach()

        metrics["opf_domain_total"] = total_penalty.detach()
        return total_penalty, metrics


class OPFEnhancedModelWrapper(torch.nn.Module):
    """Compose OPF-specific auxiliary loss around an existing HydraGNN model.

    In addition to combining the task loss and domain loss, this wrapper
    accumulates per-batch values during each epoch and prints a one-line
    breakdown at the end of that epoch (on rank 0 only).  The breakdown
    shows the task-driven loss and each individual domain-loss term
    separately, making it straightforward to diagnose whether the domain
    penalty is interfering with the data-driven objective.

    Log format (one line appended to run.log per epoch on rank 0):
      DomainBreakdown epoch=XX task=X.XXXXXXXX domain_total=X.XXXXXXXX \
          curriculum=X.XX voltage_bound=X.XXXXXXXX ac_angle_diff=X.XXXXXXXX ...
    """

    def __init__(self, original_model, domain_loss: OPFDomainLoss):
        super().__init__()
        self.model = original_model
        self.domain_loss = domain_loss
        self._last_batch = None
        self.last_extra_loss_metrics = {}
        # Per-epoch accumulation state.
        # Keyed by metric name; values are (sum, count) pairs for computing means.
        self._epoch_accum: dict[str, list[float]] = {}
        self._epoch_accum_task: list[float] = []
        self._last_seen_epoch: int = -1

    def _flush_epoch_log(self, epoch: int) -> None:
        """Log the mean task-loss and domain-loss breakdown for *epoch* on rank 0.

        Called automatically at the first batch of a new epoch so the previous
        epoch's accumulated statistics are written before training continues.

        Log format (one line per epoch in run.log, rank 0 only)::

          LossBreakdown epoch=XX \
              data_driven_mse=X.XXXXXXXX \
              physics_penalty_total=X.XXXXXXXX \
              curriculum_scale=X.XX \
              raw_voltage_bound=X.XXXXXXXX \
              raw_ac_angle_diff=X.XXXXXXXX \
              raw_tr_angle_diff=X.XXXXXXXX \
              raw_ac_line_flow=X.XXXXXXXX

        Field meanings:
          data_driven_mse        -- MSE between model predictions and OPF ground-truth
                                    targets (the standard HydraGNN task loss, no physics).
          physics_penalty_total  -- weighted, EMA-normalised sum of all feasibility
                                    penalties (voltage bound + angle diff + DC flow).
                                    This is what is added to data_driven_mse during
                                    back-propagation.  Should stay well below
                                    data_driven_mse for the task signal to dominate.
          curriculum_scale       -- ramp factor in [0, 1]; 0 during warmup, 1 at full
                                    weight.  physics_penalty_total = 0 when this is 0.
          raw_*                  -- raw (unweighted, un-normalised) value of each
                                    individual feasibility penalty.  Zero on any strictly
                                    feasible OPF solution; non-zero indicates the current
                                    prediction violates that constraint.
        """
        # Only log from rank 0 to avoid duplicate lines in the shared run.log.
        if dist.is_initialized() and dist.get_rank() != 0:
            self._epoch_accum.clear()
            self._epoch_accum_task.clear()
            return
        if not self._epoch_accum_task:
            return  # nothing accumulated yet (e.g. first call before any batch)

        n = len(self._epoch_accum_task)
        task_mean = sum(self._epoch_accum_task) / n

        # Map internal metric keys to self-explaining log field names.
        _key_labels = {
            "opf_domain_total":      "physics_penalty_total",
            "opf_curriculum_scale":  "curriculum_scale",
            "opf_voltage_bound":     "raw_voltage_bound",
            "opf_ac_angle_diff":     "raw_ac_angle_diff",
            "opf_tr_angle_diff":     "raw_tr_angle_diff",
            "opf_ac_line_flow":      "raw_ac_line_flow",
            "opf_tr_line_flow":      "raw_tr_line_flow",
        }

        parts = [f"epoch={epoch:02d}", f"data_driven_mse={task_mean:.8f}"]
        for key in sorted(self._epoch_accum):
            vals = self._epoch_accum[key]
            mean_val = sum(vals) / len(vals)
            label = _key_labels.get(key, key.removeprefix("opf_"))
            parts.append(f"{label}={mean_val:.8f}")

        logging.info("LossBreakdown " + "  ".join(parts))

        # Reset accumulators for the next epoch.
        self._epoch_accum.clear()
        self._epoch_accum_task.clear()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, data):
        self._last_batch = data
        return self.model(data)

    def loss(self, pred, value, head_index):
        total_loss, tasks_loss = self.model.loss(pred, value, head_index)
        if self._last_batch is None:
            info(
                "[OPFEnhancedModelWrapper] loss() called before forward(); "
                "domain penalty will be zero for this batch.",
                logtype="warning",
            )
        extra_loss, extra_metrics = self.domain_loss(
            pred,
            value,
            head_index,
            self._last_batch,
        )
        self.last_extra_loss_metrics = extra_metrics

        # ── Per-epoch accumulation ───────────────────────────────────────────
        # Detect epoch transitions using HYDRAGNN_EPOCH (set by the core training
        # loop).  On each new epoch, flush the previous epoch's accumulated stats
        # to logging.info so they appear in run.log alongside the Epoch: XX line.
        try:
            current_epoch = int(os.environ.get("HYDRAGNN_EPOCH", "-1"))
        except (ValueError, TypeError):
            current_epoch = -1

        if current_epoch != self._last_seen_epoch and self._last_seen_epoch >= 0:
            # Epoch boundary: flush accumulated stats for the completed epoch.
            self._flush_epoch_log(self._last_seen_epoch)
        self._last_seen_epoch = current_epoch

        # Accumulate task loss (total_loss is the data-driven term before domain is added).
        self._epoch_accum_task.append(float(total_loss.detach()))
        # Accumulate each domain metric (raw, un-normalized values for interpretability).
        for key, val in extra_metrics.items():
            self._epoch_accum.setdefault(key, []).append(float(val))

        return total_loss + extra_loss, tasks_loss


def build_solution_target(data, node_target_type: str):
    """Extract the solution target tensor for the given node type."""
    if hasattr(data, "node_types") and node_target_type in data.node_types:
        node_store = data[node_target_type]
        if not hasattr(node_store, "y") or node_store.y is None:
            raise RuntimeError(
                f"No targets found for node type '{node_target_type}' in OPF sample."
            )
        return node_store.y.to(torch.float32)

    if hasattr(data, "_node_type_names") and hasattr(data, "node_type"):
        if node_target_type not in data._node_type_names:
            raise RuntimeError(
                f"Node type '{node_target_type}' not found in OPF sample."
            )
        type_index = data._node_type_names.index(node_target_type)
        if not hasattr(data, "y") or data.y is None:
            raise RuntimeError(
                f"No homogeneous targets found for node type '{node_target_type}'."
            )
        mask = data.node_type == type_index
        return data.y[mask].to(torch.float32)

    raise RuntimeError(f"Node type '{node_target_type}' not found in OPF sample.")


def ensure_node_y_loc(data):
    if not hasattr(data, "y") or data.y is None:
        raise RuntimeError("Missing node targets (data.y) for OPF sample.")
    if data.y.dim() == 1:
        data.y = data.y.unsqueeze(-1)
    num_nodes = int(data.y.shape[0])
    target_dim = int(data.y.shape[1])
    data.y_num_nodes = torch.tensor(
        [num_nodes], dtype=torch.int64, device=data.y.device
    )
    data.y_loc = torch.tensor(
        [[0, num_nodes * target_dim]],
        dtype=torch.int64,
        device=data.y.device,
    )


def resolve_node_target_type(data, requested: str) -> str:
    if hasattr(data, "node_types"):
        if requested in data.node_types:
            return requested
        if hasattr(data, "_node_type_names") and requested in data._node_type_names:
            idx = data._node_type_names.index(requested)
            if idx < len(data.node_types):
                return data.node_types[idx]
        raise RuntimeError(
            f"Requested node_target_type '{requested}' not found in data. "
            f"Available node types: {list(data.node_types)}."
        )
    if hasattr(data, "_node_type_names") and requested in data._node_type_names:
        return requested
    raise RuntimeError(
        f"Cannot resolve node_target_type '{requested}': data has no node_types."
    )


def _as_edge_feature(value, num_edges: int, device):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except Exception:
            return None
    if value.numel() == 0:
        return None
    if value.dim() == 0:
        return None
    if value.dim() == 1:
        if int(value.shape[0]) != int(num_edges):
            return None
        value = value.view(-1, 1)
    elif value.dim() >= 2:
        if int(value.shape[0]) != int(num_edges):
            return None
        value = value.reshape(num_edges, -1)
    if value.dtype not in (torch.float16, torch.float32, torch.float64):
        value = value.to(torch.float32)
    return value.to(device=device, dtype=torch.float32)


def resolve_edge_feature_schema(
    configured_feature_names=None,
    configured_edge_dim=None,
):
    if configured_feature_names is None or len(configured_feature_names) == 0:
        raise RuntimeError(
            "edge_feature_names must be explicitly provided in the config. "
            "No implicit defaults are used."
        )
    schema = [str(name) for name in configured_feature_names if str(name).strip()]
    if not schema:
        raise RuntimeError("edge_feature_names contains only empty/whitespace entries.")
    if configured_edge_dim is not None:
        edge_dim = int(configured_edge_dim)
        if edge_dim != len(schema):
            raise RuntimeError(
                f"edge_dim={edge_dim} does not match the number of "
                f"edge_feature_names ({len(schema)}). They must be equal."
            )
    return tuple(schema)


def validate_voi_node_features(config: dict, node_target_type: str | None = None):
    """Validate that node feature config is fully specified.  Crash on anything missing."""
    nn_config = config.get("NeuralNetwork")
    if nn_config is None:
        raise RuntimeError("Config is missing 'NeuralNetwork' section.")
    var_config = nn_config.get("Variables_of_interest")
    if var_config is None:
        raise RuntimeError("Config is missing 'NeuralNetwork.Variables_of_interest'.")

    input_node_features = var_config.get("input_node_features")
    if not isinstance(input_node_features, list) or len(input_node_features) == 0:
        raise RuntimeError(
            "'input_node_features' must be an explicit non-empty list in the config."
        )

    node_feature_dims = var_config.get("node_feature_dims")
    if not isinstance(node_feature_dims, list) or len(node_feature_dims) == 0:
        raise RuntimeError(
            "'node_feature_dims' must be an explicit non-empty list in the config."
        )

    if "node_feature_names" not in var_config:
        raise RuntimeError(
            "'node_feature_names' must be explicitly provided in the config."
        )

    return config


def compute_pna_deg_for_hetero_dataset(dataset, verbosity: int = 2):
    from hydragnn.utils.print.print_utils import iterate_tqdm

    num_samples = len(dataset)
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        start = (num_samples * rank) // world_size
        end = (num_samples * (rank + 1)) // world_size
    else:
        start = 0
        end = num_samples

    local_indices = range(start, end)

    max_deg_local = 0
    for idx in iterate_tqdm(local_indices, verbosity, desc="HeteroPNA degree max"):
        data = dataset[idx]
        data_h = data.to_homogeneous(add_node_type=True, add_edge_type=True)
        d = degree(data_h.edge_index[1], num_nodes=data_h.num_nodes, dtype=torch.long)
        if d.numel() > 0:
            max_deg_local = max(max_deg_local, int(d.max().item()))

    if dist.is_initialized():
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        max_deg_tensor = torch.tensor(
            [max_deg_local], dtype=torch.long, device=reduce_device
        )
        dist.all_reduce(max_deg_tensor, op=dist.ReduceOp.MAX)
        max_deg = int(max_deg_tensor.item())
    else:
        max_deg = max_deg_local

    deg_local = torch.zeros(max_deg + 1, dtype=torch.long)
    for idx in iterate_tqdm(local_indices, verbosity, desc="HeteroPNA degree bincount"):
        data = dataset[idx]
        data_h = data.to_homogeneous(add_node_type=True, add_edge_type=True)
        d = degree(data_h.edge_index[1], num_nodes=data_h.num_nodes, dtype=torch.long)
        deg_local += torch.bincount(d, minlength=deg_local.numel())

    if dist.is_initialized():
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        deg_tensor = deg_local.to(device=reduce_device)
        dist.all_reduce(deg_tensor, op=dist.ReduceOp.SUM)
        deg = deg_tensor.cpu()
    else:
        deg = deg_local

    return deg.tolist()


def _assemble_edge_attr_hetero(data, edge_dim_dict):
    """Heterogeneous route: keep per-edge-type native widths.

    Edge types whose relation name appears in *edge_dim_dict* must carry a
    pre-assembled ``edge_attr`` tensor with the declared width.  Edge types
    absent from the dict are treated as featureless — any stale ``edge_attr``
    is removed so that ``data.edge_attr_dict`` only contains featured types.

    Returns ``(data, edge_dim_dict)`` unchanged.
    """
    for edge_type in data.edge_types:
        _, rel, _ = edge_type
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue

        expected_dim = edge_dim_dict.get(rel)
        existing = getattr(edge_store, "edge_attr", None)

        if expected_dim is None:
            # Featureless — remove any edge_attr so it stays out of
            # data.edge_attr_dict during training.
            if existing is not None:
                try:
                    delattr(edge_store, "edge_attr")
                except AttributeError:
                    pass
            continue

        if not isinstance(existing, torch.Tensor) or existing.dim() != 2:
            raise RuntimeError(
                f"Edge type {edge_type} (rel={rel}) expects edge_attr with "
                f"{expected_dim} columns but found no valid 2-D tensor."
            )
        if existing.size(1) != expected_dim:
            raise RuntimeError(
                f"Edge type {edge_type} (rel={rel}) has edge_attr width "
                f"{existing.size(1)}, expected {expected_dim} from edge_dim config."
            )

    return data, edge_dim_dict


def assemble_edge_attr(data, edge_dim, feature_schema=None):
    """One-time assembly during preprocessing.

    *edge_dim* determines the route:

    * **int** — *homogeneous* route.  Every edge type is zero-padded (or
      assembled from named columns via *feature_schema*) to a uniform width
      equal to *edge_dim*.
    * **dict** — *heterogeneous* route.  Keys are relation names (the middle
      element of an edge-type triple); values are the expected widths of
      pre-assembled ``edge_attr`` tensors.  Edge types absent from the dict
      are treated as featureless.

    Returns ``(data, edge_dim)``.
    """
    if not hasattr(data, "edge_types"):
        return data, edge_dim

    if isinstance(edge_dim, dict):
        return _assemble_edge_attr_hetero(data, edge_dim)

    target_dim = int(edge_dim)
    if target_dim <= 0:
        raise RuntimeError("int edge_dim must be positive.")

    schema = None
    if feature_schema is not None:
        schema = tuple(str(n) for n in feature_schema if str(n).strip())
        if not schema:
            schema = None

    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))
        device = edge_index.device

        # Already assembled — accept as-is or zero-pad to target_dim.
        existing = getattr(edge_store, "edge_attr", None)
        if (
            isinstance(existing, torch.Tensor)
            and existing.dim() == 2
            and existing.size(0) == num_edges
        ):
            w = existing.size(1)
            if w == target_dim:
                continue  # exact match
            if w < target_dim:
                pad = torch.zeros(
                    num_edges, target_dim - w, device=device, dtype=existing.dtype
                )
                data[edge_type].edge_attr = torch.cat([existing, pad], dim=1)
                continue
            raise RuntimeError(
                f"edge_attr for {edge_type} has {w} columns, exceeding edge_dim={target_dim}."
            )

        # Try named-column assembly if a schema was provided.
        if schema is not None:
            has_any = any(
                getattr(edge_store, name, None) is not None for name in schema
            )
            if not has_any and existing is None:
                data[edge_type].edge_attr = torch.zeros(
                    num_edges, target_dim, device=device, dtype=torch.float32
                )
                continue

            cols = []
            for attr_name in schema:
                col = _as_edge_feature(
                    getattr(edge_store, attr_name, None), num_edges, device
                )
                if col is None:
                    raise RuntimeError(
                        f"Missing or invalid edge attribute '{attr_name}' "
                        f"for edge type {edge_type}."
                    )
                if int(col.shape[1]) != 1:
                    raise RuntimeError(
                        f"Edge attribute '{attr_name}' for edge type {edge_type} has "
                        f"{int(col.shape[1])} columns; expected exactly 1."
                    )
                cols.append(col)

            data[edge_type].edge_attr = torch.cat(cols, dim=1).contiguous()

            for attr_name in schema:
                try:
                    delattr(edge_store, attr_name)
                except AttributeError:
                    pass
            continue

        # No schema and no existing tensor — zero-fill.
        if existing is None:
            data[edge_type].edge_attr = torch.zeros(
                num_edges, target_dim, device=device, dtype=torch.float32
            )

    return data, target_dim


def _validate_edge_attr_hetero(data, edge_dim_dict):
    """Check per-edge-type widths for the heterogeneous route."""
    for edge_type in data.edge_types:
        _, rel, _ = edge_type
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))

        expected_dim = edge_dim_dict.get(rel)
        edge_attr = getattr(edge_store, "edge_attr", None)

        if expected_dim is None:
            # Featureless — must NOT have edge_attr.
            if isinstance(edge_attr, torch.Tensor):
                raise RuntimeError(
                    f"Featureless edge type {edge_type} (rel={rel}) should not "
                    f"have edge_attr, but found tensor with shape {list(edge_attr.shape)}."
                )
            continue

        if not isinstance(edge_attr, torch.Tensor):
            raise RuntimeError(
                f"Edge type {edge_type} (rel={rel}) is missing edge_attr; "
                f"expected width {expected_dim}."
            )
        if edge_attr.dim() != 2:
            raise RuntimeError(
                f"edge_attr for edge type {edge_type} has {edge_attr.dim()} "
                f"dimensions; expected 2."
            )
        if edge_attr.size(0) != num_edges:
            raise RuntimeError(
                f"edge_attr row count mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(0)}, expected {num_edges}."
            )
        if edge_attr.size(1) != expected_dim:
            raise RuntimeError(
                f"edge_attr dim mismatch for edge type {edge_type} (rel={rel}): "
                f"got {edge_attr.size(1)}, expected {expected_dim}."
            )

    return data


def validate_edge_attr(data, edge_dim):
    """Validate that every edge type carries properly shaped ``edge_attr``.

    *edge_dim* can be:

    * **int** — every edge type must have ``edge_attr`` with that many columns
      (featureless types that have no ``edge_attr`` are silently skipped).
    * **dict** — per-relation-name widths; featureless types (absent from the
      dict) must NOT carry ``edge_attr``.
    """
    if not hasattr(data, "edge_types"):
        return data

    if isinstance(edge_dim, dict):
        return _validate_edge_attr_hetero(data, edge_dim)

    target_dim = int(edge_dim)

    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))

        edge_attr = getattr(edge_store, "edge_attr", None)
        if not isinstance(edge_attr, torch.Tensor):
            continue
        if edge_attr.dim() != 2:
            raise RuntimeError(
                f"edge_attr for edge type {edge_type} has "
                f"{edge_attr.dim()} dimensions; expected 2."
            )
        if edge_attr.size(0) != num_edges:
            raise RuntimeError(
                f"edge_attr row count mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(0)}, expected {num_edges}."
            )
        if edge_attr.size(1) != target_dim:
            raise RuntimeError(
                f"edge_attr dim mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(1)}, expected {target_dim}."
            )

    return data


class HeteroFromHomogeneousDataset:
    """Wraps an ADIOS-loaded homogeneous dataset, converting each sample to
    heterogeneous and validating ``edge_attr`` shape.
    """

    def __init__(self, base, edge_dim: int):
        self.base = base
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        hetero = data.to_heterogeneous()
        if hasattr(data, "y"):
            hetero.y = data.y
        if hasattr(data, "graph_attr"):
            hetero.graph_attr = data.graph_attr
        validate_edge_attr(hetero, self.edge_dim)
        return hetero


class EdgeAttrDatasetAdapter:
    """Validates ``edge_attr`` on every access — no assembly, just shape check."""

    def __init__(self, base, edge_dim: int):
        self.base = base
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        validate_edge_attr(data, self.edge_dim)
        return data

    def __getattr__(self, name):
        return getattr(self.base, name)


class NodeTargetDatasetAdapter:
    def __init__(self, base, node_target_type: str, edge_dim: int):
        self.base = base
        self.node_target_type = node_target_type
        self.edge_dim = edge_dim

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        validate_edge_attr(data, self.edge_dim)
        if (
            not hasattr(data, "node_types")
            or self.node_target_type not in data.node_types
        ):
            raise RuntimeError(
                f"Node type '{self.node_target_type}' not found in OPF sample."
            )
        if (
            not hasattr(data[self.node_target_type], "y")
            or data[self.node_target_type].y is None
        ):
            raise RuntimeError(
                f"No targets found for node type '{self.node_target_type}' in OPF sample."
            )
        data.y = data[self.node_target_type].y
        ensure_node_y_loc(data)
        return data

    def __getattr__(self, name):
        return getattr(self.base, name)


class NodeBatchAdapter:
    def __init__(self, loader, node_target_type: str, edge_dim: int):
        self.loader = loader
        self.node_target_type = node_target_type
        self.edge_dim = edge_dim
        self.dataset = loader.dataset
        self.sampler = getattr(loader, "sampler", None)

    def __iter__(self):
        for data in self.loader:
            validate_edge_attr(data, self.edge_dim)
            if (
                not hasattr(data, "node_types")
                or self.node_target_type not in data.node_types
            ):
                raise RuntimeError(
                    f"Node type '{self.node_target_type}' not found in OPF sample."
                )

            if not hasattr(data, "batch"):
                node_store = data[self.node_target_type]
                if hasattr(node_store, "batch"):
                    data.batch = node_store.batch
                elif (
                    hasattr(data, "batch_dict")
                    and self.node_target_type in data.batch_dict
                ):
                    data.batch = data.batch_dict[self.node_target_type]
                else:
                    raise RuntimeError(
                        f"Cannot find batch vector for node type "
                        f"'{self.node_target_type}' in batched OPF data."
                    )

            if (
                not hasattr(data[self.node_target_type], "y")
                or data[self.node_target_type].y is None
            ):
                raise RuntimeError(
                    f"No targets found for node type '{self.node_target_type}' in OPF sample."
                )
            data.y = data[self.node_target_type].y
            ensure_node_y_loc(data)
            yield data

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)
