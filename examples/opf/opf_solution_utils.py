"""Shared utilities for OPF solution workflows (heterogeneous and homogeneous)."""

import copy
import logging
import os
import statistics
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import degree

from hydragnn.utils.input_config_parsing import get_variable_schema


def info(*args, logtype="info", sep=" "):
    getattr(logging, logtype)(sep.join(map(str, args)))


class OPFDomainLoss(torch.nn.Module):
    """Augmented-Lagrangian enforcement of OPF inequality constraints.

    Feasibility penalties (all zero on any strictly feasible OPF solution):
      - voltage_bound_weight           : Penalty for Vm (bus_pred[:, vm_output_index]) outside [v_min, v_max].
      - angle_diff_weight              : Penalty for predicted Va angle-difference outside line [theta_min, theta_max].
      - line_flow_weight               : Penalty for full AC apparent-power branch flow (|S_ij|, |S_ji|,
                                        reconstructed from complex bus voltages via the standard pi-model
                                        branch equations) exceeding rate_a, matching the manuscript's
                                        thermal-limit equation exactly (no DC linearisation).
      - line_flow_slack               : Small numerical tolerance subtracted from rate_a before penalising,
                                        absorbing floating-point/near-binding-constraint noise on
                                        AC-feasible ground-truth solutions (the full AC formula itself
                                        introduces no linearisation bias, unlike the DC approximation it
                                        replaces). Default 1e-4.

    For each constraint family, the mean positive violation ``c >= 0`` contributes
    ``lambda*c + rho/2*c**2``. After every training epoch the non-negative dual is
    updated by projected ascent, ``lambda <- max(0, lambda + rho*c_bar)``. ``rho``
    can grow when the aggregate residual is not decreasing sufficiently.

    Curriculum scheduling: domain-loss weights are ramped up gradually so the
    model first converges on the task loss before physics constraints are enforced.
      - warmup_epochs  (default 0): epochs with zero domain-loss weight.
      - ramp_epochs    (default 0): epochs over which weights linearly increase
                                    from 0 to their configured values.
    Example: warmup_epochs=3, ramp_epochs=3 with num_epoch=10 means:
      epochs 0-2: no domain loss, epochs 3-5: linear ramp, epochs 6-9: full weight.

    Feature-index conventions (empirically verified against ground-truth pglib_opf_case14_ieee.m
    branch data — exact numeric cross-check of r, x, b, rate_a/100 against this project's own
    HDF5-converted edge_attr tensors; a previously-consulted internal docx had this schema wrong):
      bus targets  : [Va (0), Vm (1)]
      ac_line attrs: [angmin(0), angmax(1), b_fr(2), b_to(3), r(4), x(5), rate_a(6), rate_b(7), rate_c(8)]
      transformer  : [angmin(0), angmax(1), r(2), x(3), rate_a(4), rate_b(5), rate_c(6), tm(7), ...(8-10, unused/reserved)]
      (b_fr == b_to == total shunt susceptance / 2; rate_a/b/c are identical in this dataset;
       tm is the off-nominal turns ratio; transformers carry no shunt-susceptance term.)
    """

    def __init__(self, config: dict | None = None, node_target_type: str = "bus"):
        super().__init__()
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
        # line_flow_slack: a small numerical tolerance subtracted from rate_a before the
        # AC apparent-power thermal-limit penalty is evaluated. The full AC formula
        # (unlike the DC approximation it replaces) introduces no intrinsic linearisation
        # bias, but a small slack still absorbs floating-point noise and near-binding
        # constraints on ground-truth solutions. Default 1e-4.
        self.line_flow_slack = float(cfg.get("line_flow_slack", 1e-4))
        # line_flow_min_x: branches whose series-impedance magnitude |z| = sqrt(r^2+x^2)
        # falls below this floor are excluded from the thermal-limit penalty entirely
        # (rather than clamped). Some PGLib-OPF cases (e.g. case6470_rte, case4661_sdet,
        # case13659_pegase) contain near-zero or even negative reactance on certain
        # transformer/ac_line branches (likely ideal or phase-shifting transformers);
        # dividing by such a tiny admittance denominator blows up the reconstructed
        # apparent power to unphysical magnitudes even on ground-truth Va/Vm, dominating
        # the mean-of-squares statistic. Default 1e-3 p.u. sits below the smallest
        # impedance magnitude observed on well-behaved branches (~0.04 p.u. for ac_line,
        # ~0.21 p.u. for transformer on pglib_opf_case14_ieee).
        self.line_flow_min_x = float(cfg.get("line_flow_min_x", 1e-3))
        self.register_buffer("rho", torch.tensor(float(cfg.get("rho", 1.0))))
        self.rho_growth = float(cfg.get("rho_growth", 2.0))
        self.rho_max = float(cfg.get("rho_max", 1.0e4))
        self.constraint_reduction = float(cfg.get("constraint_reduction", 0.9))
        self.dual_update_interval = int(cfg.get("dual_update_interval", 1))
        self.register_buffer("previous_constraint_norm", torch.tensor(float("inf")))
        self._constraint_names = (
            "voltage_bound",
            "ac_angle_diff",
            "tr_angle_diff",
            "ac_line_flow",
            "tr_line_flow",
        )
        for name in self._constraint_names:
            self.register_buffer(f"dual_{name}", torch.tensor(0.0))
        self._constraint_sums = {name: 0.0 for name in self._constraint_names}
        self._constraint_counts = {name: 0 for name in self._constraint_names}
        # Curriculum scheduling.
        self.warmup_epochs = int(cfg.get("warmup_epochs", 0))
        self.ramp_epochs = int(cfg.get("ramp_epochs", 0))

        if self.rho <= 0 or self.rho_growth < 1 or self.rho_max < self.rho:
            raise ValueError(
                "DomainLoss requires 0 < rho <= rho_max and rho_growth >= 1."
            )
        if not 0 < self.constraint_reduction <= 1:
            raise ValueError("DomainLoss.constraint_reduction must be in (0, 1].")
        if self.dual_update_interval <= 0:
            raise ValueError("DomainLoss.dual_update_interval must be positive.")

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

    def _augmented_term(self, name, residual, scale, curriculum, update_state, metrics):
        scaled = float(scale) * residual
        if update_state:
            self._constraint_sums[name] += float(scaled.detach())
            self._constraint_counts[name] += 1
        dual = getattr(self, f"dual_{name}").to(
            device=scaled.device, dtype=scaled.dtype
        )
        metrics[f"opf_{name}"] = residual.detach()
        metrics[f"opf_{name}_dual"] = dual.detach()
        return curriculum * (dual * scaled + 0.5 * self.rho * scaled.square())

    @torch.no_grad()
    def update_multipliers(self, completed_epoch: int):
        """Apply one projected dual-ascent update from accumulated training residuals."""
        if completed_epoch < 0 or (completed_epoch + 1) % self.dual_update_interval:
            return
        if not any(self._constraint_counts.values()):
            return
        means = []
        for name in self._constraint_names:
            stats = self.rho.new_tensor(
                [self._constraint_sums[name], self._constraint_counts[name]]
            )
            if dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            mean = stats[0] / stats[1].clamp_min(1.0)
            means.append(mean)
            if stats[1] > 0:
                dual = getattr(self, f"dual_{name}")
                dual.copy_(torch.clamp_min(dual + self.rho * mean, 0.0))
            self._constraint_sums[name] = 0.0
            self._constraint_counts[name] = 0

        residual_norm = torch.linalg.vector_norm(torch.stack(means))
        if (
            torch.isfinite(self.previous_constraint_norm)
            and residual_norm
            > self.constraint_reduction * self.previous_constraint_norm
        ):
            self.rho.copy_(torch.clamp(self.rho * self.rho_growth, max=self.rho_max))
        self.previous_constraint_norm.copy_(residual_norm)

    def forward(self, pred, value, head_index, data, update_state=False):
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
                bound_residual = torch.mean(
                    F.relu(lower - voltage) + F.relu(voltage - upper)
                )
                total_penalty = total_penalty + self._augmented_term(
                    "voltage_bound",
                    bound_residual,
                    self.voltage_bound_weight,
                    curriculum,
                    update_state,
                    metrics,
                )

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
                angdiff_residual = torch.mean(
                    F.relu(delta_theta - theta_max) + F.relu(theta_min - delta_theta)
                )
                total_penalty = total_penalty + self._augmented_term(
                    f"{rel_tag}_angle_diff",
                    angdiff_residual,
                    self.angle_diff_weight,
                    curriculum,
                    update_state,
                    metrics,
                )

        # ── Full AC apparent-power thermal limit penalty ────────────────────
        # Reconstruct branch apparent power flow at both ends (S_ij, S_ji) from the
        # predicted complex bus voltages V = Vm * exp(j*Va), using the standard
        # pi-equivalent branch model (series admittance y = 1/(r+jx), shunt charging
        # susceptance b_fr/b_to, off-nominal turns ratio tm for transformers):
        #   Yff = (y + j*b_fr) / tm^2   Yft = -y / tm
        #   Ytf = -y / tm               Ytt =  y + j*b_to
        #   I_ij = Yff*Vi + Yft*Vj      I_ji = Ytf*Vi + Ytf*Vj  (from/to branch currents)
        #   S_ij = Vi * conj(I_ij)      S_ji = Vj * conj(I_ji)
        # penalising max(|S_ij|, |S_ji|) exceeding rate_a. This matches the manuscript's
        # thermal-limit equation exactly (no DC/small-angle linearisation).
        #   ac_line:     b_fr=edge_attr[:,2], b_to=edge_attr[:,3], r=edge_attr[:,4],
        #                x=edge_attr[:,5], rate_a=edge_attr[:,6], tm=1 (no tap)
        #   transformer: r=edge_attr[:,2], x=edge_attr[:,3], rate_a=edge_attr[:,4],
        #                tm=edge_attr[:,7], b_fr=b_to=0 (no shunt term in this schema)
        if self.line_flow_weight > 0.0 and bus_pred.shape[-1] > max(
            self.va_output_index, self.voltage_output_index
        ):
            Va = bus_pred[:, self.va_output_index].reshape(-1)
            Vm = bus_pred[:, self.voltage_output_index].reshape(-1)
            for rel, r_idx, x_idx, ra_idx, b_fr_idx, b_to_idx, tm_idx, rel_tag in [
                (("bus", "ac_line", "bus"), 4, 5, 6, 2, 3, None, "ac"),
                (("bus", "transformer", "bus"), 2, 3, 4, None, None, 7, "tr"),
            ]:
                if rel not in data.edge_types:
                    continue
                ea = getattr(data[rel], "edge_attr", None)
                ei = getattr(data[rel], "edge_index", None)
                needed_idx = [
                    i
                    for i in (r_idx, x_idx, ra_idx, b_fr_idx, b_to_idx, tm_idx)
                    if i is not None
                ]
                if (
                    ea is None
                    or ei is None
                    or ea.numel() == 0
                    or ea.shape[1] <= max(needed_idx)
                ):
                    continue
                r_raw = ea[:, r_idx].to(Va.device)
                x_raw = ea[:, x_idx].to(Va.device)
                # Exclude near-singular-admittance branches entirely rather than clamping:
                # y = 1/(r+jx) is undefined for |z| ~ 0, and clamping to a tiny floor
                # produces unphysical penalty spikes (see line_flow_min_x comment in __init__).
                z_mag = torch.sqrt(r_raw.pow(2) + x_raw.pow(2))
                keep = z_mag >= self.line_flow_min_x
                n_excluded = int((~keep).sum().item())
                metrics[f"opf_{rel_tag}_line_flow_n_excluded"] = n_excluded
                if not torch.any(keep):
                    metrics[f"opf_{rel_tag}_line_flow"] = bus_pred.new_zeros(())
                    continue
                r_ij = r_raw[keep]
                x_ij = x_raw[keep]
                # clamp rate_a to be non-negative; negative thermal limits are nonsensical
                # and could arise from edge cases in dataset normalisation.
                rate_a = ea[:, ra_idx].to(Va.device).clamp(min=0.0)[keep]
                b_fr = (
                    ea[:, b_fr_idx].to(Va.device)[keep]
                    if b_fr_idx is not None
                    else torch.zeros_like(r_ij)
                )
                b_to = (
                    ea[:, b_to_idx].to(Va.device)[keep]
                    if b_to_idx is not None
                    else torch.zeros_like(r_ij)
                )
                tm = (
                    ea[:, tm_idx].to(Va.device)[keep]
                    if tm_idx is not None
                    else torch.ones_like(r_ij)
                )
                src, dst = ei
                src, dst = src[keep], dst[keep]

                y = 1.0 / torch.complex(r_ij, x_ij)
                Yff = (y + 1j * b_fr) / tm.pow(2)
                Yft = -y / tm
                Ytf = -y / tm
                Ytt = y + 1j * b_to

                Vi = torch.polar(Vm[src], Va[src])
                Vj = torch.polar(Vm[dst], Va[dst])
                I_ij = Yff * Vi + Yft * Vj
                I_ji = Ytf * Vi + Ytt * Vj
                S_ij = Vi * torch.conj(I_ij)
                S_ji = Vj * torch.conj(I_ji)

                # line_flow_slack absorbs floating-point/near-binding-constraint noise;
                # the full AC formula itself introduces no linearisation bias.
                flow_residual = torch.mean(
                    F.relu(S_ij.abs() - rate_a - self.line_flow_slack)
                    + F.relu(S_ji.abs() - rate_a - self.line_flow_slack)
                )
                total_penalty = total_penalty + self._augmented_term(
                    f"{rel_tag}_line_flow",
                    flow_residual,
                    self.line_flow_weight,
                    curriculum,
                    update_state,
                    metrics,
                )

        metrics["opf_augmented_rho"] = self.rho.detach()
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

    def _flush_epoch_log(self, epoch: int, force: bool = False) -> None:
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

        *force* bypasses the "only rank 0" gate below (used by the one-shot
        --eval_domain_penalties_only path): under some launch configurations the
        MPI/SLURM rank that prints diagnostics (HYDRAGNN_DIAG_RANK) does not
        coincide with torch.distributed's rank 0, which otherwise silently
        suppresses this log line with no error. Passing force=True guarantees a
        LossBreakdown line is printed by whichever process has accumulated data,
        at the cost of possible duplicate lines across ranks in that one-shot mode
        (harmless -- there is no ongoing training loop relying on a single line).
        """
        # Only log from rank 0 to avoid duplicate lines in the shared run.log.
        if not force and dist.is_initialized() and dist.get_rank() != 0:
            self._epoch_accum.clear()
            self._epoch_accum_task.clear()
            return
        if not self._epoch_accum_task:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"0: LossBreakdown epoch={epoch:02d} WARNING: rank={rank} has an "
                "empty accumulator (0 batches seen) -- no breakdown to report.",
                flush=True,
            )
            return  # nothing accumulated yet (e.g. first call before any batch)

        n = len(self._epoch_accum_task)
        task_mean = sum(self._epoch_accum_task) / n

        # Map internal metric keys to self-explaining log field names.
        _key_labels = {
            "opf_domain_total": "physics_penalty_total",
            "opf_curriculum_scale": "curriculum_scale",
            "opf_augmented_rho": "augmented_rho",
            "opf_voltage_bound": "raw_voltage_bound",
            "opf_ac_angle_diff": "raw_ac_angle_diff",
            "opf_tr_angle_diff": "raw_tr_angle_diff",
            "opf_ac_line_flow": "raw_ac_line_flow",
            "opf_tr_line_flow": "raw_tr_line_flow",
            "opf_ac_line_flow_n_excluded": "ac_line_flow_n_excluded",
            "opf_tr_line_flow_n_excluded": "tr_line_flow_n_excluded",
        }
        # Line-flow terms are reported with a median alongside the mean: a handful
        # of near-zero-reactance branches (excluded from the penalty itself via
        # line_flow_min_x, but occasionally still present in edge cases) or genuinely
        # overloaded outlier branches can otherwise dominate the mean-of-squares.
        _median_keys = {
            "opf_ac_line_flow": "raw_ac_line_flow_median",
            "opf_tr_line_flow": "raw_tr_line_flow_median",
        }

        parts = [f"epoch={epoch:02d}", f"data_driven_mse={task_mean:.8f}"]
        for key in sorted(self._epoch_accum):
            vals = self._epoch_accum[key]
            mean_val = sum(vals) / len(vals)
            label = _key_labels.get(key, key.removeprefix("opf_"))
            parts.append(f"{label}={mean_val:.8f}")
            if key in _median_keys:
                parts.append(f"{_median_keys[key]}={statistics.median(vals):.8f}")

        # Use print rather than logging.info so the line is always visible in
        # run.log regardless of the logging level configured by HydraGNN.
        print("0: LossBreakdown " + "  ".join(parts), flush=True)

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

    def train(self, mode: bool = True):
        """Finalize training residuals before entering validation mode."""
        was_training = self.training
        result = super().train(mode)
        if was_training and not mode and self._last_seen_epoch >= 0:
            self.domain_loss.update_multipliers(self._last_seen_epoch)
        return result

    def loss(self, pred, value, head_index):
        total_loss, tasks_loss = self.model.loss(pred, value, head_index)
        if self._last_batch is None:
            info(
                "[OPFEnhancedModelWrapper] loss() called before forward(); "
                "domain penalty will be zero for this batch.",
                logtype="warning",
            )
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
            self.domain_loss.update_multipliers(self._last_seen_epoch)
            self._flush_epoch_log(self._last_seen_epoch)
        self._last_seen_epoch = current_epoch

        extra_loss, extra_metrics = self.domain_loss(
            pred,
            value,
            head_index,
            self._last_batch,
            update_state=self.training,
        )
        self.last_extra_loss_metrics = extra_metrics

        # Accumulate task loss (total_loss is the data-driven term before domain is added).
        self._epoch_accum_task.append(float(total_loss.detach()))
        # Accumulate each domain metric (raw, un-normalized values for interpretability).
        for key, val in extra_metrics.items():
            self._epoch_accum.setdefault(key, []).append(float(val))

        return total_loss + extra_loss, tasks_loss

    def finalize_domain_state(self):
        """Commit the final epoch's dual update before checkpointing."""
        self.domain_loss.update_multipliers(self._last_seen_epoch)


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


def validate_opf_variable_schema(config: dict, node_target_type: str | None = None):
    """Validate the canonical variable schema required by OPF workflows."""
    schema = get_variable_schema(config)
    if node_target_type is None:
        return config

    node_inputs = [
        spec
        for spec in schema.inputs
        if spec.level == "node" and spec.node_type == node_target_type
    ]
    if not node_inputs:
        raise RuntimeError(
            f"Variables.inputs has no entries for node_type '{node_target_type}'."
        )

    node_outputs = [
        spec
        for spec in schema.outputs
        if spec.level == "node" and spec.node_type == node_target_type
    ]
    if not node_outputs:
        raise RuntimeError(
            f"Variables.outputs has no entries for node_type '{node_target_type}'."
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

    Every edge-type triple must appear in *edge_dim_dict*. Positive dimensions
    require a pre-assembled ``edge_attr`` tensor with the declared width;
    dimension zero declares a featureless edge and removes stale attributes.

    Returns ``(data, edge_dim_dict)`` unchanged.
    """
    actual_edge_types = {tuple(edge_type) for edge_type in data.edge_types}
    configured_edge_types = set(edge_dim_dict)
    if actual_edge_types != configured_edge_types:
        raise RuntimeError(
            "Configured edge_types do not match data: "
            f"missing={sorted(actual_edge_types - configured_edge_types)}, "
            f"unexpected={sorted(configured_edge_types - actual_edge_types)}."
        )

    for edge_type in data.edge_types:
        edge_type = tuple(edge_type)
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue

        expected_dim = edge_dim_dict[edge_type]
        existing = getattr(edge_store, "edge_attr", None)

        if expected_dim == 0:
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
                f"Edge type {edge_type} expects edge_attr with "
                f"{expected_dim} columns but found no valid 2-D tensor."
            )
        if existing.size(1) != expected_dim:
            raise RuntimeError(
                f"Edge type {edge_type} has edge_attr width "
                f"{existing.size(1)}, expected {expected_dim} from edge_dim config."
            )

    return data, edge_dim_dict


def assemble_edge_attr(data, edge_dim, feature_schema=None):
    """One-time assembly during preprocessing.

    *edge_dim* determines the route:

    * **int** — *homogeneous* route.  Every edge type is zero-padded (or
      assembled from named columns via *feature_schema*) to a uniform width
      equal to *edge_dim*.
        * **dict** — *heterogeneous* route. Keys are complete edge-type triples and
            values are expected widths. Zero declares a featureless edge.

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
    actual_edge_types = {tuple(edge_type) for edge_type in data.edge_types}
    configured_edge_types = set(edge_dim_dict)
    if actual_edge_types != configured_edge_types:
        raise RuntimeError(
            "Configured edge_types do not match data: "
            f"missing={sorted(actual_edge_types - configured_edge_types)}, "
            f"unexpected={sorted(configured_edge_types - actual_edge_types)}."
        )

    for edge_type in data.edge_types:
        edge_type = tuple(edge_type)
        edge_store = data[edge_type]
        edge_index = getattr(edge_store, "edge_index", None)
        if not isinstance(edge_index, torch.Tensor):
            continue
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            continue
        num_edges = int(edge_index.size(1))

        expected_dim = edge_dim_dict[edge_type]
        edge_attr = getattr(edge_store, "edge_attr", None)

        if expected_dim == 0:
            # Featureless — must NOT have edge_attr.
            if isinstance(edge_attr, torch.Tensor):
                raise RuntimeError(
                    f"Featureless edge type {edge_type} should not "
                    f"have edge_attr, but found tensor with shape {list(edge_attr.shape)}."
                )
            continue

        if not isinstance(edge_attr, torch.Tensor):
            raise RuntimeError(
                f"Edge type {edge_type} is missing edge_attr; "
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
                f"edge_attr dim mismatch for edge type {edge_type}: "
                f"got {edge_attr.size(1)}, expected {expected_dim}."
            )

    return data


def validate_edge_attr(data, edge_dim):
    """Validate that every edge type carries properly shaped ``edge_attr``.

    *edge_dim* can be:

    * **int** — every edge type must have ``edge_attr`` with that many columns
      (featureless types that have no ``edge_attr`` are silently skipped).
        * **dict** — per-edge-triple widths; types declared with zero must not carry
            ``edge_attr``.
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
