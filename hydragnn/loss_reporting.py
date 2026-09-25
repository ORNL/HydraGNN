"""Named, unweighted loss-component reporting for training diagnostics."""

import torch
import torch.distributed as dist


def configure_supervised_components(model, output_specs):
    """Attach semantic output-column names to every loss-owning model module."""
    names = []
    for spec in output_specs:
        if spec.components:
            names.append(list(spec.components))
        elif spec.dim == 1:
            names.append([spec.name])
        else:
            names.append([f"{spec.name}[{index}]" for index in range(spec.dim)])
    for module in model.modules():
        if hasattr(module, "num_heads"):
            module.loss_component_names = names


def supervised_loss_report(model, pred, value, head_index, head_losses, var=None):
    """Return raw and weighted diagnostics for heads and their output columns."""
    report = {}
    configured = getattr(model, "loss_component_names", None)
    for index, head_loss in enumerate(head_losses):
        weight = float(model.loss_weights[index])
        names = (
            configured[index]
            if configured is not None and index < len(configured)
            else [f"head_{index}"]
        )
        head_name = names[0] if len(names) == 1 else f"head_{index}"
        report[f"supervised.{head_name}"] = {
            "raw": head_loss.detach(),
            "weight": weight,
            "weighted": (weight * head_loss).detach(),
        }

        head_pred = pred[index]
        head_target = value[head_index[index]]
        if head_target.shape != head_pred.shape:
            head_target = head_target.reshape_as(head_pred)
        if len(names) <= 1 or head_pred.ndim == 1 or head_pred.shape[-1] != len(names):
            continue
        for component_index, name in enumerate(names):
            if var is None:
                raw = model.loss_function(
                    head_pred[..., component_index], head_target[..., component_index]
                )
            else:
                raw = model.loss_function(
                    head_pred[..., component_index],
                    head_target[..., component_index],
                    var[index][..., component_index],
                )
            report[f"supervised.{name}"] = {
                "raw": raw.detach(),
                "weight": weight,
                "weighted": (weight * raw).detach(),
            }
    return report


def reset_loss_report(model):
    model._loss_report_sums = {}
    model._loss_report_count = 0


def accumulate_loss_report(model, sample_count):
    report = getattr(model, "last_loss_components", {})
    count = int(sample_count)
    model._loss_report_count += count
    for name, fields in report.items():
        target = model._loss_report_sums.setdefault(
            name,
            {
                "raw": 0.0,
                "weighted": 0.0,
                "weight": float(fields.get("weight", 1.0)),
            },
        )
        target["raw"] += float(fields["raw"].detach()) * count
        target["weighted"] += float(fields["weighted"].detach()) * count


def finalize_loss_report(model, device):
    """Average diagnostics over samples and ranks, then clear accumulation state."""
    count = torch.tensor(float(model._loss_report_count), device=device)
    if dist.is_initialized():
        dist.all_reduce(count)
        gathered_names = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_names, tuple(model._loss_report_sums))
        names = sorted({name for rank_names in gathered_names for name in rank_names})
    else:
        names = sorted(model._loss_report_sums)
    report = {}
    for name in names:
        fields = model._loss_report_sums.get(
            name, {"raw": 0.0, "weighted": 0.0, "weight": 0.0}
        )
        present = float(name in model._loss_report_sums)
        totals = torch.tensor(
            [fields["raw"], fields["weighted"], fields["weight"], present],
            dtype=torch.float64,
            device=device,
        )
        if dist.is_initialized():
            dist.all_reduce(totals)
        denominator = count.clamp_min(1.0)
        report[name] = {
            "raw": float(totals[0] / denominator),
            "weight": float(totals[2] / totals[3].clamp_min(1.0)),
            "weighted": float(totals[1] / denominator),
        }
    model.last_epoch_loss_report = report
    return report


def format_loss_report(epoch, split, total, report):
    fields = [
        f"epoch={epoch:02d}",
        f"split={split}",
        f"total={float(total):.8f}",
    ]
    for name in sorted(report):
        component = report[name]
        fields.extend(
            (
                f"{name}.raw={component['raw']:.8f}",
                f"{name}.weight={component['weight']:.8g}",
                f"{name}.weighted={component['weighted']:.8f}",
            )
        )
    return "LossComponents " + "  ".join(fields)
