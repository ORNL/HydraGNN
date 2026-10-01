from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from hydragnn.loss_reporting import (
    accumulate_loss_report,
    finalize_loss_report,
    reset_loss_report,
)
from hydragnn.utils.distributed import setup_ddp

pytestmark = pytest.mark.mpi


def test_distributed_report_includes_rank_specific_components():
    world_size, rank = setup_ddp()
    assert world_size == 2

    model = SimpleNamespace()
    reset_loss_report(model)
    model.last_loss_components = {
        f"constraints.rank_{rank}": {
            "raw": torch.tensor(float(rank + 1)),
            "weight": float(rank + 2),
            "weighted": torch.tensor(float((rank + 1) * (rank + 2))),
        }
    }
    accumulate_loss_report(model, sample_count=1)

    report = finalize_loss_report(model, torch.device("cpu"))

    assert set(report) == {"constraints.rank_0", "constraints.rank_1"}
    for component_rank in range(world_size):
        component = report[f"constraints.rank_{component_rank}"]
        assert component["raw"] == pytest.approx(float(component_rank + 1) / world_size)
        assert component["weight"] == pytest.approx(float(component_rank + 2))
        assert component["weighted"] == pytest.approx(
            float((component_rank + 1) * (component_rank + 2)) / world_size
        )

    dist.barrier()
