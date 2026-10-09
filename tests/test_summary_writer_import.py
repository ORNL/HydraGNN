import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.mpi_skip()
def test_import_does_not_load_tensorboard():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hydragnn; "
            "assert 'torch.utils.tensorboard' not in sys.modules; "
            "assert 'tensorflow' not in sys.modules",
        ],
        cwd=root,
        check=True,
        timeout=180,
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_summary_writer_is_created_only_on_rank_zero(monkeypatch, tmp_path, rank):
    import hydragnn.utils.model.model as model_utils

    calls = []
    sentinel = object()
    tensorboard = ModuleType("torch.utils.tensorboard")

    def writer(path):
        calls.append(path)
        return sentinel

    tensorboard.SummaryWriter = writer
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard)
    monkeypatch.setattr(model_utils, "get_comm_size_and_rank", lambda: (2, rank))
    result = model_utils.get_summary_writer("run", path=str(tmp_path))
    assert result is (sentinel if rank == 0 else None)
    assert calls == ([str(tmp_path / "run")] if rank == 0 else [])
