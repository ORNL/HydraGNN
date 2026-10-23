import ast
import pathlib

import pytest
import torch

import hydragnn.distributed.distributed as hdist

REPO = pathlib.Path(__file__).resolve().parents[1]
TORCHISH = {"torch", "hydragnn", "torch_geometric"}


def _tops(node):
    if isinstance(node, ast.Import):
        return {a.name.split(".")[0] for a in node.names}
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        return {node.module.split(".")[0]}
    return set()


def test_examples_import_torch_before_mpi4py():
    # mpi4py loaded before torch prevents detection of the GPU-aware backend.
    offenders = []
    for path in sorted((REPO / "examples").rglob("*.py")):
        body = ast.parse(path.read_text()).body
        mpi = next((n.lineno for n in body if "mpi4py" in _tops(n)), None)
        first = next((n.lineno for n in body if _tops(n) & TORCHISH), None)
        if mpi is not None and first is not None and mpi < first:
            offenders.append(f"{path.relative_to(REPO)}:{mpi}")
    assert not offenders, "mpi4py imported before torch in: " + ", ".join(offenders)


@pytest.fixture
def no_gpu(monkeypatch):
    for var in hdist._GPU_ASSIGNMENT_VARS + ("HYDRAGNN_BACKEND",):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(hdist.dist, "is_initialized", lambda: False)


@pytest.mark.parametrize("var", ["ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_setup_ddp_refuses_gloo_when_gpus_assigned(no_gpu, monkeypatch, var):
    monkeypatch.setenv(var, "0")
    with pytest.raises(RuntimeError, match="refusing to fall back to gloo"):
        hdist.setup_ddp()


def test_setup_ddp_ignores_empty_gpu_assignment(no_gpu, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(hdist, "init_comm_size_and_rank", lambda: (1, 0))
    monkeypatch.setattr(
        hdist.dist,
        "init_process_group",
        lambda *a, **k: (_ for _ in ()).throw(SystemExit("reached init")),
    )
    with pytest.raises(SystemExit, match="reached init"):
        hdist.setup_ddp()
