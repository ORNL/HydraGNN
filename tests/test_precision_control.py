import types

import torch
import pytest

from importlib import import_module

tvt = import_module("hydragnn.train.train_validate_test")

pytestmark = pytest.mark.mpi_skip()


def pytest_resolve_precision_aliases():
    prec, param_dtype, autocast_dtype = tvt.resolve_precision("bfloat16")
    assert prec == "bf16"
    assert param_dtype == torch.float32
    assert autocast_dtype == torch.bfloat16

    prec, param_dtype, autocast_dtype = tvt.resolve_precision("float32")
    assert prec == "fp32"
    assert param_dtype == torch.float32
    assert autocast_dtype is None

    prec, param_dtype, autocast_dtype = tvt.resolve_precision("double")
    assert prec == "fp64"
    assert param_dtype == torch.float64
    assert autocast_dtype is None


def pytest_move_batch_to_device_fp64(monkeypatch):
    monkeypatch.setattr(tvt, "get_device", lambda: torch.device("cpu"))
    tensor = torch.ones(2, dtype=torch.float32)
    moved = tvt.move_batch_to_device(tensor, torch.float64)
    assert moved.dtype == torch.float64
    assert moved.device.type == "cpu"


def pytest_bf16_autocast_cpu(monkeypatch):
    # Pretend only CPU is available but supports bfloat16
    monkeypatch.setattr(tvt, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "cpu", types.SimpleNamespace(has_bf16=True))

    autocast_ctx, scaler = tvt.get_autocast_and_scaler("bf16")

    assert scaler is None
    # Should return a real autocast context, not nullcontext, when bf16 is supported
    assert autocast_ctx.__class__.__name__ != "nullcontext"

    # Ensure the context executes without errors
    with autocast_ctx:
        out = torch.ones(1) + 1
    assert out.dtype in (torch.bfloat16, torch.float32, torch.float64)
