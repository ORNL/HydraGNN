import importlib.util
from pathlib import Path

import pytest
import torch


_MODULE_PATH = (
    Path(__file__).parents[1] / "examples" / "opf" / "opf_solution_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("opf_solution_utils", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

circular_angle_mse = _MODULE.circular_angle_mse
wrapped_angle_difference = _MODULE.wrapped_angle_difference


def pytest_wrapped_angle_difference_uses_short_path():
    degrees = torch.tensor([-179.0, 179.0])
    actual = wrapped_angle_difference(
        torch.deg2rad(degrees[:1]), torch.deg2rad(degrees[1:])
    )
    assert torch.rad2deg(actual).item() == pytest.approx(2.0, abs=1e-5)


def pytest_circular_angle_mse_uses_wrapped_residual():
    prediction = torch.deg2rad(torch.tensor([-179.0]))
    target = torch.deg2rad(torch.tensor([179.0]))
    expected = torch.deg2rad(torch.tensor(2.0)).square().item()
    assert circular_angle_mse(prediction, target).item() == pytest.approx(
        expected, abs=1e-8
    )


def pytest_uniform_offset_is_penalized_to_preserve_reference_convention():
    target = torch.tensor([-0.4, 0.1, 0.7])
    prediction = target + 1.25
    assert circular_angle_mse(prediction, target).item() == pytest.approx(1.25**2)
