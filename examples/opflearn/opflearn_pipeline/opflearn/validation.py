
import logging

import numpy as np


def validate_voltage_transformation(
    raw_voltage: np.ndarray,
    corrected_voltage: np.ndarray,
    corrected_angle_rad: np.ndarray,
    voltage_magnitude: np.ndarray,
    logger: logging.Logger | None = None,
) -> None:
    """Run numerical validations for OPFLearn voltage-angle correction.

    Validation checks:
    - magnitude preservation for finite values
    - finite corrected magnitude/angle when raw values are finite
    - reconstruction consistency
    - exact angle-correction equation
    - suspicious nonpositive voltage magnitudes
    """
    log = logger or logging.getLogger(__name__)

    raw_voltage = np.asarray(raw_voltage, dtype=np.complex128)
    corrected_voltage = np.asarray(corrected_voltage, dtype=np.complex128)
    corrected_angle_rad = np.asarray(corrected_angle_rad, dtype=np.float64)
    voltage_magnitude = np.asarray(voltage_magnitude, dtype=np.float64)

    finite_mask = np.isfinite(raw_voltage.real) & np.isfinite(raw_voltage.imag)

    if np.any(finite_mask):
        np.testing.assert_allclose(
            np.abs(corrected_voltage[finite_mask]),
            np.abs(raw_voltage[finite_mask]),
            rtol=1e-12,
            atol=1e-12,
        )

        if not np.all(np.isfinite(voltage_magnitude[finite_mask])):
            raise ValueError("Corrected voltage magnitudes contain non-finite values.")

        if not np.all(np.isfinite(corrected_angle_rad[finite_mask])):
            raise ValueError("Corrected voltage angles contain non-finite values.")

        reconstructed = voltage_magnitude * np.exp(1j * corrected_angle_rad)
        np.testing.assert_allclose(
            reconstructed[finite_mask],
            corrected_voltage[finite_mask],
            rtol=1e-12,
            atol=1e-12,
        )

        expected_angle = -np.angle(raw_voltage) * 180.0 / np.pi
        np.testing.assert_allclose(
            corrected_angle_rad[finite_mask],
            expected_angle[finite_mask],
            rtol=1e-12,
            atol=1e-12,
        )

    suspicious = int(np.sum((voltage_magnitude <= 0.0) & np.isfinite(voltage_magnitude)))
    if suspicious > 0:
        log.warning("Detected %d nonpositive vm_bus values in corrected output.", suspicious)
