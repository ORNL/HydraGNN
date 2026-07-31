
import numpy as np

ANGLE_CORRECTION_FACTOR = -180.0 / np.pi


def correct_bus_voltage_array(
    raw_voltage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct the known OPFLearnData complex bus-voltage angle error.

    Parameters
    ----------
    raw_voltage:
        One-dimensional complex array containing raw ``v_bus`` values.

    Returns
    -------
    corrected_voltage:
        Corrected complex bus voltages.
    corrected_angle_rad:
        Corrected voltage angles in radians.
    corrected_angle_deg:
        Corrected voltage angles in degrees.
    """
    raw_voltage = np.asarray(raw_voltage, dtype=np.complex128)

    magnitude = np.abs(raw_voltage)
    raw_angle_rad = np.angle(raw_voltage)

    corrected_angle_rad = ANGLE_CORRECTION_FACTOR * raw_angle_rad
    corrected_voltage = magnitude * np.exp(1j * corrected_angle_rad)
    corrected_angle_deg = np.rad2deg(corrected_angle_rad)

    return corrected_voltage, corrected_angle_rad, corrected_angle_deg


def wrapped_angle_difference(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute wrapped angular error robustly around +-pi boundaries."""
    return np.arctan2(np.sin(prediction - target), np.cos(prediction - target))
