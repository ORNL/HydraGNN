
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from opflearn.parsing import parse_complex_voltage
from opflearn.preprocessing import correct_v_bus_columns, process_csv_to_parquet
from opflearn.voltage_correction import ANGLE_CORRECTION_FACTOR, correct_bus_voltage_array


def test_parse_complex_with_spaces() -> None:
    value = "1.0887465217628047 - 0.0005979187202320081j"
    parsed = parse_complex_voltage(value)
    assert isinstance(parsed, complex)
    assert parsed.real == pytest.approx(1.0887465217628047)
    assert parsed.imag == pytest.approx(-0.0005979187202320081)


def test_parse_scientific_notation() -> None:
    value = "1.06 - 2.9086514862145653e-31j"
    parsed = parse_complex_voltage(value)
    assert parsed.real == pytest.approx(1.06)
    assert parsed.imag == pytest.approx(-2.9086514862145653e-31)


def test_parse_positive_negative_imaginary() -> None:
    positive = parse_complex_voltage("1.0 + 2.5j")
    negative = parse_complex_voltage("1.0 - 2.5j")
    assert positive.imag == pytest.approx(2.5)
    assert negative.imag == pytest.approx(-2.5)


def test_parse_missing_values() -> None:
    assert np.isnan(parse_complex_voltage(None).real)
    assert np.isnan(parse_complex_voltage("").imag)


def test_parse_malformed_string_raises() -> None:
    with pytest.raises(ValueError):
        parse_complex_voltage("this_is_not_complex")


def test_magnitude_preservation() -> None:
    raw = np.array([1.05 * np.exp(1j * -0.001), 0.98 * np.exp(1j * 0.02)], dtype=np.complex128)
    corrected, _, _ = correct_bus_voltage_array(raw)
    np.testing.assert_allclose(np.abs(corrected), np.abs(raw), rtol=1e-12, atol=1e-12)


def test_exact_minus_180_over_pi_correction() -> None:
    raw_angle_rad = -0.001
    magnitude = 1.05
    raw_voltage = np.array([magnitude * np.exp(1j * raw_angle_rad)], dtype=np.complex128)
    expected_angle_rad = -raw_angle_rad * 180.0 / np.pi

    _, corrected_angle_rad, _ = correct_bus_voltage_array(raw_voltage)
    assert corrected_angle_rad[0] == pytest.approx(expected_angle_rad, rel=1e-12, abs=1e-12)
    assert ANGLE_CORRECTION_FACTOR == pytest.approx(-180.0 / np.pi)


def test_correct_multiple_v_bus_columns() -> None:
    frame = pd.DataFrame(
        {
            "load1:pl": [1.0, 2.0],
            "bus1:v_bus": ["1.0 + 0.0j", "1.01 + 0.01j"],
            "bus2:v_bus": ["0.99 - 0.02j", "0.98 + 0.03j"],
            "gen1:pg": [0.5, 0.6],
        }
    )

    corrected = correct_v_bus_columns(frame)

    assert "bus1:v_bus" not in corrected.columns
    assert "bus2:v_bus" not in corrected.columns

    for bus in ("bus1", "bus2"):
        assert f"{bus}:v_bus_raw" in corrected.columns
        assert f"{bus}:vm_bus" in corrected.columns
        assert f"{bus}:va_bus_rad" in corrected.columns
        assert f"{bus}:va_bus_deg" in corrected.columns
        assert f"{bus}:v_bus_real_corrected" in corrected.columns
        assert f"{bus}:v_bus_imag_corrected" in corrected.columns


def test_dataframe_without_v_bus_raises() -> None:
    frame = pd.DataFrame({"load1:pl": [1.0], "gen1:pg": [0.2]})
    with pytest.raises(ValueError, match="No columns ending in ':v_bus'"):
        correct_v_bus_columns(frame)


def test_non_voltage_columns_preserved() -> None:
    frame = pd.DataFrame(
        {
            "load1:pl": [1.0],
            "load1:ql": [0.3],
            "gen1:pg": [0.8],
            "bus1:v_bus": ["1.0 + 0.1j"],
        }
    )
    corrected = correct_v_bus_columns(frame)
    assert corrected["load1:pl"].iloc[0] == pytest.approx(1.0)
    assert corrected["load1:ql"].iloc[0] == pytest.approx(0.3)
    assert corrected["gen1:pg"].iloc[0] == pytest.approx(0.8)


def test_end_to_end_csv_to_parquet(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    out_path = tmp_path / "sample.parquet"

    df = pd.DataFrame(
        {
            "load1:pl": [1.0, 1.2, 1.4],
            "load1:ql": [0.3, 0.35, 0.4],
            "gen1:pg": [0.9, 1.0, 1.1],
            "bus1:v_bus": [
                "1.05 - 0.001j",
                "1.04 + 0.002j",
                "1.03 - 0.003j",
            ],
        }
    )
    df.to_csv(csv_path, index=False)

    logger = logging.getLogger("test_opflearn")
    stats = process_csv_to_parquet(
        input_csv=csv_path,
        output_parquet=out_path,
        chunksize=2,
        overwrite=True,
        logger=logger,
    )

    assert out_path.exists()
    assert stats["rows"] == 3

    out_df = pd.read_parquet(out_path)
    assert "bus1:v_bus" not in out_df.columns
    assert "bus1:v_bus_raw" in out_df.columns
    assert "bus1:vm_bus" in out_df.columns
    assert "bus1:va_bus_rad" in out_df.columns
