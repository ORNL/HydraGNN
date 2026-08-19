
from pathlib import Path

import numpy as np
import pandas as pd

from .parsing import parse_complex_voltage
from .preprocessing import classify_columns, detect_component_counts


def _summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce")
    return {
        "min": float(np.nanmin(values.to_numpy())) if values.notna().any() else float("nan"),
        "max": float(np.nanmax(values.to_numpy())) if values.notna().any() else float("nan"),
        "mean": float(np.nanmean(values.to_numpy())) if values.notna().any() else float("nan"),
    }


def _aggregate_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    if not columns:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    # Avoid DataFrame.stack(dropna=...) because pandas behavior differs across versions.
    flattened = frame[columns].to_numpy().reshape(-1)
    values = pd.to_numeric(pd.Series(flattened), errors="coerce")
    return _summary(values)


def _range_from_columns(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    return _aggregate_summary(frame, columns)


def _raw_angle_range_from_vbus_raw(frame: pd.DataFrame) -> dict[str, float]:
    raw_cols = [c for c in frame.columns if c.endswith(":v_bus_raw")]
    if not raw_cols:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}

    all_angles: list[np.ndarray] = []
    for col in raw_cols:
        complex_vals = frame[col].map(parse_complex_voltage).to_numpy(dtype=np.complex128)
        all_angles.append(np.angle(complex_vals))

    merged = np.concatenate(all_angles)
    return {
        "min": float(np.nanmin(merged)),
        "max": float(np.nanmax(merged)),
        "mean": float(np.nanmean(merged)),
    }


def inspect_dataset_table(frame: pd.DataFrame, file_name: str) -> dict[str, object]:
    """Build a schema and statistics report for an OPFLearn table."""
    columns = list(frame.columns)
    groups = classify_columns(columns)
    counts = detect_component_counts(columns)

    load_p = [c for c in columns if c.endswith(":pl")]
    load_q = [c for c in columns if c.endswith(":ql")]
    gen_p = [c for c in columns if c.endswith(":pg")]
    gen_q = [c for c in columns if c.endswith(":qg")]
    vm_cols = [c for c in columns if c.endswith(":vm_bus")]
    va_cols = [c for c in columns if c.endswith(":va_bus_rad")]
    branch_cols = [
        c
        for c in columns
        if c.endswith(":p_to") or c.endswith(":p_fr") or c.endswith(":q_to") or c.endswith(":q_fr")
    ]

    missing_counts = frame.isna().sum().to_dict()

    report = {
        "file_name": file_name,
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "num_loads": counts["loads"],
        "num_generators": counts["generators"],
        "num_buses": counts["buses"],
        "num_branches": counts["branches"],
        "detected_input_columns": groups["input"],
        "detected_primal_output_columns": groups["primal"],
        "detected_dual_output_columns": groups["dual"],
        "missing_value_counts": missing_counts,
        "summary_load_p": _range_from_columns(frame, load_p),
        "summary_load_q": _range_from_columns(frame, load_q),
        "summary_optimal_pg": _range_from_columns(frame, gen_p),
        "summary_optimal_qg": _range_from_columns(frame, gen_q),
        "raw_angle_range_rad": _raw_angle_range_from_vbus_raw(frame),
        "corrected_angle_range_rad": _range_from_columns(frame, va_cols),
        "voltage_magnitude_range": _range_from_columns(frame, vm_cols),
        "branch_flow_range": _range_from_columns(frame, branch_cols),
    }

    return report


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet table for inspection."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format for inspection: {path}")
