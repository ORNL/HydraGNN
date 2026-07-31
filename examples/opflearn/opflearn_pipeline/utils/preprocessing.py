
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_complex_dtype

from .parsing import parse_complex_voltage
from .validation import validate_voltage_transformation
from .voltage_correction import correct_bus_voltage_array

INPUT_SUFFIXES = {"pl", "ql"}
PRIMAL_SUFFIXES = {
    "pg",
    "qg",
    "vm_gen",
    "v_bus",
    "vm_bus",
    "va_bus_rad",
    "va_bus_deg",
    "v_bus_real_corrected",
    "v_bus_imag_corrected",
    "p_to",
    "p_fr",
    "q_to",
    "q_fr",
}
DUAL_SUFFIXES = {
    "v_min",
    "v_max",
    "pg_min",
    "pg_max",
    "qg_min",
    "qg_max",
    "p_to_max",
    "p_fr_max",
    "q_to_max",
    "q_fr_max",
}


def suffix_of(column: str) -> str:
    """Return suffix after ':' in an OPFLearn column name."""
    if ":" not in column:
        return ""
    return column.rsplit(":", 1)[1]


def prefix_of(column: str) -> str:
    """Return prefix before ':' in an OPFLearn column name."""
    if ":" not in column:
        return column
    return column.rsplit(":", 1)[0]


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    """Classify columns into input, primal-output, and dual-output groups."""
    input_cols: list[str] = []
    primal_cols: list[str] = []
    dual_cols: list[str] = []

    for column in columns:
        suffix = suffix_of(column)
        if suffix in INPUT_SUFFIXES:
            input_cols.append(column)
        if suffix in PRIMAL_SUFFIXES:
            primal_cols.append(column)
        if suffix in DUAL_SUFFIXES or suffix.endswith("_max") or suffix.endswith("_min"):
            dual_cols.append(column)

    return {
        "input": sorted(set(input_cols)),
        "primal": sorted(set(primal_cols)),
        "dual": sorted(set(dual_cols)),
    }


def detect_component_counts(columns: list[str]) -> dict[str, int]:
    """Infer load/gen/bus/line counts from column prefixes."""
    groups = {"loads": set(), "generators": set(), "buses": set(), "branches": set()}

    for column in columns:
        prefix = prefix_of(column)
        if prefix.startswith("load"):
            groups["loads"].add(prefix)
        elif prefix.startswith("gen"):
            groups["generators"].add(prefix)
        elif prefix.startswith("bus"):
            groups["buses"].add(prefix)
        elif prefix.startswith("line"):
            groups["branches"].add(prefix)

    return {k: len(v) for k, v in groups.items()}


def correct_v_bus_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Correct all columns ending in ``:v_bus`` and preserve raw values.

    For each ``<bus>:v_bus`` column, the function creates:
    - ``<bus>:v_bus_raw``
    - ``<bus>:vm_bus``
    - ``<bus>:va_bus_rad``
    - ``<bus>:va_bus_deg``
    - ``<bus>:v_bus_real_corrected``
    - ``<bus>:v_bus_imag_corrected``

    The original ``<bus>:v_bus`` column is dropped.
    """
    result = frame.copy()

    voltage_columns = [column for column in result.columns if column.endswith(":v_bus")]
    if not voltage_columns:
        raise ValueError("No columns ending in ':v_bus' were found.")

    for column in voltage_columns:
        parsed_values: list[complex] = []
        for row_index, value in result[column].items():
            try:
                parsed_values.append(parse_complex_voltage(value))
            except ValueError as exc:
                raise ValueError(
                    f"Failed to parse {column!r} at row index {row_index}: {value!r}"
                ) from exc

        raw_values = np.asarray(parsed_values, dtype=np.complex128)
        corrected_voltage, corrected_angle_rad, corrected_angle_deg = correct_bus_voltage_array(raw_values)
        vm_bus = np.abs(raw_values)

        validate_voltage_transformation(
            raw_voltage=raw_values,
            corrected_voltage=corrected_voltage,
            corrected_angle_rad=corrected_angle_rad,
            voltage_magnitude=vm_bus,
        )

        prefix = column.removesuffix(":v_bus")
        result[f"{prefix}:v_bus_raw"] = result[column].astype(str)
        result[f"{prefix}:vm_bus"] = vm_bus
        result[f"{prefix}:va_bus_rad"] = corrected_angle_rad
        result[f"{prefix}:va_bus_deg"] = corrected_angle_deg
        result[f"{prefix}:v_bus_real_corrected"] = corrected_voltage.real
        result[f"{prefix}:v_bus_imag_corrected"] = corrected_voltage.imag

        result.drop(columns=[column], inplace=True)

    return result


def process_csv_to_parquet(
    input_csv: Path,
    output_parquet: Path,
    chunksize: int,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, int]:
    """Preprocess a single OPFLearn CSV into a Parquet dataset.

    Uses chunked reading and ``pyarrow.parquet.ParquetWriter`` so large files can
    be processed with bounded memory.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if output_parquet.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_parquet}. Use --overwrite to replace it."
        )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    n_rows_total = 0
    n_cols_total = 0

    try:
        for chunk_id, chunk in enumerate(
            pd.read_csv(input_csv, chunksize=chunksize, low_memory=False)
        ):
            corrected = correct_v_bus_columns(chunk)
            if any(is_complex_dtype(dtype) for dtype in corrected.dtypes):
                raise ValueError("Complex-typed columns are not allowed in Parquet output.")

            table = pa.Table.from_pandas(corrected, preserve_index=False)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(output_parquet, output_schema, compression="snappy")
                n_cols_total = table.num_columns
                logger.info("Initialized Parquet schema with %d columns.", n_cols_total)
            else:
                assert output_schema is not None
                if table.schema != output_schema:
                    table = table.select(output_schema.names).cast(output_schema, safe=False)

            writer.write_table(table)
            n_rows_total += len(corrected)
            logger.info("Processed chunk %d with %d rows.", chunk_id, len(corrected))
    finally:
        if writer is not None:
            writer.close()

    logger.info(
        "Finished preprocessing %s -> %s (rows=%d, columns=%d)",
        input_csv,
        output_parquet,
        n_rows_total,
        n_cols_total,
    )

    return {"rows": n_rows_total, "columns": n_cols_total}


def process_directory(
    input_dir: Path,
    output_dir: Path,
    chunksize: int,
    overwrite: bool,
    logger: logging.Logger,
) -> list[dict[str, str | int]]:
    """Process all CSV files in a directory to Parquet files."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {input_dir}")

    results: list[dict[str, str | int]] = []
    for csv_file in csv_files:
        out_path = output_dir / (csv_file.stem + ".parquet")
        stats = process_csv_to_parquet(
            input_csv=csv_file,
            output_parquet=out_path,
            chunksize=chunksize,
            overwrite=overwrite,
            logger=logger,
        )
        results.append(
            {
                "input": str(csv_file),
                "output": str(out_path),
                "rows": stats["rows"],
                "columns": stats["columns"],
            }
        )
    return results
