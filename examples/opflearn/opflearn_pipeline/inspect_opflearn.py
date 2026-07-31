
import argparse
import json
import logging
import sys
from pathlib import Path

from opflearn.inspect import inspect_dataset_table, load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inspect OPFLearn CSV/Parquet schema and statistics.",
    )
    parser.add_argument("input_path", type=Path, help="Path to processed Parquet or raw CSV file.")
    parser.add_argument("--json", action="store_true", help="Emit report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("inspect_opflearn")

    script_dir = Path(__file__).resolve().parent
    input_path = args.input_path if args.input_path.is_absolute() else (script_dir / args.input_path).resolve()

    try:
        frame = load_table(input_path)
        report = inspect_dataset_table(frame=frame, file_name=str(input_path))

        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            logger.info("File: %s", report["file_name"])
            logger.info("Rows: %d", report["rows"])
            logger.info("Columns: %d", report["columns"])
            logger.info(
                "Components: loads=%d generators=%d buses=%d branches=%d",
                report["num_loads"],
                report["num_generators"],
                report["num_buses"],
                report["num_branches"],
            )
            logger.info("Detected input columns: %d", len(report["detected_input_columns"]))
            logger.info("Detected primal-output columns: %d", len(report["detected_primal_output_columns"]))
            logger.info("Detected dual-output columns: %d", len(report["detected_dual_output_columns"]))
            logger.info("Raw angle range (rad): %s", report["raw_angle_range_rad"])
            logger.info("Corrected angle range (rad): %s", report["corrected_angle_range_rad"])
            logger.info("Voltage magnitude range: %s", report["voltage_magnitude_range"])
            logger.info("Branch flow range: %s", report["branch_flow_range"])
            logger.info("Load P summary: %s", report["summary_load_p"])
            logger.info("Load Q summary: %s", report["summary_load_q"])
            logger.info("Optimal Pg summary: %s", report["summary_optimal_pg"])
            logger.info("Optimal Qg summary: %s", report["summary_optimal_qg"])

        return 0
    except Exception:
        logger.exception("Inspection failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
