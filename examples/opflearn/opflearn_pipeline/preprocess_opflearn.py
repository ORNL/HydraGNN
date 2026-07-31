
import argparse
import logging
import sys
from pathlib import Path

from utils.preprocessing import process_csv_to_parquet, process_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Preprocess OPFLearn CSV files and write corrected Parquet outputs.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input CSV file path.")
    parser.add_argument("--output", type=Path, default=None, help="Output Parquet file path for single-file mode.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input directory containing one or more CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for processed Parquet files.")
    parser.add_argument("--chunksize", type=int, default=1000, help="Rows per processing chunk.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else (base_dir / value).resolve()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("preprocess_opflearn")

    script_dir = Path(__file__).resolve().parent

    input_csv = resolve_path(script_dir, args.input)
    output_parquet = resolve_path(script_dir, args.output)
    input_dir = resolve_path(script_dir, args.input_dir)
    output_dir = resolve_path(script_dir, args.output_dir)

    try:
        if input_csv is not None:
            if output_parquet is None:
                raise ValueError("--output is required when --input is provided.")
            stats = process_csv_to_parquet(
                input_csv=input_csv,
                output_parquet=output_parquet,
                chunksize=args.chunksize,
                overwrite=args.overwrite,
                logger=logger,
            )
            logger.info(
                "Completed single-file preprocessing (rows=%d, columns=%d).",
                stats["rows"],
                stats["columns"],
            )
            return 0

        if input_dir is not None:
            if output_dir is None:
                raise ValueError("--output-dir is required when --input-dir is provided.")
            results = process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                chunksize=args.chunksize,
                overwrite=args.overwrite,
                logger=logger,
            )
            logger.info("Completed directory preprocessing for %d files.", len(results))
            return 0

        default_input_dir = (script_dir / "dataset/opflearn/extracted").resolve()
        default_output_dir = (script_dir / "dataset/opflearn/processed").resolve()
        logger.info(
            "No explicit input provided; processing directory %s -> %s",
            default_input_dir,
            default_output_dir,
        )
        results = process_directory(
            input_dir=default_input_dir,
            output_dir=default_output_dir,
            chunksize=args.chunksize,
            overwrite=args.overwrite,
            logger=logger,
        )
        logger.info("Completed directory preprocessing for %d files.", len(results))
        return 0
    except Exception:
        logger.exception("OPFLearn preprocessing failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
