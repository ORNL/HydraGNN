
import argparse
import logging
import sys
from pathlib import Path

from opflearn.download import (
    DEFAULT_CASES,
    download_archive,
    download_cases,
    extract_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download OPFLearnData archive or selected case CSV files.",
    )
    parser.add_argument("--archive", action="store_true", help="Download the complete OPFLearn_Datasets.zip archive.")
    parser.add_argument("--cases", nargs="*", default=None, help="Selected case names, e.g. case14_ieee case118_ieee.")
    parser.add_argument("--all-cases", action="store_true", help="Download all known individual case CSV files.")
    parser.add_argument("--extract", action="store_true", help="Extract ZIP archive after download.")
    parser.add_argument("--force", action="store_true", help="Re-download even when target file already exists.")
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/opflearn/raw"), help="Output directory for downloaded files.")
    parser.add_argument("--extract-dir", type=Path, default=Path("dataset/opflearn/extracted"), help="Output directory for extracted archive contents.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("download_opflearn")

    script_dir = Path(__file__).resolve().parent
    raw_dir = (script_dir / args.raw_dir).resolve() if not args.raw_dir.is_absolute() else args.raw_dir
    extract_dir = (script_dir / args.extract_dir).resolve() if not args.extract_dir.is_absolute() else args.extract_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    cases: list[str] = []
    if args.cases:
        cases = args.cases
    elif args.all_cases:
        cases = list(DEFAULT_CASES)

    try:
        archive_path = raw_dir / "OPFLearn_Datasets.zip"

        if args.archive:
            archive_path = download_archive(raw_dir=raw_dir, force=args.force, logger=logger)

        if cases:
            download_cases(cases=cases, raw_dir=raw_dir, force=args.force, logger=logger)

        if args.extract:
            if not archive_path.exists():
                raise FileNotFoundError(
                    f"Archive not found at {archive_path}. Pass --archive first or place the zip there."
                )
            extract_archive(archive_path=archive_path, extract_dir=extract_dir, logger=logger)

        if not args.archive and not cases and not args.extract:
            logger.info("No action selected; defaulting to --archive --extract.")
            archive_path = download_archive(raw_dir=raw_dir, force=args.force, logger=logger)
            extract_archive(archive_path=archive_path, extract_dir=extract_dir, logger=logger)

        return 0
    except Exception:
        logger.exception("Failed to download/extract OPFLearn data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
