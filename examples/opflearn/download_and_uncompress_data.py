import argparse
import logging
import os
import sys

# Reuse the OPFLearn pipeline utilities (download + preprocess).
_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opflearn_pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from pathlib import Path

from utils.download import download_cases
from utils.preprocessing import process_csv_to_parquet


def _case_csv_name(case_name: str) -> str:
    # Pipeline case names omit the leading "pglib_opf_" that the source files carry.
    stem = case_name[len("pglib_opf_"):] if case_name.startswith("pglib_opf_") else case_name
    return stem


def ensure_opflearn_prepared(
    root: str,
    case_name: str,
    rank: int,
    comm,
    chunksize: int = 1000,
    overwrite: bool = False,
) -> str:
    """Download and preprocess a single OPFLearn case into a Parquet dataset.

    Returns the path to the processed Parquet file (identical on every rank).
    """
    stem = _case_csv_name(case_name)
    raw_dir = Path(root) / "opflearn" / "raw"
    processed_dir = Path(root) / "opflearn" / "processed"
    parquet_path = processed_dir / f"pglib_opf_{stem}.parquet"

    if rank == 0:
        logger = logging.getLogger("ensure_opflearn_prepared")
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        csv_paths = download_cases(cases=[stem], raw_dir=raw_dir, force=overwrite, logger=logger)
        csv_path = csv_paths[0]

        if overwrite or not parquet_path.exists():
            process_csv_to_parquet(
                input_csv=csv_path,
                output_parquet=parquet_path,
                chunksize=chunksize,
                overwrite=True,
                logger=logger,
            )

    if comm is not None:
        comm.Barrier()
    return str(parquet_path)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--case_name", type=str, default="case14_ieee")
    parser.add_argument("--data_root", type=str, default="dataset")
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    from mpi4py import MPI

    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_root)
    parquet_path = ensure_opflearn_prepared(
        root=root,
        case_name=args.case_name,
        rank=rank,
        comm=comm,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    if rank == 0:
        logging.info("Prepared OPFLearn parquet: %s", parquet_path)


if __name__ == "__main__":
    main()
