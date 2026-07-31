import argparse
import logging
import sys
from pathlib import Path

import torch.distributed as dist

import hydragnn
from opflearn.pyg_serialization import serialize_directory, serialize_parquet_to_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert processed OPFLearn parquet files into PyG objects serialized as HydraGNN HDF5 datasets.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Single input parquet file.")
    parser.add_argument("--output", type=Path, default=None, help="Single output HDF5 dataset directory (*.h5).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/opflearn/processed"),
        help="Directory with processed parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/opflearn/serialized_hdf5"),
        help="Output directory for HDF5 dataset directories.",
    )
    parser.add_argument("--include-infeasible", action="store_true", help="Also serialize INFEASIBLE_*.parquet files.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of rows per input file.")
    parser.add_argument("--include-duals", action="store_true", help="Reserved: include dual outputs in y (not yet enabled).")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split shuffling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output datasets.")
    return parser.parse_args()


def _resolve(base_dir: Path, p: Path | None) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (base_dir / p).resolve()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("serialize_opflearn_to_hdf5")

    script_dir = Path(__file__).resolve().parent
    input_file = _resolve(script_dir, args.input)
    output_file = _resolve(script_dir, args.output)
    input_dir = _resolve(script_dir, args.input_dir)
    output_dir = _resolve(script_dir, args.output_dir)

    try:
        # HDF5Writer currently relies on HydraGNN iterate_tqdm, which reads
        # torch.distributed rank unconditionally.
        hydragnn.utils.distributed.setup_ddp()

        if input_file is not None:
            if output_file is None:
                raise ValueError("--output is required when --input is specified.")
            stats = serialize_parquet_to_hdf5(
                input_parquet=input_file,
                output_hdf5_dir=output_file,
                max_samples=args.max_samples,
                include_duals=args.include_duals,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                seed=args.seed,
                overwrite=args.overwrite,
                logger=logger,
            )
            logger.info("Done: %s", stats)
            return 0

        assert input_dir is not None
        assert output_dir is not None
        results = serialize_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            include_infeasible=args.include_infeasible,
            max_samples=args.max_samples,
            include_duals=args.include_duals,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            overwrite=args.overwrite,
            logger=logger,
        )
        logger.info("Serialized %d files to HDF5 datasets.", len(results))
        return 0
    except Exception:
        logger.exception("Failed to serialize OPFLearn parquet to HDF5.")
        return 1
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
