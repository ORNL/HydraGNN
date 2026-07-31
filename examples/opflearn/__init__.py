import os
import sys
from dataclasses import dataclass

# Ensure the OPFLearn pipeline utilities are importable for serialization.
_PIPELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opflearn_pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from download_and_uncompress_data import ensure_opflearn_prepared
from utils.pyg_serialization import serialize_parquet_to_hdf5


@dataclass(frozen=True)
class _DataOps:
    ensure_opflearn_prepared: callable
    serialize_parquet_to_hdf5: callable


data_ops = _DataOps(
    ensure_opflearn_prepared=ensure_opflearn_prepared,
    serialize_parquet_to_hdf5=serialize_parquet_to_hdf5,
)

__all__ = ["data_ops"]
