from dataclasses import dataclass

from .download_and_uncompress_data import ensure_opflearn_prepared


def _serialize_parquet_to_hdf5(*args, **kwargs):
    from .opflearn_pipeline.utils.pyg_serialization import serialize_parquet_to_hdf5

    return serialize_parquet_to_hdf5(*args, **kwargs)


@dataclass(frozen=True)
class _DataOps:
    ensure_opflearn_prepared: callable
    serialize_parquet_to_hdf5: callable


data_ops = _DataOps(
    ensure_opflearn_prepared=ensure_opflearn_prepared,
    serialize_parquet_to_hdf5=_serialize_parquet_to_hdf5,
)

__all__ = ["data_ops"]
