"""Downstream — downstream regional-anomaly detection on the fnet forecaster outputs.

Model-agnostic: everything here reads a forecaster ``--out_dir`` (never the model
or the raw parquet). ``downstream_io.load_split`` is the single adapter that maps the
saved artifacts into the arrays/graph/timestamps the detectors consume.
"""

from .downstream_io import DownstreamData, load_split

__all__ = ["DownstreamData", "load_split"]
