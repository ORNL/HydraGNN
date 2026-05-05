"""HDF5-based storage for heterogeneous PyG data.

Each MPI rank writes its own shard file inside a directory:
    <basedir>/
        meta.h5              – total counts, rank offsets
        shard-0000.h5        – samples from rank 0
        shard-0001.h5        – samples from rank 1
        …

Each shard stores samples as variable-length byte datasets (one per label)
containing pickle-serialised HeteroData objects.  No homogeneous conversion.

The writer supports two modes of operation:

**Batch mode** (original API — backward compatible)::

    w = HDF5Writer(basedir, comm)
    w.add("trainset", list_of_samples)
    w.save()

**Streaming mode** (memory-efficient — like AdiosWriter)::

    w = HDF5Writer(basedir, comm)
    w.begin("trainset")          # open a streaming label
    for sample in process(...):
        w.put(sample)            # serialize & flush immediately
    w.end_label()                # finalize the current label
    w.save()                     # write metadata only
"""

import os
import pickle

import h5py
import numpy as np
from mpi4py import MPI

from .abstractbasedataset import AbstractBaseDataset
from hydragnn.utils.print import iterate_tqdm


# ──────────────────────────────────────────────────────────────────────
#  Writer
# ──────────────────────────────────────────────────────────────────────
class HDF5Writer:
    """Write HDF5 shards with optional streaming to avoid OOM.

    Supports two usage patterns:

    1. **Batch mode** — ``add(label, data)`` then ``save()`` (original API,
       backward compatible).
    2. **Streaming mode** — ``begin(label)`` / ``put(sample)`` /
       ``end_label()`` then ``save()``.  Samples are serialized and flushed
       to disk in small batches; the caller never needs to hold all samples
       in memory.
    """

    def __init__(self, basedir, comm=MPI.COMM_WORLD, batch_size=64):
        self.basedir = basedir
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.batch_size = batch_size

        # Batch-mode accumulator (used only when add() is called)
        self._labels = {}  # label -> list[data]

        # Streaming-mode state
        self._fh = None  # open h5py.File handle
        self._shard_path = None
        self._stream_label = None  # active label being streamed
        self._stream_ds = None  # active HDF5 dataset
        self._stream_buf = []  # pending samples not yet flushed
        self._stream_offset = 0  # next write position in the dataset
        self._label_counts = {}  # label -> final count (for metadata)
        self._streaming_used = False

    # ── batch-mode API (backward compatible) ──────────────────────────

    def add(self, label, data):
        if label not in self._labels:
            self._labels[label] = []
        if isinstance(data, list):
            self._labels[label].extend(data)
        else:
            self._labels[label].append(data)

    # ── streaming-mode API ────────────────────────────────────────────

    def begin(self, label):
        """Start streaming samples for *label*.  Opens the shard file on
        first call."""
        self._streaming_used = True
        if self._stream_label is not None:
            self.end_label()

        if self._fh is None:
            os.makedirs(self.basedir, exist_ok=True)
            self._shard_path = os.path.join(self.basedir, f"shard-{self.rank:04d}.h5")
            self._fh = h5py.File(self._shard_path, "w")

        if label in self._fh:
            # Resume appending to an existing dataset (e.g. second case
            # writing more samples into the same "trainset" dataset).
            self._stream_ds = self._fh[label]
            self._stream_offset = self._stream_ds.shape[0]
        else:
            vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
            # Use a resizable (chunked) dataset so we can append without
            # knowing the total count up-front.
            self._stream_ds = self._fh.create_dataset(
                label,
                shape=(0,),
                maxshape=(None,),
                dtype=vlen_dt,
                chunks=(self.batch_size,),
            )
            self._stream_offset = 0
        self._stream_label = label
        self._stream_buf = []

    def put(self, sample):
        """Add a single sample.  Automatically flushes to disk every
        *batch_size* samples."""
        self._stream_buf.append(sample)
        if len(self._stream_buf) >= self.batch_size:
            self._flush_stream_buf()

    def end_label(self):
        """Finalize the current streaming label — flush remaining samples."""
        if self._stream_buf:
            self._flush_stream_buf()
        if self._stream_label is not None:
            self._label_counts[self._stream_label] = self._stream_offset
        self._stream_label = None
        self._stream_ds = None
        self._stream_buf = []

    def _flush_stream_buf(self):
        """Serialize buffered samples and write them to the open dataset."""
        n = len(self._stream_buf)
        if n == 0:
            return
        new_end = self._stream_offset + n
        self._stream_ds.resize((new_end,))
        for j, sample in enumerate(self._stream_buf):
            self._stream_ds[self._stream_offset + j] = np.frombuffer(
                pickle.dumps(sample, protocol=4), dtype=np.uint8
            )
        self._stream_offset = new_end
        self._stream_buf.clear()

    # ── save (works for both modes) ───────────────────────────────────

    def save(self):
        # Finalize any open streaming label
        if self._stream_label is not None:
            self.end_label()

        # If batch-mode data was collected, write it now
        if self._labels:
            self._save_batch_mode()

        # Close the shard file if streaming opened it
        if self._fh is not None:
            self._fh.close()
            self._fh = None

        self.comm.Barrier()

        # ── Metadata ─────────────────────────────────────────────────

        # Merge label names from both modes
        local_labels = sorted(
            set(list(self._labels.keys()) + list(self._label_counts.keys()))
        )
        all_labels = self.comm.allgather(local_labels)
        labels = sorted({lbl for lst in all_labels for lbl in lst})

        per_label_counts = {}
        for label in labels:
            if label in self._label_counts:
                local_n = self._label_counts[label]
            else:
                local_n = len(self._labels.get(label, []))
            counts = self.comm.gather(local_n, root=0)
            if self.rank == 0:
                per_label_counts[label] = counts

        if self.rank == 0:
            meta_path = os.path.join(self.basedir, "meta.h5")
            with h5py.File(meta_path, "w") as fh:
                fh.attrs["num_shards"] = self.size
                for label in labels:
                    grp = fh.create_group(label)
                    counts = np.array(per_label_counts[label], dtype=np.int64)
                    grp.create_dataset("counts", data=counts)
                    grp.attrs["total"] = int(counts.sum())

        self.comm.Barrier()

    def _save_batch_mode(self):
        """Write all batch-accumulated data (original logic)."""
        if self.rank == 0:
            os.makedirs(self.basedir, exist_ok=True)
        self.comm.Barrier()

        BATCH = self.batch_size
        shard_path = os.path.join(self.basedir, f"shard-{self.rank:04d}.h5")
        vlen_dt = h5py.vlen_dtype(np.dtype("uint8"))
        mode = "a" if self._streaming_used else "w"
        with h5py.File(shard_path, mode) as fh:
            for label, samples in self._labels.items():
                n = len(samples)
                if n > 0:
                    ds = fh.create_dataset(label, shape=(n,), dtype=vlen_dt)
                    for start in iterate_tqdm(
                        range(0, n, BATCH),
                        2,
                        total=(n + BATCH - 1) // BATCH,
                        desc=f"HDF5 write {label}",
                    ):
                        end = min(start + BATCH, n)
                        for j, sample in enumerate(samples[start:end]):
                            ds[start + j] = np.frombuffer(
                                pickle.dumps(sample, protocol=4),
                                dtype=np.uint8,
                            )
                        # Release processed samples to reduce memory pressure
                        for k in range(start, end):
                            samples[k] = None
                else:
                    fh.create_dataset(label, shape=(0,), dtype=vlen_dt)


# ──────────────────────────────────────────────────────────────────────
#  Dataset (reader)
# ──────────────────────────────────────────────────────────────────────
class HDF5Dataset(AbstractBaseDataset):
    """Read back a split written by HDF5Writer."""

    def __init__(self, basedir, label, var_config=None):
        super().__init__()
        self.basedir = basedir
        self.label = label
        self.var_config = var_config

        meta_path = os.path.join(basedir, "meta.h5")
        with h5py.File(meta_path, "r") as fh:
            grp = fh[label]
            self.counts = grp["counts"][:]
            self.ntotal = int(grp.attrs["total"])

        # Build a mapping from global index -> (shard_rank, local_index)
        self._offsets = np.zeros(len(self.counts) + 1, dtype=np.int64)
        np.cumsum(self.counts, out=self._offsets[1:])

        # Cache open file handles lazily
        self._handles = {}

    def len(self):
        return self.ntotal

    def get(self, idx):
        # Find which shard this global index belongs to
        shard = int(np.searchsorted(self._offsets[1:], idx, side="right"))
        local_idx = idx - int(self._offsets[shard])
        fh = self._open_shard(shard)
        blob = fh[self.label][local_idx]
        data = pickle.loads(blob.tobytes())
        return data

    def _open_shard(self, shard):
        if shard not in self._handles:
            path = os.path.join(self.basedir, f"shard-{shard:04d}.h5")
            self._handles[shard] = h5py.File(path, "r")
        return self._handles[shard]

    def __del__(self):
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
