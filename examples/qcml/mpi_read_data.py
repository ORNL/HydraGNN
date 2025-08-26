from mpi4py import MPI
import tensorflow as tf
import tensorflow_datasets as tfds

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
world = comm.Get_size()

LOCAL_DATA_DIR = './dataset'
# --- load base dataset identically on all ranks ---
base = tfds.load('qcml/dft_force_field', split='full', data_dir=LOCAL_DATA_DIR)

# Utility: count elements (linear scan; do it once, small cost)
def count(ds):
    return int(ds.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy())

# (A) Show global size (rank 0 only)
total = count(base) if rank == 0 else 0
total = comm.bcast(total, root=0)
if rank == 0:
    print(f"[rank 0] global dataset size = {total}")

# (B) SHARD FIRST, BEFORE shuffle/batch
ds = base.shard(num_shards=world, index=rank)

# If you zip with another dataset, shard both identically BEFORE zip:
# other = tfds.load('qcml/dft_force_field_d4', split='full', data_dir=LOCAL_DATA_DIR)
# ds     = tf.data.Dataset.zip((base.shard(world, rank), other.shard(world, rank)))

# (C) Count elements on this rank (to catch empty shards)
local_n = count(ds)
print(f"[rank {rank}] shard size = {local_n}")

if local_n == 0:
    # Nothing to iterate here; either reduce num_shards or use index-range splitting.
    # Exit cleanly to avoid OUT_OF_RANGE.
    import sys
    sys.exit(0)

# (D) Make it epoch-safe: repeat, then take a bounded number of steps
#     Use ragged or padded batches to handle variable-length examples.
ds = ds.repeat()  # infinite stream across epochs
# Prefer ragged_batch; otherwise use padded_batch with proper padded_shapes
if hasattr(ds, "ragged_batch"):
    ds = ds.ragged_batch(128, drop_remainder=False)
else:
    # Fallback: infer padded shapes from element_spec
    def to_padded_shape(spec):
        if isinstance(spec, tf.TensorSpec):
            return [d if d is not None else None for d in spec.shape.as_list()]
        return spec
    padded_shapes = tf.nest.map_structure(to_padded_shape, ds.element_spec)
    ds = ds.padded_batch(128, padded_shapes=padded_shapes, drop_remainder=False)

ds = ds.prefetch(tf.data.AUTOTUNE)

# (E) Consume safely: limit steps to avoid exhausting finite shards
steps_per_epoch = max(1, local_n // 128)  # or set explicitly
for step, batch in enumerate(ds.take(steps_per_epoch)):
    # work with batch
    pass
