# Cost-aware graph batching

Graph datasets can contain samples with very different numbers of nodes. A
fixed number of graphs per batch can consequently produce large variations in
memory use and execution time. HydraGNN can instead construct batches using a
node budget:

```json
"Training": {
    "batch_size": 32,
    "Batching": {
        "mode": "node_budget",
        "max_nodes": 4096,
        "max_graphs": 64,
        "oversized_sample": "error",
        "shuffle": true,
        "seed": 0
    }
}
```

`max_nodes` limits the sum of graph nodes in a batch. `max_graphs` is an
optional secondary bound for datasets containing many small graphs. The number
of graphs per batch is therefore variable; `batch_size` is ignored in this
mode and remains available for configurations using the default `fixed` mode.
When `drop_last` is enabled, the final batch is dropped unless it reaches
either `max_nodes` or the configured `max_graphs` limit.

The `oversized_sample` policy controls graphs larger than `max_nodes`:

- `error` (default) stops immediately and identifies the sample.
- `single` places the graph alone in a batch that exceeds the budget.
- `skip` omits the graph and records it in the sampler diagnostics.

Shuffling is reproducible from `seed + epoch`. The core sampler also accepts
precomputed costs, allowing dataset implementations to avoid loading every
sample merely to determine its size.

DDStore-backed `DistDataset` instances expose their global node counts from
existing variable-shape metadata. Batch planning therefore performs no DDStore
payload reads: each rank retrieves only the samples assigned to its batches.

In distributed training, all ranks construct the same cost-bounded batches.
HydraGNN groups similarly sized batches into distributed steps, assigns one to
each rank, and rotates rank assignments to avoid repeatedly giving the largest
batch to the same rank. Batches are padded by repetition so every rank executes
the same number of optimizer steps. Setting `drop_last` discards the incomplete
final distributed step instead.

## Continuous fixed-step training

Very large training datasets can instead use the training-only streaming mode:

```json
"Training": {
    "batch_size": 32,
    "Batching": {
        "mode": "streaming_node_budget",
        "target_nodes": 3800,
        "max_nodes": 4096,
        "steps_per_epoch": 1000,
        "max_graphs": 64,
        "metadata_chunk_size": 32,
        "forward_window": 1,
        "oversized_sample": "error",
        "shuffle": true,
        "seed": 0,
        "prefetch_batches": 4
    }
}
```

Here a training epoch means exactly `steps_per_epoch` optimizer steps per rank,
not one exact dataset pass. Each rank continuously advances through its
rank-local part of a traversal and starts its next independently. Unfinished
work carries into the next training epoch. This avoids complete batch plans,
dense shuffled index arrays, preliminary count passes, padding, truncation, and
per-batch sampler collectives.

If `steps_per_epoch` is omitted, HydraGNN derives it as
`ceil(total_nodes / (target_nodes * world_size))` only when the dataset exposes
an exact stored aggregate. It never scans an old dataset to derive this value.
`drop_last` and oversampling are invalid in streaming mode. Validation and test
loaders remain finite.

The sampler cursor persists only in memory while the same loader is reused.
Sampler state is deliberately absent from checkpoints: loading model or
optimizer state starts a fresh deterministic stream and may repeat previously
trained samples. Ranks may also be in different traversal IDs and can therefore
process the same physical graph under different traversals at the same time.
See [`ComboSampler.md`](../ComboSampler.md) for detailed semantics, provenance,
advantages, drawbacks, and follow-up work.
