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
