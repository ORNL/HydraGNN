"""Secondary paired single-SVE inference timings on one GPU, fixed fold0 models.

Times the existing PyG payload transfer separately from resident forward passes.
Excludes disk loading, graph construction, inverse target transforms and EVT map.
"""
import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
)
from pathlib import Path
import json
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inference"))
import replay_saved as replay
import train_controls as controls

runner, joint = replay.runner, replay.joint


def main():
    torch.set_num_threads(2)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(37500 + os.getpid() % 1000),
        WORLD_SIZE="1",
        RANK="0",
    )
    torch.distributed.init_process_group("gloo", init_method="env://")
    source = replay.FROZEN / "gnn/repeat0_fold0"
    prep = json.loads((source / "preparation.json").read_text())
    cfg = prep["original_source_config"]
    part = runner.load_partition(0, 0)
    with (source / "fitted_preprocessing.pkl").open("rb") as f:
        ds = pickle.load(f)
    ds._raw = None
    ds._load()
    selected = []
    for b in range(5):
        selected.extend(
            [i for i in part["evaluation"] if int(ds._raw[i].size_bucket[0]) == b][:5]
        )
    assert len(selected) == 25
    graphs = [ds._transform(ds._raw[i]) for i in selected]
    models = {
        "graph": source / "baseline/best.pt",
        **{name: HERE / f"repeat0_fold0/{name}/best.pt" for name in controls.ARMS},
    }
    rows = []
    device = torch.device("cuda:0")
    for name, checkpoint in models.items():
        model = runner.construct(cfg, part["initialization_seed"], device)
        model.load_state_dict(
            torch.load(checkpoint, weights_only=False, map_location=device)[
                "model_state_dict"
            ],
            strict=True,
        )
        model.eval()
        values = (
            graphs
            if name == "graph"
            else controls.controls(graphs, name == "no_message_no_degree")
        )
        for i, g in zip(selected, values):
            cpu = Batch.from_data_list([g])
            gpu = cpu.clone().to(device)
            with torch.no_grad():
                for _ in range(10):
                    model(gpu)
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                forward = []
                forward_wall = []
                transfer = []
                transfer_forward = []
                for rep in range(30):
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    a.record()
                    model(gpu)
                    b.record()
                    torch.cuda.synchronize()
                    forward_wall.append((time.perf_counter() - t) * 1000)
                    forward.append(a.elapsed_time(b))
                    payload = cpu.clone()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    transferred = payload.to(device)
                    torch.cuda.synchronize()
                    middle = time.perf_counter()
                    model(transferred)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    transfer.append((middle - t) * 1000)
                    transfer_forward.append((end - t) * 1000)
                    del transferred, payload
                peak = max(0, torch.cuda.max_memory_allocated() - resident)
            for rep in range(30):
                rows.append(
                    dict(
                        model=name,
                        dataset_idx=i,
                        size_um=joint.SIZES[int(g.size_bucket[0])],
                        grain_count=g.num_nodes,
                        edge_count=g.edge_index.shape[1],
                        replicate=rep,
                        batch_size=1,
                        forward_cuda_ms=forward[rep],
                        forward_wall_ms=forward_wall[rep],
                        host_to_device_wall_ms=transfer[rep],
                        transfer_plus_forward_wall_ms=transfer_forward[rep],
                        peak_extra_allocator_bytes=peak,
                    )
                )
            del gpu, cpu
        del model
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "inference_timing_raw.csv", index=False)
    group = frame.groupby(["model", "size_um"])
    summary = group.agg(
        n_sves=("dataset_idx", "nunique"),
        n_timings=("replicate", "size"),
        median_grain_count=("grain_count", "median"),
        forward_cuda_ms=("forward_cuda_ms", "median"),
        forward_wall_ms=("forward_wall_ms", "median"),
        host_to_device_wall_ms=("host_to_device_wall_ms", "median"),
        transfer_plus_forward_wall_ms=("transfer_plus_forward_wall_ms", "median"),
    ).reset_index()
    summary.to_csv(HERE / "inference_timing_summary.csv", index=False)
    metadata = dict(
        status="complete",
        device=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        batch_size=1,
        sves_per_size=5,
        selection="first five fold0 outer-evaluation IDs in each size bucket; same identities for all three models",
        warmup_per_sve=10,
        timed_repeats_per_sve=30,
        models={
            name: dict(path=str(p), sha256=joint.sha(p)) for name, p in models.items()
        },
        scope="forward only and transfer of existing full PyG batch payload; excludes graph creation, disk IO, inverse transforms and EVT mapping; timings conditional on this hardware/runtime, not asymptotic or total-training scaling",
        payload="existing transformed PyG fields, including target metadata unused by forward; cloning done outside timing",
        cuda_timing="CUDA events and explicit synchronization; wall transfer and wall transfer-plus-forward timed separately",
        training_overlap="Concurrent workloads are not measured by this script",
        code_sha256=joint.sha(__file__),
    )
    (HERE / "inference_timing_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    torch.distributed.destroy_process_group()
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
