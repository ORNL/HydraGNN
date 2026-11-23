"""Prediction-only threshold extraction, with resumable source-hashed records."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import time
import numpy as np
import core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "revision_R2/studies_2026_11_15/data"
PARTITIONS = (
    ROOT
    / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation/partitions.json"
)
CASES = ["lcf_fs", "lcf_wp", "hcf_fs", "hcf_wp"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {
            str(k): clean(v) for k, v in value.items() if k not in {"excess", "samples"}
        }
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_tasks(kind):
    """Keep outcome metadata outside all selector call arguments."""
    tasks = []
    sources = {str(PARTITIONS): sha(PARTITIONS)}
    observed = {}
    for part in json.loads(PARTITIONS.read_text()):
        r, f = part["repeat"], part["fold"]
        assert not set(part["evaluation"]) & (
            set(part["fitting"]) | set(part["stopping"])
        )
        for subset, prefix in [("stopping", "stopping_"), ("evaluation", "")]:
            if kind == "observed" and (r != 0 or subset != "evaluation"):
                continue
            path = DATA / f"grain_predictions_{prefix}repeat{r}_fold{f}.npz"
            sources[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as archive:
                z = {key: archive[key] for key in archive.files}
                assert list(z["cases"]) == CASES
                assert set(z["dataset_idx"]) == set(part[subset])
                for j, idx in enumerate(z["dataset_idx"]):
                    a, b = z["offsets"][j : j + 2]
                    assert b - a == z["grain_count"][j]
                    for c, case in enumerate(CASES):
                        truth = float(z["y_true"][j, c])
                        assert np.isclose(
                            z["true_peak"][a:b, c].max(), truth, rtol=2e-6
                        )
                        x = np.array(
                            z["pred_peak" if kind == "predicted" else "true_peak"][
                                a:b, c
                            ],
                            dtype=float,
                        )
                        assert np.isfinite(x).all() and len(x) >= 2
                        meta = dict(
                            kind=kind,
                            repeat=int(r),
                            fold=int(f),
                            subset=subset,
                            dataset_idx=int(idx),
                            case=case,
                            size_um=int(z["size_um"][j]),
                            grain_count=int(b - a),
                            truth=truth,
                        )
                        key = f"{kind}_{r}_{f}_{subset}_{int(idx)}_{case}"
                        meta["key"] = key
                        meta["field_sha256"] = hashlib.sha256(x.tobytes()).hexdigest()
                        # Distinct, stable seeds. Physical observed fields are processed once.
                        seed = (
                            20261116
                            + (100000000 if kind == "observed" else 0)
                            + int(r) * 1000000
                            + int(f) * 100000
                            + int(idx) * 10
                            + c
                        )
                        tasks.append((meta, x, seed))
                        if kind == "observed":
                            assert (int(idx), case) not in observed
                            observed[(int(idx), case)] = True
    if kind == "observed":
        assert len(tasks) == 940 * 4
    else:
        assert sum(t[0]["subset"] == "evaluation" for t in tasks) == 940 * 2 * 4
    return tasks, sources


def one_fixed(x, threshold, quantile):
    y = np.sort(x[x > threshold] - threshold)
    if len(y) < core.MIN_EXCESSES or np.ptp(x) == 0:
        return dict(
            selected_index=-1,
            selected_quantile=quantile,
            selected_threshold=float(threshold),
            selected_count=len(y),
            raw_maximum=float(x.max()),
            status="sparse_maximum_fallback",
        )
    candidate = dict(
        threshold=float(threshold),
        excess=y,
        fit=core.fit_gpd(y),
        maximum=float(x.max()),
        count=len(y),
        n_total=len(x),
    )
    value, status = core.finite_maximum(candidate)
    return dict(
        selected_index=-1,
        selected_quantile=quantile,
        selected_threshold=float(threshold),
        selected_count=len(y),
        raw_maximum=value,
        status=status,
        fit=candidate["fit"],
    )


def work(task):
    family, meta, values, seed = task
    start = time.perf_counter()
    cs = core.candidates(values)
    if family == "fixed":
        methods = {
            f"fixed_q{q:g}": one_fixed(values, float(np.quantile(values, q)), q)
            for q in [0.5, 0.75, 0.9]
        }
        u = float(values.mean() + values.std(ddof=1))
        methods["fixed_mean_sd"] = one_fixed(values, u, float(np.mean(values <= u)))
        methods["grain_maximum"] = dict(
            raw_maximum=float(values.max()),
            status="direct_maximum",
            selected_count=1,
            selected_quantile=1.0,
            selected_threshold=float(values.max()),
        )
    elif family == "eqd":
        from eqd import select_eqd

        methods = {"eqd": select_eqd(values, seed, candidate_list=cs, B=100, m=500)}
    elif family == "diagnostics":
        from diagnostics import select_diagnostics

        methods = select_diagnostics(values, seed, candidate_cache=cs)
    elif family == "predictive":
        from predictive import select_predictive

        methods = select_predictive(values, seed, candidate_cache=cs)
    else:
        raise ValueError(family)
    for name, result in methods.items():
        assert np.isfinite(result["raw_maximum"]), (meta["key"], name, result)
        if "threshold" in result:
            result.setdefault("selected_threshold", result["threshold"])
        if "quantile" in result:
            result.setdefault("selected_quantile", result["quantile"])
        j = result.get("selected_index", -1)
        if j is not None and j >= 0:
            selected = next(c for c in cs if c["index"] == j)
            result.setdefault("selected_count", selected["count"])
            result.setdefault("selected_quantile", selected["quantile"])
            result.setdefault("selected_threshold", selected["threshold"])
            result.setdefault("fit", selected["fit"])
        else:
            result.setdefault("selected_count", 0)
    return clean(
        dict(
            **meta,
            family=family,
            seed=seed,
            seconds=time.perf_counter() - start,
            candidates=cs,
            methods=methods,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family", choices=["fixed", "eqd", "diagnostics", "predictive"], required=True
    )
    p.add_argument("--kind", choices=["predicted", "observed"], required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path, default=HERE / "field_results")
    args = p.parse_args()
    start = time.time()
    core.build_native()
    tasks, sources = load_tasks(args.kind)
    if args.limit:
        # A size-balanced numerical smoke test, never a performance selection.
        tasks = sorted(tasks, key=lambda t: (t[0]["dataset_idx"], t[0]["case"]))
        indices = np.linspace(0, len(tasks) - 1, min(args.limit, len(tasks)), dtype=int)
        tasks = [tasks[i] for i in indices]
    code = ["core.py", "core_native.cpp", "run_fields.py"]
    if args.family != "fixed":
        code.append(
            {
                "eqd": "eqd.py",
                "diagnostics": "diagnostics.py",
                "predictive": "predictive.py",
            }[args.family]
        )
    code_hashes = {f: sha(HERE / f) for f in code}
    if args.family == "diagnostics":
        for candidate in (
            (HERE / "diagnostic_reference").glob("*")
            if (HERE / "diagnostic_reference").exists()
            else []
        ):
            if candidate.is_file():
                sources[str(candidate)] = sha(candidate)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.kind}_{args.family}"
    manifest_path = args.output / f"{stem}_manifest.json"
    manifest = dict(
        kind=args.kind,
        family=args.family,
        tasks=len(tasks),
        sources=sources,
        code_hashes=code_hashes,
        limit=args.limit,
        seed_rule="20261116 + kind*1e8 + repeat*1e6 + fold*1e5 + idx*10 + case",
    )
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ["sources", "code_hashes", "tasks", "limit"]:
            assert (
                previous[key] == manifest[key]
            ), f"Changed {key}; use a new output directory."
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = args.output / f"{stem}.jsonl"
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            d = json.loads(line)
            assert d["key"] not in done
            done.add(d["key"])
    assert done <= {t[0]["key"] for t in tasks}
    pending = [(args.family, *task) for task in tasks if task[0]["key"] not in done]
    print(
        json.dumps(
            dict(
                family=args.family,
                kind=args.kind,
                total=len(tasks),
                already_done=len(done),
                workers=args.workers,
            )
        ),
        flush=True,
    )
    with output.open("a") as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(work, pending, chunksize=4), 1):
            out.write(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
            if i % 100 == 0:
                out.flush()
                print(
                    f"{stem}: {len(done)+i}/{len(tasks)} elapsed={time.time()-start:.1f}s",
                    flush=True,
                )
    manifest.update(
        status="complete",
        completed=len(tasks),
        output_sha256=sha(output),
        elapsed_seconds=time.time() - start,
        workers=args.workers,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
