"""Isolated four-arm sensitivity; original numerical modules stay immutable."""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import argparse, hashlib, importlib, json, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ARMS = {
    "lo049_n10": (-0.49, 10),
    "lo049_n4": (-0.49, 4),
    "lo090_n10": (-0.90, 10),
    "lo090_n4": (-0.90, 4),
}
CORE = None
EQD = None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {
            str(k): clean(v) for k, v in x.items() if k not in ["excess", "samples"]
        }
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def prepare():
    original = {
        name: sha(BASE / name)
        for name in ["core.py", "core_native.cpp", "eqd.py", "run_fields.py"]
    }
    manifest = {
        "arms": {},
        "original_sources": original,
        "protocol_sha256": sha(HERE / "PROTOCOL.txt"),
    }
    for name, (low, count) in ARMS.items():
        directory = HERE / name
        directory.mkdir(exist_ok=True)
        replacements = {
            "core.py": [
                ("MIN_EXCESSES=10", f"MIN_EXCESSES={count}"),
                ("SHAPE_BOUNDS=(-.49,.95)", f"SHAPE_BOUNDS=({low:.2f},.95)"),
            ],
            "core_native.cpp": [
                (
                    "constexpr double lo = -0.49, hi = 0.95;",
                    f"constexpr double lo = {low:.2f}, hi = 0.95;",
                )
            ],
            "eqd.py": [],
        }
        for filename, changes in replacements.items():
            text = (BASE / filename).read_text()
            # The base arm is an exact source copy, preserving decimal spelling.
            if name != "lo049_n10":
                for old, new in changes:
                    assert text.count(old) == 1, (filename, old)
                    text = text.replace(old, new)
            path = directory / filename
            if path.exists():
                assert path.read_text() == text, ("changed clone", path)
            else:
                path.write_text(text)
        digest = sha(directory / "core_native.cpp")[:16]
        lib = directory / f"core_native_{digest}.so"
        if not lib.exists():
            subprocess.run(
                [
                    "g++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(directory / "core_native.cpp"),
                    "-o",
                    str(lib),
                ],
                check=True,
            )
        manifest["arms"][name] = dict(
            lower_bound=low,
            minimum_count=count,
            code_hashes={
                f: sha(directory / f) for f in ["core.py", "core_native.cpp", "eqd.py"]
            },
            library_sha256=sha(lib),
        )
    (HERE / "clones_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def initialize(directory):
    global CORE, EQD
    sys.path.insert(0, str(directory))
    for name in ["core", "eqd"]:
        sys.modules.pop(name, None)
    CORE = importlib.import_module("core")
    EQD = importlib.import_module("eqd")
    assert Path(CORE.__file__).parent == Path(directory)
    CORE.build_native()


def fixed(values, threshold, quantile):
    excess = np.sort(values[values > threshold] - threshold)
    if len(excess) < CORE.MIN_EXCESSES or np.ptp(values) == 0:
        return dict(
            selected_index=-1,
            selected_quantile=quantile,
            selected_threshold=float(threshold),
            selected_count=len(excess),
            raw_maximum=float(values.max()),
            status="sparse_maximum_fallback",
        )
    candidate = dict(
        threshold=float(threshold),
        excess=excess,
        fit=CORE.fit_gpd(excess),
        maximum=float(values.max()),
        count=len(excess),
        n_total=len(values),
    )
    value, status = CORE.finite_maximum(candidate)
    return dict(
        selected_index=-1,
        selected_quantile=quantile,
        selected_threshold=float(threshold),
        selected_count=len(excess),
        raw_maximum=value,
        status=status,
        fit=candidate["fit"],
    )


def work(task):
    meta, values, seed = task
    start = time.perf_counter()
    candidates = CORE.candidates(values)
    selected = EQD.select_eqd(values, seed, candidate_list=candidates, B=100, m=500)
    j = selected.get("selected_index", -1)
    if j >= 0:
        c = next(c for c in candidates if c["index"] == j)
        selected["selected_count"] = c["count"]
        selected["fit"] = c["fit"]
    else:
        selected["selected_count"] = 0
    threshold = float(values.mean() + values.std(ddof=1))
    methods = {
        "eqd": selected,
        "fixed_q0.5": fixed(values, float(np.quantile(values, 0.5)), 0.5),
        "fixed_mean_sd": fixed(values, threshold, float(np.mean(values <= threshold))),
    }
    assert all(np.isfinite(v["raw_maximum"]) for v in methods.values())
    return clean(
        dict(
            **meta,
            seed=seed,
            seconds=time.perf_counter() - start,
            methods=methods,
            candidates=candidates,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--arm", choices=list(ARMS))
    args = parser.parse_args()
    clone_manifest = prepare()
    if args.prepare_only:
        return
    numerical = HERE / "numerical_verification.json"
    assert (
        numerical.exists() and json.loads(numerical.read_text())["status"] == "passed"
    ), "Run numerical verification before fields."
    sys.path.insert(0, str(BASE))
    from run_fields import load_tasks

    predicted, sp = load_tasks("predicted")
    observed, so = load_tasks("observed")
    tasks = predicted + observed
    sources = {**sp, **so}
    for name in [args.arm] if args.arm else ARMS:
        started = time.time()
        output = HERE / f"fields_{name}.jsonl"
        manifest_path = HERE / f"fields_{name}_manifest.json"
        manifest = dict(
            arm=name,
            tasks=len(tasks),
            sources=sources,
            clone=clone_manifest["arms"][name],
            protocol_sha256=clone_manifest["protocol_sha256"],
            runner_sha256=sha(__file__),
            numerical_verification_sha256=sha(numerical),
        )
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            for key, value in manifest.items():
                assert old[key] == value, (name, key, "changed run input")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        done = set()
        if output.exists():
            with output.open() as file:
                for line in file:
                    key = json.loads(line)["key"]
                    assert key not in done
                    done.add(key)
        pending = [t for t in tasks if t[0]["key"] not in done]
        print(
            f"{name}: {len(done)}/{len(tasks)} complete; workers={args.workers}",
            flush=True,
        )
        with output.open("a") as file, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize, initargs=(HERE / name,)
        ) as pool:
            for i, result in enumerate(pool.map(work, pending, chunksize=4), 1):
                file.write(
                    json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
                )
                if i % 500 == 0:
                    file.flush()
                    print(
                        f"{name}: {len(done)+i}/{len(tasks)} elapsed={time.time()-started:.1f}s",
                        flush=True,
                    )
        manifest.update(
            status="complete",
            completed=len(tasks),
            output_sha256=sha(output),
            seconds=time.time() - started,
            workers=args.workers,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f'{name}: complete in {manifest["seconds"]:.1f}s', flush=True)


if __name__ == "__main__":
    main()
