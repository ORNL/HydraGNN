import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import time, json, subprocess, sys, shutil, importlib.util

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    while not all(
        (HERE / f"repeat{r}_fold{f}/completion.json").exists()
        for r in range(2)
        for f in range(5)
    ):
        time.sleep(10)
    for name in ["verify_boosted.py", "analyze_boosted.py"]:
        if (
            name == "verify_boosted.py"
            and (HERE / "independent_verification.json").exists()
        ):
            assert (
                json.loads((HERE / "independent_verification.json").read_text())[
                    "status"
                ]
                == "pass"
            )
            continue
        p = subprocess.run(
            [sys.executable, str(HERE / name)],
            stdout=(HERE / (name + ".log")).open("w"),
            stderr=subprocess.STDOUT,
        )
        assert p.returncode == 0, (name, (HERE / (name + ".log")).read_text()[-2500:])
        print("COMPLETE", name, flush=True)
    snapshot = HERE / "scalar_search_snapshot.py"
    if not snapshot.exists():
        shutil.copy2(HERE.parent / "bridge_search/run_search.py", snapshot)
    spec = importlib.util.spec_from_file_location("common_boosted", snapshot)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    for family in ["hgb_mean", "hgb_q90"]:
        out = HERE / family / "shared_search"
        common.run(HERE / family / "data", out)
        protocol = json.loads((out / "protocol.json").read_text())
        protocol.update(
            encoder_family=family,
            neural_checkpoints_frozen=False,
            frozen_tree_encoder=True,
            stopping_previously_used_for_checkpoint_selection=False,
            checkpoint_note="HGB uses fixed100iterations and no stopping labels. Only scalar fitting/selection uses stopping data.",
            node_output_definition="Conditional mean"
            if family == "hgb_mean"
            else "Conditional0.90quantile under pinball loss; not a mean or a quantile across tree means.",
        )
        (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        print("SHARED SEARCH COMPLETE", family, flush=True)
    (HERE / "postprocessing_complete.json").write_text(
        json.dumps(
            dict(
                status="complete",
                families=["hgb_mean", "hgb_q90"],
                common_script_sha256=common.sha(snapshot),
                all10folds=True,
            ),
            indent=2,
        )
        + "\n"
    )
