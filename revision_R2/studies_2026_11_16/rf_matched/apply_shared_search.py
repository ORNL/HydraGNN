"""Apply the common stopping-CV scalar search to one fixed forest encoder."""
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
import sys

sys.dont_write_bytecode = True
import argparse, importlib.util, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    args = p.parse_args()
    source = HERE.parent / "bridge_search/run_search.py"
    spec = importlib.util.spec_from_file_location("shared_scalar_search", source)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    out = HERE / args.family / "shared_search"
    common.run(HERE / args.family / "data", out)
    path = out / "protocol.json"
    protocol = json.loads(path.read_text())
    protocol.update(
        encoder_family=args.family,
        neural_checkpoints_frozen=False,
        frozen_tree_encoder=True,
        stopping_previously_used_for_checkpoint_selection=False,
        checkpoint_note="Tree hyperparameters fixed before fitting; no stopping labels used by grain or fitting-only direct forest. The stopping set fits and selects scalar readout maps only.",
        direct_feature_note="direct_head is the direct SVE forest fitted exclusively on original fitting SVEs; no union-direct stopping predictions used.",
    )
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    helpers = {}
    for name in ["pure_selections.py", "run_ensembles.py"]:
        helper = source.parent / name
        subprocess.run([sys.executable, str(helper), str(out)], check=True)
        helpers[name] = common.sha(helper)
    (out / "rf_context.json").write_text(
        json.dumps(
            dict(
                status="complete",
                family=args.family,
                common_script_sha256=common.sha(source),
                wrapper_sha256=common.sha(__file__),
                helper_sha256=helpers,
                no_stopping_labels_in_forest_fit=True,
                no_outer_labels_in_fit_or_selection=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print("SHARED SEARCH COMPLETE", args.family, flush=True)
