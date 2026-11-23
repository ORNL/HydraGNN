"""Check completion, matched settings, stopping rules and full OOF identities."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FROZEN = (
    HERE.parents[2]
    / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
)
TRUTH = np.load(FROZEN / "regressors/raw_truth940.npz")["raw_truth"]
ARMS = ["no_message_degree", "no_message_no_degree"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    details = []
    direct = []
    for name in ARMS:
        for r in range(2):
            ids = []
            grains = 0
            for f in range(5):
                source = FROZEN / f"gnn/repeat{r}_fold{f}"
                ref = json.loads((source / "baseline/completion.json").read_text())
                assert ref["checkpoint_sha256"] == sha(source / "baseline/best.pt")
                prep = json.loads((source / "preparation.json").read_text())
                dest = HERE / f"repeat{r}_fold{f}" / name
                done = json.loads((dest / "completion.json").read_text())
                assert done["status"] == "complete"
                assert done["device"] == ref["device"]
                assert done["initial_state_sha256"] == prep["full_initial_state_sha256"]
                assert done["checkpoint_sha256"] == sha(dest / "best.pt")
                assert done["preprocessing_sha256"] == sha(
                    source / "fitted_preprocessing.pkl"
                )
                assert done["permutations_sha256"] == sha(
                    source / "fitting_statistics.npz"
                )
                history = json.loads((dest / "history.json").read_text())
                scores = [x["raw_stopping_maximum_score"] for x in history]
                assert len(history) == done["actual_epochs"] <= 120
                assert int(np.argmin(scores)) == done["selected_epoch_zero_based"]
                assert (
                    len(history) == 120
                    or len(history) - 1 - done["selected_epoch_zero_based"] == 50
                )
                assert done["runtime_gate"]["status"] == "pass"
                assert done["runtime_gate"]["full_parameters"] == 377974
                path = HERE / name / f"data/grain_predictions_repeat{r}_fold{f}.npz"
                meta = json.loads(path.with_suffix(".json").read_text())
                assert meta["archive_sha256"] == sha(path)
                z = np.load(path)
                assert np.array_equal(z["y_true"], TRUTH[z["dataset_idx"]])
                assert (
                    np.isfinite(z["pred_peak"]).all()
                    and np.isfinite(z["direct_prediction"]).all()
                )
                ids.extend(z["dataset_idx"].tolist())
                grains += len(z["pred_peak"])
                direct.append(pd.read_csv(dest / "evaluation_predictions.csv"))
                details.append(
                    {
                        k: done[k]
                        for k in [
                            "model",
                            "repeat",
                            "fold",
                            "device",
                            "actual_epochs",
                            "selected_epoch_zero_based",
                            "elapsed_seconds",
                            "checkpoint_sha256",
                        ]
                    }
                )
            assert sorted(ids) == list(range(940)) and grains == 467390
    frame = pd.concat(direct, ignore_index=True)
    assert not frame.duplicated(["model", "repeat", "dataset_idx", "case"]).any()
    frame.to_csv(HERE / "direct_predictions_long.csv", index=False)
    pd.DataFrame(details).to_csv(HERE / "training_summary.csv", index=False)
    result = dict(
        status="pass",
        models=20,
        pairs=10,
        unique_sves_per_repeat=940,
        grains_per_repeat=467390,
        all_devices_initial_states_preprocessing_permutations_matched=True,
        all_stopping_rules_verified=True,
        all_raw_truth_matches=True,
        source_checkpoints_unchanged=True,
        nominal_parameters=377974,
        inactive_neighbor_weight_parameters=139264,
        scope="No explicit adjacency in both controls; explicit degree masked at training and inference in no_message_no_degree. Geometric descriptors and permutation-invariant pooling retained. Effective functional capacity is reduced by removing neighbor operations; nominal architecture/initial weights are matched.",
        models_detail=details,
    )
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
