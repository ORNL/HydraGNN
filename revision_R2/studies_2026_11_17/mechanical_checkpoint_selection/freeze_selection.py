"""Freeze stopping-only choices and all inference inputs before outer replay."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import pandas as pd
from fip_release_paths import stored_fingerprint

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
BASE = ROOT / "revision_restart_2026_10_05/new_studies/final_repeated_evaluation"
PROBE = (
    ROOT
    / "revision_R2/studies_2026_11_16/model_redevelopment/stopping_checkpoint_probe"
)
TREE = P.parent / "mechanical_tree_baselines"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (P / "selection.json").exists(), "Do not overwrite frozen selection."
    assert not (P / "outer").exists(), "Selection must precede outer replay."
    parts = json.loads((BASE / "partitions.json").read_text())
    assert (
        sha(BASE / "partitions.json")
        == "a6c7cf5a79114d6fb296a3511b155a43ede3932ebd2d266030925ad6aa9a0c39"
    )
    prior = json.loads((PROBE / "verification.json").read_text())
    assert prior["status"] == "pass"
    batches = pd.read_csv(PROBE / "batch_losses.csv", float_precision="round_trip")
    summaries = pd.read_csv(
        PROBE / "checkpoint_summaries.csv", float_precision="round_trip"
    )
    paths = [
        Path(__file__),
        P / "PRESPEC.txt",
        BASE / "partitions.json",
        PROBE / "verification.json",
        PROBE / "probe.py",
        PROBE / "batch_losses.csv",
        PROBE / "checkpoint_summaries.csv",
        TREE / "verification.json",
        TREE / "prediction_arrays.npz",
        TREE / "metrics_by_repeat.csv",
    ]
    decisions = []
    for part, check in zip(parts, prior["fold_checks"]):
        r, f = part["repeat"], part["fold"]
        assert (r, f) == (check["repeat"], check["fold"])
        losses = {}
        caches = {}
        for name in ["best", "last"]:
            g = batches[
                (batches.repeat == r)
                & (batches.fold == f)
                & (batches.checkpoint == name)
            ]
            assert g.n_sves.sum() == 75
            assert sum([json.loads(x) for x in g.ids], []) == part["stopping"]
            assert all(abs(g.joint_loss - (g.node_loss + g.graph_loss) / 2) < 1e-6)
            losses[name] = float((g.n_sves * g.joint_loss).sum() / 75)
            old = summaries[
                (summaries.repeat == r)
                & (summaries.fold == f)
                & (summaries.checkpoint == name)
            ].iloc[0]
            assert abs(losses[name] - old.joint_loss) < 1e-14
            cache = PROBE / f"repeat{r}_fold{f}_{name}.npz"
            expected = [x for x in check["checkpoints"] if x["checkpoint"] == name][0]
            assert sha(cache) == expected["output_sha256"]
            caches[name] = str(cache)
            cp = BASE / "gnn" / f"repeat{r}_fold{f}" / "baseline" / f"{name}.pt"
            assert sha(cp) == stored_fingerprint(check["input_sha256"], cp)
            paths.extend([cache, cp])
        pair = BASE / "gnn" / f"repeat{r}_fold{f}"
        paths.extend(
            [
                pair / "preparation.json",
                pair / "fitted_preprocessing.pkl",
                pair / "fitting_statistics.npz",
                pair / "baseline/completion.json",
                pair / "baseline/stopping_grain_quality.csv",
                ROOT
                / "revision_R2/studies_2026_11_15/data"
                / f"grain_predictions_repeat{r}_fold{f}.npz",
                ROOT
                / "revision_R2/studies_2026_11_16/hierarchy_output_probe/original"
                / f"repeat{r}_fold{f}"
                / "evaluation.npz",
            ]
        )
        decisions.append(
            dict(
                repeat=r,
                fold=f,
                best_joint_loss=losses["best"],
                last_joint_loss=losses["last"],
                selected="last" if losses["last"] < losses["best"] else "best",
                stopping_ids=part["stopping"],
                evaluation_ids=part["evaluation"],
                cached_stopping=caches,
            )
        )
    out = dict(
        status="frozen_before_outer_replay",
        created_utc=datetime.now(timezone.utc).isoformat(),
        tie_rule="best",
        loss="0.5 node + 0.5 graph; graph-count-weighted batch16 mean on stopping IDs",
        decisions=decisions,
        hashes={str(p): sha(p) for p in paths},
    )
    (P / "selection.json").write_text(json.dumps(out, indent=2) + "\n")
    pd.DataFrame(
        [
            {
                k: v
                for k, v in d.items()
                if k not in ["stopping_ids", "evaluation_ids", "cached_stopping"]
            }
            for d in decisions
        ]
    ).to_csv(P / "selection.csv", index=False, float_format="%.17g")
    print(pd.read_csv(P / "selection.csv").to_string(index=False))


if __name__ == "__main__":
    main()
