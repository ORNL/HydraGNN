"""Run an explicitly selected published study from an assembled release workspace."""

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fip_release_paths import release_root

STUDIES = {
    "replay": "revision_R2/studies_2026_11_15/inference/replay_saved.py",
    "bridge": "revision_R2/studies_2026_11_15/bridge/evaluate_bridges.py",
    "size-baselines": "revision_R2/studies_2026_11_15/baselines/run_size_baselines.py",
    "calibration": "revision_R2/studies_2026_11_15/baselines/run_moment_calibration.py",
    "bridge-comparison": "revision_R2/studies_2026_11_15/bridge/compare_calibrated_bridge.py",
    "trees": "revision_R2/studies_2026_11_15/label_matched_trees/run_label_matched.py",
    "controls": "revision_R2/studies_2026_11_15/topology/train_controls.py",
    "conditional-gumbel": "revision_R2/studies_2026_11_15/conditional_mle/run_conditional_mle.py",
    "thresholds": "revision_R2/studies_2026_11_16/threshold_literature/run_fields.py",
    "threshold-analysis": "revision_R2/studies_2026_11_16/threshold_literature/analyze_results.py",
    "residual-head": "revision_R2/studies_2026_11_16/model_redevelopment/critical_grain/frozen_residual/study.py",
    "mechanical-selection": "revision_R2/studies_2026_11_17/mechanical_checkpoint_selection/freeze_selection.py",
    "mechanical-replay": "revision_R2/studies_2026_11_17/mechanical_checkpoint_selection/replay_outer.py",
    "mechanical-analysis": "revision_R2/studies_2026_11_17/mechanical_checkpoint_selection/analyze.py",
    "mechanical-trees": "revision_R2/studies_2026_11_17/mechanical_tree_baselines/run_study.py",
    "uncertainty": "revision_R2/studies_2026_11_17/uncertainty940_primary/capture.py",
    "uncertainty-analysis": "revision_R2/studies_2026_11_17/uncertainty940_primary/analyze.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument(
        "--show", action="store_true", help="Print the entrypoint without executing it."
    )
    args, rest = parser.parse_known_args()
    path = release_root() / STUDIES[args.study]
    if args.show:
        print(
            json.dumps(
                {"study": args.study, "entrypoint": str(path), "exists": path.is_file()}
            )
        )
        return
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path)] + (rest[1:] if rest[:1] == ["--"] else rest)
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
