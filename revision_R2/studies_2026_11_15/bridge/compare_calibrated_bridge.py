"""Paired exploratory comparisons for the prespecified affine-bridge control."""
import pandas as pd
import numpy as np
import evaluate_bridges as e

p = pd.read_csv(e.HERE / "predictions_long.csv", low_memory=False)
candidate = "Graph Affine bridge calibration"
controls = [
    "size_mean",
    "Log count linear",
    "Log count quadratic",
    "Within-size count ridge",
    "Outer-training size mean",
    "Outer-training Log count linear",
    "Outer-training Log count quadratic",
    "Outer-training Within-size count ridge",
    "Random forest",
    "Histogram boosting",
    "direct_graph_head",
    "Graph Affine direct calibration",
    "Graph Affine maximum calibration",
    "Graph Tail-moment ridge",
    "Graph Size-residual tail ridge",
]
outputs = []
for control in controls:
    if control not in set(p.model):
        continue
    ref = p[p.model == control][e.KEYS + ["prediction"]].rename(
        columns={"prediction": "reference_prediction"}
    )
    pair = p[p.model == candidate].merge(ref, on=e.KEYS, validate="one_to_one")
    assert np.isfinite(pair.reference_prediction).all()
    pair["size_mean_prediction"] = pair.reference_prediction
    outputs.append(
        e.bootstrap_comparisons(pair, 2000, filename=None, reference_name=control)
    )
pd.concat(outputs, ignore_index=True).to_csv(
    e.HERE / "calibrated_bridge_paired_comparisons.csv",
    index=False,
    float_format="%.17g",
)
print("Calibrated bridge comparisons complete")
