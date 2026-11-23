"""Replay an explicitly identified published residual head without training."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fip_release_paths import release_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--arm", choices=("bulk_control", "critical_treatment"), required=True
    )
    parser.add_argument("--source-version", default="55f907da468ecf20")
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="SHA-256 from the published data manifest.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify and strictly load the head; no grain or SVE inference.",
    )
    args = parser.parse_args()
    import numpy as np
    import torch

    directory = (
        release_root()
        / "revision_R2/studies_2026_11_16/model_redevelopment/critical_grain/frozen_residual"
    )
    sys.path.insert(0, str(directory))
    from model import FrozenResidualHead

    checkpoint = (
        directory
        / "runs"
        / args.source_version
        / args.arm
        / f"repeat{args.repeat}_fold{args.fold}"
        / "best.pt"
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != args.checkpoint_sha256:
        raise ValueError("Checkpoint differs from the published manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    head = FrozenResidualHead(
        state["grain_variance"],
        state["maximum_variance"],
        state["size_weights"],
        2026111700 + 10 * args.repeat + args.fold,
    )
    head.load_state_dict(state, strict=True)
    head = head.to(args.device).eval().requires_grad_(False)
    report = {
        "source_version": args.source_version,
        "repeat": args.repeat,
        "fold": args.fold,
        "arm": args.arm,
        "checkpoint_sha256": digest,
        "strict_load": True,
        "training": False,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    if not args.check_only:
        import cache

        part = next(
            p
            for p in cache.legacy.load_partitions()
            if p["repeat"] == args.repeat and p["fold"] == args.fold
        )
        inp = cache.inputs(part)
        original, original_info = cache.original_model(inp, torch.device(args.device))
        # The published seal identifies selection before outer evaluation. Rebase
        # its path for this workspace; preserve its checkpoint identity and hash.
        seal = json.loads(checkpoint.with_name("selected_head.json").read_text())
        assert (
            seal["selected_head_sha256"] == digest and seal["strict_reloaded"] is True
        )
        seal["selected_head_checkpoint"] = str(checkpoint)
        seal_path = args.out / "selected_head.json"
        seal_path.write_text(json.dumps(seal, indent=2))
        values, cache_report = cache.cache_subset(
            inp,
            original,
            "evaluation",
            torch.device(args.device),
            selection_seal=seal_path,
        )
        buckets = np.repeat(
            values["size_bucket"].numpy(), values["grain_count"].numpy()
        )
        chunks = []
        with torch.no_grad():
            for first in range(0, len(buckets), 16384):
                sl = slice(first, first + 16384)
                chunks.append(
                    head(
                        values["features"][sl].to(args.device),
                        values["original_prediction"][sl].to(args.device),
                        torch.as_tensor(
                            buckets[sl], dtype=torch.long, device=args.device
                        ),
                    )
                    .cpu()
                    .numpy()
                )
        prediction = np.concatenate(chunks)
        np.savez_compressed(
            args.out / "grain_predictions.npz",
            pred_peak=prediction,
            true_peak=values["true_peak"].numpy(),
            dataset_idx=values["dataset_idx"].numpy(),
            offsets=values["offsets"].numpy(),
            y_true=values["y_true"].numpy(),
            size_bucket=values["size_bucket"].numpy(),
        )
        report.update(
            original_checkpoint=original_info,
            cache=cache_report,
            grains=len(prediction),
            sves=len(values["dataset_idx"]),
        )
    (args.out / "replay.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
