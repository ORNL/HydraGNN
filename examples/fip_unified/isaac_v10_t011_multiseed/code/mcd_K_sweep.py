"""mcd_K_sweep.py — convergence sweep of MCD K (number of stochastic passes).

Runs K_max forward passes once on the chosen rep, then computes σ_e on
prefixes K ∈ {5, 10, 20, 30, 50, 75, 100, 150, 200} (clipped to K_max).
Outputs:

  results/<rep>/mcd_K_sweep.json         — σ_e per K per cell (max + logsigma channels)
  results/<rep>/mcd_K_sweep.png/.pdf     — convergence plot

Reuses _rebuild_model, _activate_dropout, _one_pass from run_mcd_inference_v10.

Usage:
  python mcd_K_sweep.py                  # split=42, K_max=200 (default)
  python mcd_K_sweep.py --split 42 --K_max 200
  python mcd_K_sweep.py --split 7 --K_max 100
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

from run_mcd_inference_v10 import (
    _rebuild_model,
    _activate_dropout,
    _one_pass,
    _ensure_distributed,
    CELL_KEYS,
)

WINNER_TAG = "t011"
K_GRID = [5, 10, 20, 30, 50, 75, 100, 150, 200]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=int, default=42)
    ap.add_argument("--init", type=int, default=42)
    ap.add_argument("--K_max", type=int, default=200)
    ap.add_argument(
        "--save_full_passes",
        action="store_true",
        help="Persist the full (K_max, n_grains, n_total) passes array (large).",
    )
    args = ap.parse_args()

    res_dir = STUDY / "results" / f"{WINNER_TAG}_split{args.split}_seed{args.init}"
    if not res_dir.exists():
        print(f"  ERROR: {res_dir} does not exist")
        return 1

    # log_dir lives in STUDY/logs/, mirroring run_mcd_inference_v10.run_one resolution
    log_dir = STUDY / "logs" / res_dir.name
    if not log_dir.exists():
        log_dir = STUDY / "logs" / f"v4_{res_dir.name}"
    if not log_dir.exists():
        for cand in (STUDY / "logs").glob(f"*{res_dir.name}*"):
            log_dir = cand
            break
    if not log_dir.exists():
        print(f"  ERROR: log_dir not found for {res_dir.name}")
        return 1
    print(f"  [load] {res_dir.name} -> {log_dir.name}")

    _ensure_distributed()
    print(f"[K sweep] rep = {res_dir.name}")
    print(
        f"[K sweep] K_max = {args.K_max}; K grid = "
        f"{[k for k in K_GRID if k <= args.K_max]}"
    )

    model, fipds, test_loader, cfg = _rebuild_model(res_dir, log_dir)
    n_drop = _activate_dropout(model)
    print(f"  dropout layers active: {n_drop}")
    if n_drop == 0:
        print("  no dropout — sweep is degenerate; aborting")
        return 1

    n_super = len(cfg["Recipe"].get("per_grain_targets") or [])
    if n_super == 0:
        n_super = 1 + len(cfg["Recipe"].get("per_grain_aux_targets") or [])
    n_sigma = 4 if cfg["Recipe"].get("use_heteroscedastic_max") else 0
    n_total = n_super + n_sigma
    print(f"  head shape: n_super={n_super}, n_sigma={n_sigma}, n_total={n_total}")

    device = next(model.parameters()).device

    # Run K_max passes; accumulate the per-pass arrays
    passes = []
    for k in range(args.K_max):
        pn_arr, _pg_arr = _one_pass(model, test_loader, device, n_super)
        passes.append(pn_arr)
        if (k + 1) % 10 == 0:
            print(f"    pass {k+1}/{args.K_max}", flush=True)
    P = np.stack(passes, axis=0)  # (K_max, n_grains, n_total)
    print(f"  full passes array: {P.shape}")

    if args.save_full_passes:
        np.savez_compressed(
            res_dir / "mcd_K_sweep_passes.npz", P=P, K_max=args.K_max, n_super=n_super
        )

    # Compute σ_e on prefixes
    Ks = [k for k in K_GRID if k <= args.K_max]
    sigma_e_at_K = {c: {"max": [], "logsigma": []} for c in CELL_KEYS}
    for K_sub in Ks:
        std_sub = P[:K_sub].std(axis=0)  # (n_grains, n_total)
        for i, c in enumerate(CELL_KEYS):
            sigma_e_at_K[c]["max"].append(float(std_sub[:, i].mean()))
            if n_super == 16:
                sigma_e_at_K[c]["logsigma"].append(float(std_sub[:, 12 + i].mean()))

    # Reference: K_max
    ref_std = P.std(axis=0)
    ref_max = {c: float(ref_std[:, i].mean()) for i, c in enumerate(CELL_KEYS)}
    ref_logsig = (
        {c: float(ref_std[:, 12 + i].mean()) for i, c in enumerate(CELL_KEYS)}
        if n_super == 16
        else {}
    )

    # Save JSON
    out_json = {
        "rep": res_dir.name,
        "K_max": args.K_max,
        "K_grid": Ks,
        "n_super": n_super,
        "n_grains": int(P.shape[1]),
        "sigma_e_max_channel": {c: sigma_e_at_K[c]["max"] for c in CELL_KEYS},
        "sigma_e_logsigma_channel": {c: sigma_e_at_K[c]["logsigma"] for c in CELL_KEYS},
        "ref_K_max_max_channel": ref_max,
        "ref_K_max_logsigma_channel": ref_logsig,
    }
    out_p = res_dir / "mcd_K_sweep.json"
    out_p.write_text(json.dumps(out_json, indent=2))
    print(f"  saved {out_p}")

    # Convergence plot
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharex=True)

    ax = axes[0]
    for i, c in enumerate(CELL_KEYS):
        ys = sigma_e_at_K[c]["max"]
        rel = [abs(y - ref_max[c]) / ref_max[c] * 100 for y in ys]
        ax.plot(Ks, rel, "-o", ms=4, label=c.upper().replace("_", "/"))
    ax.set_xlabel("K (number of MCD passes)")
    ax.set_ylabel(r"$|σ_e(K) - σ_e(K_{max})| / σ_e(K_{max})$  (%)")
    ax.set_title(f"σ_e convergence — per-grain MAX channels (rep {res_dir.name})")
    ax.axhline(5, color="0.5", ls=":", lw=0.7, label="5% threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    ax = axes[1]
    if n_super == 16:
        for i, c in enumerate(CELL_KEYS):
            ys = sigma_e_at_K[c]["logsigma"]
            rel = [abs(y - ref_logsig[c]) / ref_logsig[c] * 100 for y in ys]
            ax.plot(Ks, rel, "-o", ms=4, label=c.upper().replace("_", "/"))
        ax.set_title("σ_e convergence — log-σ channels")
        ax.axhline(5, color="0.5", ls=":", lw=0.7)
    ax.set_xlabel("K (number of MCD passes)")
    ax.set_ylabel(r"$|σ_e(K) - σ_e(K_{max})| / σ_e(K_{max})$  (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(res_dir / "mcd_K_sweep.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(res_dir / "mcd_K_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {res_dir / 'mcd_K_sweep.pdf'} (+ .png)")

    # Print recommendation
    print()
    print("=== Convergence summary (per-grain MAX channels) ===")
    print(f"{'K':>5} " + " ".join(f"{c:>10}" for c in CELL_KEYS))
    for j, K_sub in enumerate(Ks):
        rels = [
            abs(sigma_e_at_K[c]["max"][j] - ref_max[c]) / ref_max[c] * 100
            for c in CELL_KEYS
        ]
        print(f"{K_sub:>5} " + " ".join(f"{r:>9.1f}%" for r in rels))
    # Pick smallest K where ALL cells are < 5%
    for j, K_sub in enumerate(Ks):
        rels = [
            abs(sigma_e_at_K[c]["max"][j] - ref_max[c]) / ref_max[c] * 100
            for c in CELL_KEYS
        ]
        if max(rels) < 5.0:
            print(f"\n  Recommended K (max-channel < 5% deviation): {K_sub}")
            break
    else:
        print(f"\n  No K under 5% — consider K > {args.K_max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
