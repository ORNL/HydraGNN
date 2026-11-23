"""promote_winners.py — re-run cells that beat thresholds with extra seeds.

After Tracks A/L complete with seed=42, promote any cell whose seed-42
score satisfies `manifest.PROMOTE_THRESHOLDS` (raw scores) to extra seeds
in `manifest.PROMOTE_EXTRA_SEEDS` (default [137, 2024]).

Each promoted cell becomes a 3-seed (42, 137, 2024) ensemble.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _runner_utils import run_cell, cell_score, qualifies_for_promotion
from manifest import BASE_RECIPE, TRACK_A, TRACK_L, PROMOTE_EXTRA_SEEDS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_promotions", type=int, default=8)
    ap.add_argument("--extra_seeds", type=int, nargs="+", default=PROMOTE_EXTRA_SEEDS)
    ap.add_argument("--num_epoch", type=int, default=None)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no_skip_done", action="store_true")
    args = ap.parse_args()

    qualified = []
    for cell in TRACK_A + TRACK_L:
        if qualifies_for_promotion(cell["tag"], 42):
            qualified.append(cell)
    qualified.sort(key=lambda c: -cell_score(c["tag"], 42).get("pgmax_pearson", 0.0))
    qualified = qualified[: args.max_promotions]
    print(
        f"[promote] {len(qualified)} cells qualified: {[c['tag'] for c in qualified]}",
        flush=True,
    )

    # Shard work across slots: flatten (cell, seed) pairs.
    pairs = [(c, s) for c in qualified for s in args.extra_seeds]
    pairs = pairs[args.start_idx :: args.stride]

    for cell, seed in pairs:
        try:
            run_cell(
                cell["tag"],
                seed,
                BASE_RECIPE,
                cell["extra"],
                num_epoch=args.num_epoch,
                skip_done=not args.no_skip_done,
            )
        except Exception as e:
            import traceback

            print(f"[promote] {cell['tag']}_seed{seed} FAILED: {e}", flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
