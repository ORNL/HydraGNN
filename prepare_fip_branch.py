"""Apply the allowlisted FIP release to a clean HydraGNN checkout; never push."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_COMMIT = "63cfee31cd93266691b971e6de521bb7b17dabd1"
OVERLAY = Path(__file__).resolve().parent
IGNORE = """
# FIP publication outputs and downloaded simulation/model data
/revision_R2/**
/revision_restart_2026_10_05/**
/revision_final/**
/examples/cpfem_fip_data/outputs/
/examples/data/
/model_ready_data/
/Supplementary_data/
**/*.h5
**/fitted_preprocessing.pkl
**/best.pt
**/last.pt
**/*.npz
**/*.npy
**/*.pkl
**/*.pk
**/*.csv
**/*.csv.gz
**/__pycache__/
"""


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--branch", default="paper/fip-r2-release")
    parser.add_argument("--verify-overlay", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((OVERLAY / "FIP_CODE_MANIFEST.json").read_text())
    files = manifest["files"]
    for relative, record in files.items():
        source = (OVERLAY / relative).resolve()
        if not source.is_relative_to(OVERLAY):
            raise ValueError(f"Unsafe manifest path: {relative}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Release file changed: {relative}")
        if source.suffix in (
            ".pt",
            ".pk",
            ".pkl",
            ".npz",
            ".npy",
            ".csv",
            ".gz",
            ".zip",
        ):
            raise ValueError(
                f"Scientific data does not belong in this Git overlay: {relative}"
            )
    if args.verify_overlay:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "verified_code_files": len(files),
                    "git_changes": False,
                }
            )
        )
        return
    if args.checkout is None:
        parser.error("--checkout is required unless --verify-overlay is used")
    repo = args.checkout.resolve()
    if git(repo, "rev-parse", "HEAD") != BASE_COMMIT:
        raise ValueError(f"Start from published base commit {BASE_COMMIT}")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            "Checkout is not clean. Preserve its work and use a clean checkout."
        )
    git(repo, "switch", "-c", args.branch)
    paths = [*files, "FIP_CODE_MANIFEST.json"]
    for relative in paths:
        source = OVERLAY / relative
        destination = (repo / relative).resolve()
        if not destination.is_relative_to(repo):
            raise ValueError(f"Unsafe destination: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    ignore = repo / ".gitignore"
    ignore.write_text(ignore.read_text() + "\n" + IGNORE)
    for start in range(0, len(paths), 50):
        git(repo, "add", "-f", "--", *paths[start : start + 50])
    git(repo, "add", "--", ".gitignore")
    staged = set(git(repo, "diff", "--cached", "--name-only").splitlines())
    if not staged <= set(paths) | {".gitignore"}:
        raise RuntimeError("Unexpected staged files")
    print(git(repo, "diff", "--cached", "--stat"))
    print(
        "Local branch prepared. Review and commit the staged changes, then push when ready."
    )


if __name__ == "__main__":
    main()
