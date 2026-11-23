"""Check release code, scientific imports and source-path portability without training."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fip_release_paths import rebase_path, release_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports",
        action="store_true",
        help="Import neural modules using the installed scientific environment.",
    )
    args = parser.parse_args()
    paths = (
        list((ROOT / "examples").rglob("*.py"))
        + list((ROOT / "revision_R2").rglob("*.py"))
        + [ROOT / "fip_release_paths.py"]
    )
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    assert (
        rebase_path("/relocated/source/examples/data/full_data_v5.pt")
        == release_root() / "examples/data/full_data_v5.pt"
    )
    imported = []
    if args.imports:
        for name, relative in [
            (
                "release_replay",
                "revision_R2/studies_2026_11_15/inference/replay_saved.py",
            ),
            (
                "release_bridge",
                "revision_R2/studies_2026_11_15/bridge/evaluate_bridges.py",
            ),
        ]:
            path = ROOT / relative
            sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            imported.append(relative)
    print(
        json.dumps(
            {
                "status": "pass",
                "compiled_files": len(paths),
                "imports": imported,
                "training": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
