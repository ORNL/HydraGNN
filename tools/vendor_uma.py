#!/usr/bin/env python3
##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################
"""Vendor fairchem.core UMA closure into HydraGNN.

Copies the enumerated closure under
  hydragnn/utils/model/uma/_vendored/fairchem/core/**
then rewrites all `from fairchem.core.X` / `import fairchem.core.X`
occurrences to point at the vendored location and prepends the combined
MIT/BSD-3-Clause copyright and SPDX header.

Also copies non-.py resources (Jd.pt, wigner_d_coefficients.pt,
pretrained_models.json) that live next to visited modules.

Adds a marker file at each vendored subtree root so we know its
provenance (MIT license, upstream commit).
"""

from __future__ import annotations
import argparse
import ast
import datetime as dt
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO / "hydragnn" / "utils" / "model" / "uma" / "_vendored"

SEEDS = [
    "fairchem.core.models.uma.escn_md",
    "fairchem.core.models.uma.escn_md_block",
    "fairchem.core.models.uma.escn_moe",
    "fairchem.core.models.uma.outputs",
]

RESOURCE_EXTS = {".pt", ".yaml", ".yml", ".json"}

DUAL_LICENSE_HEADER = """\
##############################################################################
# Copyright (c) 2026, Oak Ridge National Laboratory                          #
# Copyright (c) Meta Platforms, Inc. and affiliates.                         #
#                                                                            #
# Portions derived from FAIR-Chem are distributed under the MIT License;     #
# HydraGNN modifications are distributed under the BSD 3-clause license.     #
# Original upstream copyright and license notices are preserved below.       #
#                                                                            #
# SPDX-License-Identifier: MIT AND BSD-3-Clause                              #
##############################################################################
"""


# ---------- discovery ----------


def module_file(mod: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(mod)
    except (ModuleNotFoundError, ValueError, ImportError):
        return None
    if spec is None or spec.origin in (None, "built-in"):
        return None
    return Path(spec.origin)


def extract_imports(path: Path, mod_name: str | None = None) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            if node.level > 0:
                # Relative import: resolve against mod_name
                if mod_name is None:
                    continue
                parts = mod_name.split(".")
                # For a module file X.Y.Z, level=1 means package X.Y
                base_parts = parts[: -node.level] if len(parts) >= node.level else []
                if node.module:
                    base_parts = base_parts + node.module.split(".")
                if not base_parts:
                    continue
                base = ".".join(base_parts)
            else:
                base = node.module
            out.add(base)
            for alias in node.names:
                out.add(f"{base}.{alias.name}")
    return out


def is_fairchem(mod: str) -> bool:
    return mod == "fairchem" or mod.startswith("fairchem.")


def closure() -> tuple[set[Path], set[Path], Path]:
    visited_mods: set[str] = set()
    visited_files: set[Path] = set()
    queue: deque[str] = deque(SEEDS)

    while queue:
        mod = queue.popleft()
        if mod in visited_mods:
            continue
        visited_mods.add(mod)

        path = module_file(mod)
        if path is None and "." in mod:
            parent, _, _ = mod.rpartition(".")
            path = module_file(parent)
            if path is not None:
                visited_mods.add(parent)
        if path is None:
            continue
        visited_files.add(path)
        for imp in extract_imports(path, mod_name=mod):
            if is_fairchem(imp) and imp not in visited_mods:
                queue.append(imp)

    # Also ensure every parent __init__ up to fairchem/core is included
    # (so packages import correctly)
    package_inits: set[Path] = set()
    fc_root: Path | None = None
    for p in visited_files:
        # walk up to find "fairchem" dir
        cur = p.parent
        while cur.name != "core" or cur.parent.name != "fairchem":
            cur = cur.parent
            if cur == cur.parent:
                break
        fc_root = cur
        break
    if fc_root is None:
        raise RuntimeError("could not locate fairchem/core root")

    for p in list(visited_files):
        cur = p.parent
        while True:
            init = cur / "__init__.py"
            if init.exists():
                package_inits.add(init)
            if cur == fc_root.parent:  # stop after fairchem/
                break
            if cur == cur.parent:
                break
            cur = cur.parent
    visited_files |= package_inits

    # Resources next to visited files
    resources: set[Path] = set()
    for p in visited_files:
        for sib in p.parent.iterdir():
            if sib.is_file() and sib.suffix in RESOURCE_EXTS:
                resources.add(sib)

    return visited_files, resources, fc_root.parent.parent  # site-packages root


# ---------- copy + rewrite ----------

IMPORT_RE = re.compile(r"\bfairchem\.core\b")

MARKER = """# NOTE: This tree was derived from fairchem-core (Meta MIT-licensed)
# for use inside HydraGNN. Imports and package initializers are transformed by
# tools/vendor_uma.py; see PROVENANCE.json for the pinned source and details.
# Do not edit these files directly; re-run vendor_uma.py to refresh.
"""


def relpath_under_fairchem(src: Path, fairchem_root: Path) -> Path:
    return src.relative_to(fairchem_root)


def copy_and_rewrite(src: Path, fairchem_root: Path) -> Path:
    rel = relpath_under_fairchem(src, fairchem_root)
    dst = VENDOR_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".py":
        # Neutralize package __init__.py files: original ones do eager
        # side-effect imports (registry, pretrained_mlip, etc.) that pull
        # in modules outside our closure. We only need __init__ presence.
        if src.name == "__init__.py":
            dst.write_text(
                DUAL_LICENSE_HEADER
                + "# Vendored from fairchem-core (MIT). Original __init__ contents\n"
                "# stripped to avoid eager imports of modules outside the\n"
                "# UMA forward-pass closure.\n"
            )
            return dst
        text = src.read_text()
        text = IMPORT_RE.sub("hydragnn.utils.model.uma._vendored.fairchem.core", text)
        dst.write_text(DUAL_LICENSE_HEADER + text)
    else:
        shutil.copy2(src, dst)
    return dst


def ensure_init(dirpath: Path):
    init = dirpath / "__init__.py"
    if not init.exists():
        init.write_text(DUAL_LICENSE_HEADER + MARKER)


# ---------- post-vendor patches ----------
# The vendored UMA backbone is copied verbatim, but a few module-level imports
# pull heavy fairchem-core deps (torchtnt/ray/wandb via ``mlip_unit``; ``hydra``
# via ``models/base``) that are never reached during import/construct/forward --
# they live only in checkpoint / inference code paths that HydraGNN does not
# use, so we defer them to their single use sites. The remaining lightweight
# import-time deps (``omegaconf``, ``monty``) are declared in
# requirements-specific-models.txt and imported normally by the vendored code.
# Each patch is (relpath-under-fairchem/core, old, new) applied to the rewritten
# text; every ``old`` must match exactly once or the run aborts.
PATCHES: list[tuple[str, str, str]] = [
    # escn_md: defer OutputSpec/Task import (drags in the whole mlip_unit infra)
    (
        "models/uma/escn_md.py",
        "from hydragnn.utils.model.uma._vendored.fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task\n\nfrom .escn_md_block import eSCNMD_Block",
        "from .escn_md_block import eSCNMD_Block",
    ),
    (
        "models/uma/escn_md.py",
        "        # Direct force models can't compute stress via autograd\n"
        "        if self.direct_forces:\n"
        "            return []\n"
        "\n"
        "        tasks = []",
        "        # Direct force models can't compute stress via autograd\n"
        "        if self.direct_forces:\n"
        "            return []\n"
        "\n"
        "        # Imported lazily: mlip_unit pulls the fairchem training/inference\n"
        "        # infra (torchtnt/ray/wandb); this method is only reached for\n"
        "        # checkpoint-based inference, never during construct/forward.\n"
        "        from hydragnn.utils.model.uma._vendored.fairchem.core.units.mlip_unit.mlip_unit import (\n"
        "            OutputSpec,\n"
        "            Task,\n"
        "        )\n"
        "\n"
        "        tasks = []",
    ),
    # models/base: defer hydra import (only used for checkpoint task instantiate)
    (
        "models/base.py",
        "from typing import TYPE_CHECKING\n\nimport hydra\nimport torch\nfrom torch import nn",
        "from typing import TYPE_CHECKING\n\nimport torch\nfrom torch import nn",
    ),
    (
        "models/base.py",
        "            tasks_config: List of task configurations from checkpoint\n"
        '        """\n'
        "        tasks = [hydra.utils.instantiate(task_config) for task_config in tasks_config]",
        "            tasks_config: List of task configurations from checkpoint\n"
        '        """\n'
        "        # Imported lazily: hydra is only needed for checkpoint-based task\n"
        "        # instantiation, never during construct/forward.\n"
        "        import hydra\n"
        "\n"
        "        tasks = [hydra.utils.instantiate(task_config) for task_config in tasks_config]",
    ),
]


def apply_patches():
    for rel, old, new in PATCHES:
        f = VENDOR_ROOT / "fairchem" / "core" / rel
        text = f.read_text()
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"patch for {rel}: expected exactly 1 match, found {count}"
            )
        f.write_text(text.replace(old, new))
        print(f"# patched {rel}")


def _git_sha(source: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Cannot establish FAIR-Chem git provenance for {source}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--fairchem-source",
        type=Path,
        help="Explicit FAIR-Chem git checkout used for reproducible provenance",
    )
    args = parser.parse_args()
    apply = args.apply

    if args.fairchem_source is not None:
        source = args.fairchem_source.resolve()
        package_root = source / "src"
        if not (package_root / "fairchem" / "core").is_dir():
            raise RuntimeError(f"No src/fairchem/core tree under {source}")
        sys.path.insert(0, str(package_root))
        fairchem_sha = _git_sha(source)
    else:
        raise RuntimeError(
            "--fairchem-source is required; vendoring from an arbitrary installed "
            "package cannot record reproducible commit provenance"
        )

    py_files, resources, fairchem_root = closure()
    print(f"# py files:        {len([p for p in py_files if p.suffix == '.py'])}")
    print(f"# resource files:  {len(resources)}")

    if not apply:
        for p in sorted(py_files):
            print("  py", p.relative_to(fairchem_root))
        for p in sorted(resources):
            print(" res", p.relative_to(fairchem_root))
        return

    if VENDOR_ROOT.exists():
        shutil.rmtree(VENDOR_ROOT)
    VENDOR_ROOT.mkdir(parents=True)

    # Root marker + top-level __init__
    (VENDOR_ROOT / "__init__.py").write_text(DUAL_LICENSE_HEADER + MARKER)
    license_file = source / "LICENSE.md"
    if not license_file.is_file():
        raise RuntimeError(f"No FAIR-Chem LICENSE.md under {source}")
    shutil.copy2(license_file, VENDOR_ROOT / "FAIRCHEM_LICENSE.md")
    (VENDOR_ROOT / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "upstream": "https://github.com/FAIR-Chem/fairchem",
                "commit": fairchem_sha,
                "vendored_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "transformations": [
                    "fairchem.core imports rewritten to HydraGNN vendored namespace",
                    "package __init__.py files stripped to avoid unrelated eager imports",
                    "documented lazy-import patches in tools/vendor_uma.py applied",
                    "MIT AND BSD-3-Clause copyright/SPDX header prepended while preserving upstream notices",
                    "upstream FAIR-Chem LICENSE.md copied as FAIRCHEM_LICENSE.md",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    for p in sorted(py_files):
        copy_and_rewrite(p, fairchem_root)
    for p in sorted(resources):
        copy_and_rewrite(p, fairchem_root)

    # Ensure every intermediate dir has an __init__.py
    for d in VENDOR_ROOT.rglob("*"):
        if d.is_dir():
            ensure_init(d)

    # Defer heavy fairchem-core imports that are never reached during
    # import/construct/forward (see PATCHES).
    apply_patches()

    print(f"# vendored into {VENDOR_ROOT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
