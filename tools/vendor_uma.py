#!/usr/bin/env python3
"""Vendor fairchem.core UMA closure into HydraGNN.

Copies the enumerated closure verbatim under
  hydragnn/utils/model/uma/_vendored/fairchem/core/**
then rewrites all `from fairchem.core.X` / `import fairchem.core.X`
occurrences to point at the vendored location.

Also copies non-.py resources (Jd.pt, wigner_d_coefficients.pt,
pretrained_models.json) that live next to visited modules.

Adds a marker file at each vendored subtree root so we know its
provenance (MIT license, upstream commit).
"""
from __future__ import annotations
import ast
import importlib.util
import re
import shutil
import sys
from collections import deque
from pathlib import Path

REPO = Path("/Users/7ml/Documents/Codes/HydraGNN").resolve()
VENDOR_ROOT = REPO / "hydragnn" / "utils" / "model" / "uma" / "_vendored"

SEEDS = [
    "fairchem.core.models.uma.escn_md",
    "fairchem.core.models.uma.escn_md_block",
    "fairchem.core.models.uma.escn_moe",
    "fairchem.core.models.uma.outputs",
]

RESOURCE_EXTS = {".pt", ".yaml", ".yml", ".json"}


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

MARKER = """# NOTE: This tree was vendored verbatim from fairchem-core (Meta MIT-licensed)
# for use inside HydraGNN. Imports were rewritten by
# tools/vendor_uma.py to point at hydragnn.utils.model.uma._vendored.
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
                "# Vendored from fairchem-core (MIT). Original __init__ contents\n"
                "# stripped to avoid eager imports of modules outside the\n"
                "# UMA forward-pass closure.\n"
            )
            return dst
        text = src.read_text()
        text = IMPORT_RE.sub("hydragnn.utils.model.uma._vendored.fairchem.core", text)
        dst.write_text(text)
    else:
        shutil.copy2(src, dst)
    return dst


def ensure_init(dirpath: Path):
    init = dirpath / "__init__.py"
    if not init.exists():
        init.write_text("")


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


def main():
    apply = "--apply" in sys.argv

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
    (VENDOR_ROOT / "__init__.py").write_text(MARKER)

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
