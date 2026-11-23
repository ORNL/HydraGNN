"""Portable roots for the published FIP example and serialized dataset states."""

import os
import hashlib
import json
from pathlib import Path


def release_root():
    """The assembled code/data workspace; defaults to this checkout."""
    return Path(
        os.environ.get("FIP_RELEASE_ROOT", Path(__file__).resolve().parent)
    ).resolve()


def rebase_path(value):
    """Relocate a dataset path stored in a published preprocessing pickle."""
    text = str(value).replace("\\", "/")
    path = Path(value)
    root = release_root()
    if path.is_absolute():
        resolved = path.resolve()
        if resolved.is_relative_to(root):
            return resolved
    for marker in (
        "examples/",
        "hydragnn/",
        "revision_R2/",
        "revision_final/",
        "revision_restart_2026_10_05/",
    ):
        if text.startswith(marker):
            return root / text
        if "/" + marker in text:
            return root / marker / text.split("/" + marker, 1)[1]
    return path if path.is_absolute() else root / path


def stored_fingerprint(mapping, path):
    """Read a path-keyed hash record after relocating the release."""
    path = rebase_path(path).resolve()
    matches = [
        value for key, value in mapping.items() if rebase_path(key).resolve() == path
    ]
    if len(matches) != 1:
        raise KeyError(f"Expected one fingerprint for {path}; found {len(matches)}")
    return matches[0]


def matches_published_hash(path, expected):
    """Verify bytes directly or through the explicit source-to-release manifest.

    Only a manifest entry binding both the original and released bytes permits
    a formatting/path relocation. Scientific binary files still match directly.
    """
    path = rebase_path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == expected:
        return True
    relative = path.resolve().relative_to(release_root()).as_posix()
    code_manifest = release_root() / "FIP_CODE_MANIFEST.json"
    if code_manifest.exists():
        entry = json.loads(code_manifest.read_text())["files"].get(relative, {})
        if (
            entry.get("original_source_sha256") == expected
            and entry.get("sha256") == actual
        ):
            return True
    data_manifest = Path(
        os.environ.get("FIP_DATA_MANIFEST", release_root() / "FILE_MANIFEST.json")
    )
    if data_manifest.exists():
        for entry in json.loads(data_manifest.read_text()):
            if (
                entry["path"] == relative
                and entry.get("original_sha256") == expected
                and entry.get("sha256") == actual
            ):
                return True
    return False
