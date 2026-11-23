"""Integrity checks for the relocation of published path/hash records."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fip_release_paths import matches_published_hash, rebase_path, stored_fingerprint


class ReleasePaths(unittest.TestCase):
    def test_relocation_is_idempotent_under_named_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "revision_R2" / "published_checkout"
            with patch.dict(os.environ, {"FIP_RELEASE_ROOT": str(root)}):
                expected = root / "revision_R2" / "study" / "preparation.json"
                self.assertEqual(rebase_path(expected), expected)
                self.assertEqual(
                    rebase_path(rebase_path("revision_R2/study/preparation.json")),
                    expected,
                )
                self.assertEqual(
                    stored_fingerprint(
                        {"revision_R2/study/preparation.json": "hash"}, expected
                    ),
                    "hash",
                )

    def test_relative_and_serialized_paths_agree(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"FIP_RELEASE_ROOT": directory}
        ):
            expected = Path(directory).resolve() / "examples/data/full_data_v5.pt"
            self.assertEqual(rebase_path("examples/data/full_data_v5.pt"), expected)
            self.assertEqual(
                rebase_path("/prior/checkout/examples/data/full_data_v5.pt"), expected
            )
            self.assertEqual(
                stored_fingerprint({"examples/data/full_data_v5.pt": "hash"}, expected),
                "hash",
            )

    def test_transformed_source_requires_both_hashes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"FIP_RELEASE_ROOT": directory}
        ):
            root = Path(directory)
            source = root / "example.py"
            source.write_text("published source")
            old = hashlib.sha256(b"original source").hexdigest()
            new = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = {
                "files": {"example.py": {"sha256": new, "original_source_sha256": old}}
            }
            (root / "FIP_CODE_MANIFEST.json").write_text(json.dumps(manifest))
            self.assertTrue(matches_published_hash(source, old))
            self.assertTrue(matches_published_hash(source, new))
            self.assertFalse(matches_published_hash(source, "unrelated"))
            source.write_text("subsequent modification")
            self.assertFalse(matches_published_hash(source, old))


if __name__ == "__main__":
    unittest.main()
