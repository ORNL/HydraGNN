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

import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from hydragnn.utils.datasets.download import download_file, safe_extract_tar


class _Response(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_download_file_resumes_partial_content(monkeypatch, tmp_path):
    payload = b"complete dataset contents"
    destination = tmp_path / "dataset.bin"
    partial = tmp_path / "dataset.bin.part"
    partial.write_bytes(payload[:9])

    def fake_urlopen(request):
        assert request.headers["Range"] == "bytes=9-"
        return _Response(payload[9:], status=206)

    monkeypatch.setattr("hydragnn.utils.datasets.download.urlopen", fake_urlopen)
    result = download_file(
        "https://example.invalid/dataset.bin",
        destination,
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert result.read_bytes() == payload
    assert not partial.exists()


def test_download_file_restarts_if_server_ignores_range(monkeypatch, tmp_path):
    destination = tmp_path / "dataset.bin"
    (tmp_path / "dataset.bin.part").write_bytes(b"partial")
    payload = b"replacement"
    monkeypatch.setattr(
        "hydragnn.utils.datasets.download.urlopen",
        lambda request: _Response(payload, status=200),
    )

    download_file("https://example.invalid/dataset.bin", destination)
    assert destination.read_bytes() == payload


def test_download_file_reuses_verified_destination(monkeypatch, tmp_path):
    destination = tmp_path / "dataset.bin"
    destination.write_bytes(b"already complete")
    monkeypatch.setattr(
        "hydragnn.utils.datasets.download.urlopen",
        lambda request: pytest.fail("completed file should not be downloaded"),
    )

    assert (
        download_file(
            "https://example.invalid/dataset.bin",
            destination,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )
        == destination
    )


def test_download_file_rejects_existing_directory(monkeypatch, tmp_path):
    destination = tmp_path / "dataset.bin"
    destination.mkdir()
    monkeypatch.setattr(
        "hydragnn.utils.datasets.download.urlopen",
        lambda request: pytest.fail("invalid destination should not be downloaded"),
    )

    with pytest.raises(ValueError, match="exists and is not a file"):
        download_file("https://example.invalid/dataset.bin", destination)


def test_download_file_removes_partial_after_checksum_mismatch(monkeypatch, tmp_path):
    destination = tmp_path / "dataset.bin"
    partial = tmp_path / "dataset.bin.part"
    monkeypatch.setattr(
        "hydragnn.utils.datasets.download.urlopen",
        lambda request: _Response(b"corrupted", status=200),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_file(
            "https://example.invalid/dataset.bin",
            destination,
            sha256=hashlib.sha256(b"expected").hexdigest(),
        )

    assert not partial.exists()
    assert not destination.exists()


def _write_tar(path: Path, member_name: str, data: bytes = b"data"):
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def test_safe_extract_tar_streams_regular_files(monkeypatch, tmp_path):
    archive = tmp_path / "dataset.tar.gz"
    _write_tar(archive, "split/sample.txt")
    monkeypatch.setattr(
        tarfile.TarFile,
        "getmembers",
        lambda self: pytest.fail("streaming extraction must not load all members"),
    )

    destination = safe_extract_tar(archive, tmp_path / "output")
    assert (destination / "split/sample.txt").read_bytes() == b"data"


def test_safe_extract_tar_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    _write_tar(archive, "../outside.txt")

    with pytest.raises(ValueError, match="unsafe archive path"):
        safe_extract_tar(archive, tmp_path / "output")


@pytest.mark.parametrize(
    ("member_type", "message"),
    [
        (tarfile.SYMTYPE, "archive links are not allowed"),
        (tarfile.LNKTYPE, "archive links are not allowed"),
        (tarfile.CHRTYPE, "unsupported archive entry"),
    ],
)
def test_safe_extract_tar_rejects_links_and_special_entries(
    tmp_path, member_type, message
):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("unsafe-entry")
        member.type = member_type
        if member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
            member.linkname = "target"
        tar.addfile(member)

    with pytest.raises(ValueError, match=message):
        safe_extract_tar(archive, tmp_path / "output")
