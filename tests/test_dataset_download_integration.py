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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import threading


def _archive_payload():
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        content = b"small representative dataset"
        member = tarfile.TarInfo("dataset/sample.txt")
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    return payload.getvalue()


def test_downloader_cli_resumes_redirect_and_extracts(tmp_path):
    payload = _archive_payload()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/archive.tar.gz")
                self.end_headers()
                return
            if self.path != "/archive.tar.gz":
                self.send_error(404)
                return

            range_header = self.headers.get("Range")
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload) - start))
            self.send_header(
                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            )
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        archive = tmp_path / "dataset.tar.gz"
        archive.with_name(archive.name + ".part").write_bytes(payload[:17])
        output = tmp_path / "output"
        downloader = (
            Path(__file__).parents[1]
            / "hydragnn"
            / "utils"
            / "datasets"
            / "download.py"
        )
        subprocess.run(
            [
                sys.executable,
                str(downloader),
                f"http://127.0.0.1:{server.server_port}/redirect",
                str(archive),
                "--sha256",
                hashlib.sha256(payload).hexdigest(),
                "--extract-to",
                str(output),
                "--remove-archive",
            ],
            check=True,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert (output / "dataset/sample.txt").read_bytes() == (
        b"small representative dataset"
    )
    assert not archive.exists()


def test_materials_download_entry_points_use_shared_transport():
    repository = Path(__file__).parents[1]
    shell_consumers = {
        "examples/mptrj/download_data_andes.sh": "MPtrj_2022.9_full.json",
        "examples/transition1x/download_dataset.sh": "transition1x-release.h5",
        "examples/ani1_x/download_andes.sh": "ani1x-release.h5",
        "examples/open_direct_air_capture_2023/download_dataset.sh": (
            "extxyz_train.tar.gz"
        ),
        "examples/open_catalyst_2022/download_dataset.sh": ("DS_jgaid7espcoc_0.tar.xz"),
        "examples/open_molecules_2025/download_dataset.sh": "train.tar.gz",
        "examples/open_polymers_2026/download_dataset.sh": (
            "op26_train_val_260108.tar.gz"
        ),
        "examples/qm7x/download_dataset.sh": "3905361.zip",
    }
    for relative_path, expected_filename in shell_consumers.items():
        source = (repository / relative_path).read_text()
        assert "hydragnn/utils/datasets/download.py" in source
        assert expected_filename in source

    for relative_path in (
        "examples/open_catalyst_2020/download_dataset.py",
        "examples/open_materials_2024/download_dataset.py",
    ):
        source = (repository / relative_path).read_text()
        assert "hydragnn.utils.datasets.download import" in source
        assert "download_file(" in source
