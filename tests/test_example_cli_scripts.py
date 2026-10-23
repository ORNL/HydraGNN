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
"""Every example built on ``hydragnn.cli`` imports, parses ``--help``, and ships its config."""

import glob
import json
import os
import re
import subprocess
import sys

import pytest

from tests._script_runner import REPO_ROOT

EXAMPLES = os.path.join(REPO_ROOT, "examples")
OPTIONAL_IMPORTS = {
    "mptrj/train.py": "jarvis",
    "qcml/train.py": "tensorflow",
}


def _cli_scripts():
    scripts = []
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "**", "*.py"), recursive=True)):
        with open(path) as f:
            if "cli.build_parser(" in f.read():
                scripts.append(os.path.relpath(path, EXAMPLES))
    return scripts


SCRIPTS = _cli_scripts()


def test_cli_scripts_discovered():
    assert len(SCRIPTS) >= 25


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_cli_script(script):
    if script in OPTIONAL_IMPORTS:
        pytest.importorskip(OPTIONAL_IMPORTS[script])
    path = os.path.join(EXAMPLES, script)
    workdir = os.path.dirname(path)
    with open(path) as f:
        source = f.read()
    for inputfile in re.findall(r'default_inputfile="([^"]+)"', source):
        with open(os.path.join(workdir, inputfile)) as f:
            config = json.load(f)
        assert "NeuralNetwork" in config, inputfile

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_ROOT, env.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, os.path.basename(path), "--help"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "--inputfile" in result.stdout
