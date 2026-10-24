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

import ast
import glob
import json
import os
import subprocess
import sys

import pytest

from tests._script_runner import REPO_ROOT

EXAMPLES = os.path.join(REPO_ROOT, "examples")
OPTIONAL_IMPORTS = {
    "ani1_x/train.py": ("h5py",),
    "ani1_x/train_mlip.py": ("h5py",),
    "mptrj/train.py": ("jarvis",),
    "opf/train_opf_solution_heterogeneous.py": ("h5py",),
    "pglearn/train_pglearn_solution_heterogeneous.py": ("h5py",),
    "pglearn/train_pglearn_solution_homogeneous.py": ("h5py",),
    "qm7x/train.py": ("h5py",),
    "qm7x/train_mlip.py": ("h5py",),
    "transition1x/train.py": ("h5py",),
}


def _cli_scripts():
    scripts = []
    for path in sorted(glob.glob(os.path.join(EXAMPLES, "**", "*.py"), recursive=True)):
        with open(path) as f:
            if "cli.build_parser(" in f.read():
                scripts.append(os.path.relpath(path, EXAMPLES))
    return scripts


SCRIPTS = _cli_scripts()


def _default_inputfiles(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == (
            "build_parser"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            value = kw.get(
                "default_inputfile", node.args[1] if len(node.args) > 1 else None
            )
            if value is not None:
                found.append(ast.literal_eval(value))
    return found


def test_cli_scripts_discovered():
    assert len(SCRIPTS) >= 25


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_cli_script(script, tmp_path):
    for module in OPTIONAL_IMPORTS.get(script, ()):
        pytest.importorskip(module)
    path = os.path.join(EXAMPLES, script)
    workdir = os.path.dirname(path)
    with open(path) as f:
        source = f.read()
    inputfiles = _default_inputfiles(source)
    assert inputfiles, f"{script}: no default_inputfile passed to build_parser"
    for inputfile in inputfiles:
        with open(os.path.join(workdir, inputfile)) as f:
            config = json.load(f)
        assert "NeuralNetwork" in config, inputfile

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_ROOT, env.get("PYTHONPATH")) if p
    )
    # Some examples run gcloud at import time; keep it off the user's config.
    env["CLOUDSDK_CONFIG"] = str(tmp_path / "gcloud")
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
