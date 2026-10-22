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

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def example_script(example, script=None):
    return os.path.join(REPO_ROOT, "examples", example, (script or example) + ".py")


def run_script(file_path, *args, env=None, cwd=None):
    """Run a script with the current interpreter and fail with its output on error."""
    env = dict(os.environ if env is None else env)
    # Make the in-repo hydragnn importable without installing it.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (REPO_ROOT, env.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, file_path, *args],
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, (
        f"{os.path.relpath(file_path, REPO_ROOT)} {' '.join(args)} exited with "
        f"code {result.returncode}. Output:\n{result.stdout[-20000:]}"
    )
