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
from setuptools import setup, find_namespace_packages

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def parse_requirements(filename):
    """Parse a requirements file and return a list of dependencies."""
    requirements = []
    filepath = os.path.join(os.path.dirname(__file__), filename)

    if not os.path.exists(filepath):
        return requirements

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and -r references
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)

    return requirements


def get_install_requires():
    """Get install requirements from the modular requirements files."""
    requirements = []

    # Read base requirements
    requirements.extend(parse_requirements("requirements-base.txt"))

    # Read PyTorch requirements
    requirements.extend(parse_requirements("requirements-torch.txt"))

    # Read PyTorch Geometric requirements
    requirements.extend(parse_requirements("requirements-pyg.txt"))

    return requirements


install_requires = get_install_requires()
test_requires = parse_requirements("requirements-dev.txt")
version_namespace = {}
with open(os.path.join(os.path.dirname(__file__), "hydragnn", "_version.py")) as f:
    exec(f.read(), version_namespace)

setup(
    name="HydraGNN",
    version=version_namespace["__version__"],
    python_requires=">=3.11,<3.15",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    package_dir={"hydragnn": "hydragnn"},
    packages=find_namespace_packages(include=["hydragnn", "hydragnn.*"]),
    # Vendored FairChem UMA backbone ships non-Python resources (spherical
    # harmonic / Wigner-D coefficient tensors and model metadata) that must be
    # packaged alongside the sources so the backbone can be constructed.
    include_package_data=True,
    package_data={
        "hydragnn": [
            "utils/model/uma/_vendored/**/*.pt",
            "utils/model/uma/_vendored/**/*.json",
            "utils/model/uma/_vendored/*.md",
            "utils/model/uma/_vendored/**/*.md",
        ]
    },
    install_requires=install_requires,
    extras_require={"test": test_requires},
    description="Distributed PyTorch implementation of multi-headed graph neural networks",
    license="BSD-3",
    license_files=["LICENSE", "LICENSES/PYG-MIT.txt"],
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    url="https://github.com/ORNL/HydraGNN",
    author="Massimiliano Lupo Pasini, Samuel Temple Reeve, Pei Zhang, Jong Youl Choi",
    author_email="lupopasinim@ornl.gov",
)
