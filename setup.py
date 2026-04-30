import os
from setuptools import setup, find_packages

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

setup(
    name="HydraGNN",
    version="4.0rc1",
    package_dir={"hydragnn": "hydragnn"},
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"test": test_requires},
    description="Distributed PyTorch implementation of multi-headed graph neural networks",
    license="BSD-3",
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    url="https://github.com/ORNL/HydraGNN",
    author="Massimiliano Lupo Pasini, Samuel Temple Reeve, Pei Zhang, Jong Youl Choi",
    author_email="lupopasinim@ornl.gov",
)
