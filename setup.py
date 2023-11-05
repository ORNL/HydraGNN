import os
from setuptools import setup, find_packages

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


install_requires = [
    "pickle5",
    "matplotlib",
    "tqdm",
    "tensorboard",
    "torch>=1.8",
    "torch-geometric>=1.7.2",
    "torch-scatter",
    "torch-sparse",
]
test_requires = ["black", "pytest", "pytest-mpi"]

setup(
    name="HydraGNN",
    version="3.0-rc1",
    package_dir={"hydragnn": "hydragnn"},
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"test": test_requires},
    description="Distributed PyTorch implementation of multi-headed graph convolutional neural networks",
    license="BSD-3",
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    url="https://github.com/ORNL/HydraGNN",
    author="Massimiliano Lupo Pasini, Samuel Temple Reeve, Pei Zhang, Jong Youl Choi",
    author_email="lupopasinim@ornl.gov",
)
