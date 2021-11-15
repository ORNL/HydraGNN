import os
from setuptools import setup, find_packages

# Note: setup() has access to cmd arguments of the setup.py script via sys.argv

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="HydraGNN",
    version="1.0",
    package_dir={"hydragnn": "hydragnn"},
    packages=find_packages(),
    description="Distributed PyTorch implementation of multi-headed graph convolutional neural networks",
    license="BSD-3",
    long_description=read("README.md"),
    url="https://github.com/ORNL/HydraGNN",
    author="Massimiliano Lupo Pasini, Samuel Temple Reeve, Pei Zhang, Jong Youl Choi",
    author_email="lupopasinim@ornl.gov",
)
