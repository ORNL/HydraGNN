#!/bin/bash
# HydraGNN Package Installation Script
# This script ensures reproducible package installation
# Recommended for both local development and CI environments

set -e  # Exit on any error

PYTHON_MINOR_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "${PYTHON_MINOR_VERSION}" in
    3.11|3.12|3.14)
        ;;
    *)
        echo "Unsupported Python ${PYTHON_MINOR_VERSION}. HydraGNN supports Python 3.11, 3.12, and 3.14." >&2
        exit 1
        ;;
esac

echo "Installing HydraGNN dependencies with consistent settings..."

# Install base dependencies
echo "Installing base dependencies..."
pip install --no-build-isolation -v -r requirements-base.txt 

# Install PyTorch dependencies
echo "Installing PyTorch dependencies..."
pip install --no-build-isolation -v -r requirements-torch.txt 

# Install PyTorch Geometric dependencies
echo "Installing PyTorch Geometric dependencies..."
pip install --no-build-isolation -v -r requirements-pyg.txt 

# Install development dependencies (optional)
if [ "$1" == "dev" ]; then
    echo "Installing development dependencies..."
    pip install --no-build-isolation -v -r requirements-dev.txt
fi

# Install model-specific backbone dependencies (e.g. FAIRChem UMA).
# NOTE: build isolation is intentionally ENABLED here (no --no-build-isolation)
# so omegaconf's sdist-only antlr4-python3-runtime dependency can build.
echo "Installing model-specific backbone dependencies..."
pip install -v -r requirements-specific-models.txt

# Install optional dependencies (optional)
if [ "$1" == "all" ] || [ "$2" == "optional" ]; then
    echo "Installing optional dependencies..."
    pip install --no-build-isolation -v -r requirements-optional.txt
fi

echo "Installation complete!"
echo "Installed package versions:"
pip list | grep -E "(numpy|scipy|torch|scikit-learn|matplotlib|ase)"
