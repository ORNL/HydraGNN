#!/bin/bash
# HydraGNN Package Installation Script
# This script ensures reproducible package installation
# Recommended for both local development and CI environments

set -e  # Exit on any error

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

# Install optional dependencies (optional)
if [ "$1" == "all" ] || [ "$2" == "optional" ]; then
    echo "Installing optional dependencies..."
    pip install --no-build-isolation -v -r requirements-optional.txt
fi

echo "Installation complete!"
echo "Installed package versions:"
pip list | grep -E "(numpy|scipy|torch|scikit-learn|matplotlib|ase)"