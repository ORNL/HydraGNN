#!/bin/bash
# HydraGNN Package Installation Script
# This script ensures reproducible package installation
# Recommended for both local development and CI environments

set -e  # Exit on any error

INSTALL_DEV=0
INSTALL_OPTIONAL=0
for group in "$@"; do
    case "${group}" in
        all)
            INSTALL_DEV=1
            INSTALL_OPTIONAL=1
            ;;
        dev)
            INSTALL_DEV=1
            ;;
        optional)
            INSTALL_OPTIONAL=1
            ;;
        *)
            echo "Unknown dependency group: ${group}" >&2
            echo "Usage: $0 [all|dev|optional] [...]" >&2
            exit 2
            ;;
    esac
done

PYTHON_MINOR_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "${PYTHON_MINOR_VERSION}" in
    3.11|3.12|3.13|3.14)
        ;;
    *)
        echo "Unsupported Python ${PYTHON_MINOR_VERSION}. HydraGNN supports Python 3.11 through 3.14." >&2
        exit 1
        ;;
esac

echo "Installing HydraGNN dependencies with consistent settings..."

EXTRAS="torch"
[ "${INSTALL_DEV}" -eq 1 ] && EXTRAS="${EXTRAS},dev"
[ "${INSTALL_OPTIONAL}" -eq 1 ] && EXTRAS="${EXTRAS},optional"

python -m pip install --upgrade "setuptools>=77" wheel

echo "Installing HydraGNN with extras [${EXTRAS}]..."
python -m pip install --no-build-isolation -v -e ".[${EXTRAS}]"

echo "Installing PyTorch Geometric extensions..."
python -m pip install --no-build-isolation -v -e ".[pyg]"

# Build isolation stays enabled so omegaconf's sdist-only antlr4 dependency can build.
echo "Installing model-specific backbone dependencies..."
python -m pip install -v -e ".[uma]"

echo "Installation complete!"
echo "HydraGNN source version: $(python -c 'import hydragnn._version as v; print(v.__version__)')"
echo "Installed package versions:"
python -m pip list | grep -E "(numpy|scipy|torch|scikit-learn|matplotlib|ase)"
