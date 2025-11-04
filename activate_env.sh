#!/bin/bash
# Activate HydraGNN virtual environment and set PYTHONPATH
# Usage: source activate_env.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found at $SCRIPT_DIR/.venv"
    return 1
fi

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "✓ PYTHONPATH set to: $SCRIPT_DIR"

# Verify Python
echo "✓ Python: $(which python)"
echo "✓ Python version: $(python --version)"
echo ""
echo "Environment ready! You can now run:"
echo "  - python examples/open_materials_2024/train.py --inputfile omat24_energy.json"
echo "  - pytest tests/test_feature_config.py -v"
echo "  - python -m hydragnn.utils.input_config_parsing.feature_config"
