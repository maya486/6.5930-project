#!/bin/bash
# Script to start Jupyter notebook with the correct environment

# Activate the virtual environment and start Jupyter
cd "$(dirname "$0")"
source ../accelforge_env/bin/activate

echo "=== Starting Jupyter Notebook ==="
echo ""
echo "IMPORTANT: In your notebook, select the kernel:"
echo "  Kernel -> Change Kernel -> Python (accelforge_env)"
echo ""

jupyter notebook
