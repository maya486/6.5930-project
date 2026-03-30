#!/bin/bash
# Script to start Jupyter notebook with the correct environment

# Activate the virtual environment and start Jupyter
cd "$(dirname "$0")"
source ../accelforge_env/bin/activate
jupyter notebook
