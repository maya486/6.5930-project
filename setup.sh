#!/bin/bash
# One-time setup script for the project environment

set -e  # Exit on error

echo "=== 6.5930 Project Setup ==="
echo ""

# Check if we're in the right directory
if [ ! -f "setup.sh" ]; then
    echo "Error: Please run this script from the 6.5930-project directory"
    exit 1
fi

# Check if accelforge repo exists
if [ ! -d "../accelforge" ]; then
    echo "Error: AccelForge repository not found!"
    echo "Please clone it first:"
    echo "  cd .."
    echo "  git clone https://github.com/maya486/accelforge.git"
    exit 1
fi

# Check if virtual environment already exists
if [ -d "../accelforge_env" ]; then
    echo "Virtual environment already exists at ../accelforge_env"
    read -p "Do you want to recreate it? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf ../accelforge_env
    else
        echo "Using existing environment."
        echo "Setup complete! Run ./start_jupyter.sh to start Jupyter."
        exit 0
    fi
fi

# Check Python version
echo "Checking Python version..."
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
        echo "Error: Python 3.8 or higher required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "Using: $PYTHON_CMD"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv ../accelforge_env
echo "Virtual environment created at ../accelforge_env"
echo ""

# Activate virtual environment
source ../accelforge_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install numba from binary (to avoid compilation issues)
echo "Installing numba from binary..."
pip install --only-binary :all: numba
echo ""

# Install AccelForge in editable mode
echo "Installing AccelForge in editable mode..."
pip install -e ../accelforge
echo ""

# Install Jupyter and other requirements
echo "Installing Jupyter and other dependencies..."
pip install -r requirements.txt
echo ""

echo "=== Setup Complete! ==="
echo ""
echo "To start working:"
echo "  ./start_jupyter.sh"
echo ""
echo "Or manually:"
echo "  source ../accelforge_env/bin/activate"
echo "  jupyter notebook"
