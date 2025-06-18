#!/bin/bash

# install.sh - Script to set up a Python virtual environment and install the package

# Exit on error
set -e

# Help text
usage() {
    echo "Usage: $0 [--gui]"
    echo "       --gui    Install with GUI dependencies"
    exit 1
}

# Parse arguments
EXTRA=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gui) EXTRA="[gui]" ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

# Find package name from pyproject.toml
if [ ! -f pyproject.toml ]; then
    echo "❌ pyproject.toml not found in this directory."
    exit 1
fi

PACKAGE=$(grep -E '^name\s*=' pyproject.toml | head -n1 | cut -d '"' -f2)

if [ -z "$PACKAGE" ]; then
    echo "❌ Could not read package name from pyproject.toml"
    exit 1
fi

VENV_DIR=".venv_$PACKAGE"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "🔌 Activating environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install the package
echo "📦 Installing $PACKAGE$EXTRA ..."
pip install --upgrade pip
pip install ".$EXTRA"

echo ""
echo "✅ Installation complete."
echo "💡 To activate the environment manually later:"
echo "   source $VENV_DIR/bin/activate"
echo "💡 To deactivate the environment, just run:"
echo "    deactivate"
