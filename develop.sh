#!/bin/bash

# Exit on error
set -e

# Help text
usage() {
    echo "Usage: $0 [--gui]"
    echo "       --gui    Include GUI dependencies"
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

# Check for pyproject.toml
if [ ! -f pyproject.toml ]; then
    echo "❌ pyproject.toml not found."
    exit 1
fi

# Get package name
PACKAGE=$(grep -E '^name\s*=' pyproject.toml | head -n1 | cut -d '"' -f2)

if [ -z "$PACKAGE" ]; then
    echo "❌ Could not extract package name from pyproject.toml"
    exit 1
fi

VENV_DIR=".venv_$PACKAGE"

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "🌀 Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the venv
echo "⚙️  Activating environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install current package in editable mode
echo "📦 Installing $PACKAGE in editable mode$EXTRA..."
pip install --upgrade pip
pip install -e ".$EXTRA"

# Install developer dependencies
if [ -f requirements-dev.txt ]; then
    echo "🔍 Checking local paths in requirements-dev.txt..."

    # Check all -e paths
    while read -r line; do
        # Remove comments and whitespace
        CLEAN=$(echo "$line" | sed 's/#.*//' | xargs)
        
        if [[ $CLEAN == -e\ * ]]; then
            PATH_REF=$(echo "$CLEAN" | cut -d ' ' -f2)
            if [ ! -d "$PATH_REF" ]; then
                echo "❌ Local path does not exist: $PATH_REF"
                exit 1
            fi
        fi
    done < requirements-dev.txt

    echo "🧩 Installing dev requirements..."
    pip install -r requirements-dev.txt
else
    echo "ℹ️ requirements-dev.txt not found — skipping dev dependencies"
fi

echo ""
echo "✅ Development environment is ready."
echo "💡 To activate it manually:"
echo "   source $VENV_DIR/bin/activate"
echo "💡 To deactivate the environment, just run:"
echo "    deactivate"

