#!/usr/bin/env bash
# PodcastCleaner — One-Step Setup
#
# Usage: bash setup.sh
#
# Creates a virtual environment, installs all dependencies,
# copies example config, and runs a dependency check.

set -euo pipefail

echo "=== PodcastCleaner Setup ==="

# Check Python version
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.10+ required but not found."
    echo "Install Python 3.10+ and try again."
    exit 1
fi

echo "Using $PYTHON ($($PYTHON --version))"

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "Warning: ffmpeg not found. Install it before running the pipeline."
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Install
echo "Installing PodcastCleaner and all dependencies..."
pip install --upgrade pip
pip install -e '.[all]'

# Config
if [ ! -f "config.yaml" ]; then
    echo "Creating config.yaml from example..."
    cp config.example.yaml config.yaml
else
    echo "config.yaml already exists — skipping."
fi

# Verify
echo ""
echo "=== Dependency Check ==="
podcast-cleaner check

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit config.yaml to your preferences"
echo "  2. source .venv/bin/activate"
echo "  3. podcast-cleaner run --url \"https://youtube.com/watch?v=...\""
echo "  4. podcast-cleaner run --help for all options"
