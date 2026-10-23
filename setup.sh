#!/usr/bin/env bash
set -e

# Auto-detect the working Python command
if command -v python3 &>/dev/null && python3 --version &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null && python --version &>/dev/null; then
    PY=python
elif command -v py &>/dev/null && py --version &>/dev/null; then
    PY=py
else
    echo "!! No working Python found."
    echo "   Install from https://www.python.org/downloads/windows/"
    echo "   During install, TICK 'Add Python to PATH'."
    echo "   Also disable 'python.exe' alias in:"
    echo "   Settings > Apps > Advanced app settings > App execution aliases"
    exit 1
fi

echo "==> Using Python: $PY ($($PY --version))"

echo "==> Creating virtual environment"
$PY -m venv venv

# Activate venv (works on Git Bash / MSYS / Linux / macOS)
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate      # Windows
else
    source venv/bin/activate          # Unix
fi

echo "==> Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing requirements"
pip install -r requirements.txt

echo "==> Checking FFmpeg"
if ! command -v ffmpeg &> /dev/null; then
    echo "!! FFmpeg not found. Install it first."
    exit 1
fi

echo "==> Done. Activate with: source venv/Scripts/activate"
