#!/usr/bin/env bash
set -e

PY=""
for candidate in "py -3.12" "python3.12" "python3"; do
    if $candidate --version &>/dev/null 2>&1; then
        VER=$($candidate --version 2>&1)
        if [[ "$VER" == *"3.12"* ]] || [[ "$VER" == *"3.11"* ]]; then
            PY=$candidate
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "!! Python 3.11 or 3.12 not found."
    echo "   Install from https://www.python.org/downloads/release/python-31210/"
    exit 1
fi

echo "==> Using Python: $PY ($($PY --version 2>&1))"
echo "==> Removing any broken venv from previous attempt"
rm -rf venv

echo "==> Creating virtual environment"
$PY -m venv venv

if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
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
