#!/usr/bin/env bash
set -e

if [ ! -d "venv" ]; then
    echo "Run setup.sh first"; exit 1
fi

# Windows vs Unix venv activation
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

if [ -z "$1" ]; then
    echo "Usage: bash run.sh path/to/video.mp4"
    exit 1
fi

python main.py "$@"
