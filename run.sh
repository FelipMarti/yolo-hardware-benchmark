#!/bin/bash
set -e

MODEL=yolo11n.pt
PYTHON=.venv/bin/python

echo "Using Python:"
$PYTHON -c "import sys; print(sys.executable)"

echo "Exporting..."
$PYTHON scripts/export.py --model $MODEL

echo "Benchmarking..."
$PYTHON scripts/benchmark.py --model yolo11n
