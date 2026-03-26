#!/bin/bash
# Reproduce all experiments from scratch.
# Usage: bash run_all.sh

set -e  # stop on first error

echo "=== Step 1: Generate synthetic data ==="
python experiments/generate_data.py

echo ""
echo "=== Step 2: Run ablation study ==="
python experiments/run_ablation.py

echo ""
echo "=== Step 3: Train full models ==="
python experiments/train_full.py

echo ""
echo "=== Step 4: Analyze results ==="
python analysis/analyze.py

echo ""
echo "=== All steps complete ==="