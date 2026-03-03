#!/bin/bash
set -e

echo "=== Turkish Reasoning Logit Lens - Full Pipeline ==="
echo ""

# Check environment
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set. Export your HuggingFace token first."
    echo "  export HF_TOKEN=your_token_here"
    exit 1
fi

echo "[1/5] Running unit tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "[2/5] Running behavioral baseline (Notebook 01)..."
jupyter nbconvert --to notebook --execute notebooks/01_behavioral_baseline.ipynb \
    --ExecutePreprocessor.timeout=3600

echo ""
echo "[3/5] Running logit lens analysis (Notebook 02)..."
jupyter nbconvert --to notebook --execute notebooks/02_logit_lens_analysis.ipynb \
    --ExecutePreprocessor.timeout=7200

echo ""
echo "[4/5] Generating visualizations (Notebook 03)..."
jupyter nbconvert --to notebook --execute notebooks/03_visualizations.ipynb \
    --ExecutePreprocessor.timeout=600

echo ""
echo "[5/5] Running bonus analyses (Notebook 04)..."
jupyter nbconvert --to notebook --execute notebooks/04_bonus_analyses.ipynb \
    --ExecutePreprocessor.timeout=3600

echo ""
echo "=== Done! Figures saved to results/figures/ ==="
