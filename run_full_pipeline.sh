#!/bin/bash
# SAWN Full Automated Pipeline
# Runs: MoViNet Training -> Evaluation -> Verify outputs
# Usage: source venv/bin/activate && ./run_full_pipeline.sh

set -e  # Exit on error

DATA_DIR="New_SawnDataset"
MODEL_DIR="models/movinet"
OUTPUT_DIR="outputs/evaluation"

echo "============================================"
echo " SAWN Full Pipeline Started: $(date)"
echo "============================================"

# ── Step 1: Training (skip if best model exists) ──
if [ -f "$MODEL_DIR/movinet_best.pt" ]; then
    echo "[SKIP] Training already done. Found: $MODEL_DIR/movinet_best.pt"
else
    echo "[TRAIN] Starting MoViNet training..."
    python scripts/train_movinet.py --data_dir "$DATA_DIR" --epochs 30 --lr 1e-4
    echo "[TRAIN] Done. $(date)"
fi

# ── Step 2: Evaluation on test set ──
echo ""
echo "[EVAL] Running evaluation on test set..."
python scripts/evaluate.py \
    --model "$MODEL_DIR/movinet_best.pt" \
    --test_dir "$DATA_DIR"

if [ $? -ne 0 ]; then
    echo "[ERROR] Evaluation failed!"
    exit 1
fi
echo "[EVAL] Done. $(date)"

# ── Step 3: Verify outputs ──
echo ""
echo "[VERIFY] Checking output files..."

if [ -f "$OUTPUT_DIR/confusion_matrix.png" ]; then
    echo "  ✓ confusion_matrix.png"
else
    echo "  ✗ confusion_matrix.png missing"
fi

if [ -f "$OUTPUT_DIR/evaluation_results.csv" ]; then
    echo "  ✓ evaluation_results.csv"
else
    echo "  ✗ evaluation_results.csv missing"
fi

if [ -f "$OUTPUT_DIR/evaluation_summary.txt" ]; then
    echo "  ✓ evaluation_summary.txt"
    echo ""
    cat "$OUTPUT_DIR/evaluation_summary.txt"
else
    echo "  ✗ evaluation_summary.txt missing"
fi

echo ""
echo "============================================"
echo " SAWN Pipeline Complete: $(date)"
echo "============================================"
