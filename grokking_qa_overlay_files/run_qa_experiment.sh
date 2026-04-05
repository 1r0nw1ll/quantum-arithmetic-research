#!/bin/bash
# Quick-start script for QA overlay on grokking experiments

set -e  # Exit on error

echo "========================================"
echo "QA Overlay: Grokking Experiment Runner"
echo "========================================"
echo ""

# Default parameters (override with environment variables)
DATASET=${DATASET:-"modular_addition"}
LOSS_FUNC=${LOSS_FUNC:-"cross_entropy"}
SEED=${SEED:-0}
NUM_EPOCHS=${NUM_EPOCHS:-50000}
LOG_FREQ=${LOG_FREQ:-100}
DEVICE=${DEVICE:-"cuda:0"}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Loss Function: $LOSS_FUNC"
echo "  Seed: $SEED"
echo "  Num Epochs: $NUM_EPOCHS"
echo "  Log Frequency: $LOG_FREQ"
echo "  Device: $DEVICE"
echo ""

# Check if required files exist
if [ ! -f "qa_logger.py" ]; then
    echo "ERROR: qa_logger.py not found!"
    echo "Please copy the QA overlay files to this directory."
    exit 1
fi

if [ ! -f "grokking_experiments_qa.py" ]; then
    echo "ERROR: grokking_experiments_qa.py not found!"
    echo "Please copy the QA overlay files to this directory."
    exit 1
fi

# Run experiment with QA instrumentation
echo "Running experiment with QA instrumentation..."
echo ""

python grokking_experiments_qa.py \
    --dataset "$DATASET" \
    --loss_function "$LOSS_FUNC" \
    --seed "$SEED" \
    --num_epochs "$NUM_EPOCHS" \
    --log_frequency "$LOG_FREQ" \
    --device "$DEVICE" \
    --full_batch

echo ""
echo "========================================"
echo "Experiment complete!"
echo "========================================"
echo ""

# Check for output files
RUN_ID="${DATASET}_${LOSS_FUNC}_seed${SEED}"
QA_LOG="qa_logs/${RUN_ID}.jsonl"

if [ -f "$QA_LOG" ]; then
    echo "✓ QA log created: $QA_LOG"
    NUM_RECORDS=$(wc -l < "$QA_LOG")
    echo "  Records: $NUM_RECORDS"
    echo ""

    echo "Sample record (first):"
    head -1 "$QA_LOG" | python -m json.tool | head -30
    echo "  ..."
    echo ""

    # Run analysis if requested
    read -p "Run QA analysis and generate plots? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running analysis..."
        # Update the run ID in the analysis script
        sed -i "s/RUN_ID = .*/RUN_ID = \"$RUN_ID\"/" qa_analysis_notebook.py
        python qa_analysis_notebook.py
        echo ""
        echo "✓ Analysis complete! Check generated PNG files."
    fi
else
    echo "✗ QA log not found: $QA_LOG"
    echo "  Check for errors above."
fi

echo ""
echo "To run analysis manually:"
echo "  1. Edit qa_analysis_notebook.py: set RUN_ID = \"$RUN_ID\""
echo "  2. Run: python qa_analysis_notebook.py"
echo ""
echo "To compare with StableMax:"
echo "  LOSS_FUNC=stablemax ./run_qa_experiment.sh"
echo ""
