#!/bin/bash
#
# Player4 Sync Script
# Run this on player4 (Gemini CLI) to sync the signal_experiments project from player2
#

PLAYER2_IP="192.168.4.60"
PLAYER2_USER="player2"
SOURCE_PATH="/home/player2/signal_experiments/"
DEST_PATH="./signal_experiments/"

echo "=========================================="
echo "Player4 → Player2 Sync Script"
echo "=========================================="
echo "Source: ${PLAYER2_USER}@${PLAYER2_IP}:${SOURCE_PATH}"
echo "Destination: ${DEST_PATH}"
echo "=========================================="
echo ""

# Check network connectivity
echo "[1/5] Checking network connectivity to player2..."
if ping -c 1 -W 2 ${PLAYER2_IP} > /dev/null 2>&1; then
    echo "  ✓ Player2 (${PLAYER2_IP}) is reachable"
else
    echo "  ✗ Cannot reach player2 at ${PLAYER2_IP}"
    echo "  Please check network connection"
    exit 1
fi

# Check SSH connectivity
echo ""
echo "[2/5] Testing SSH connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes ${PLAYER2_USER}@${PLAYER2_IP} "echo '  ✓ SSH connection successful'" 2>/dev/null; then
    :
else
    echo "  ℹ SSH key authentication failed, will prompt for password"
fi

# Start rsync
echo ""
echo "[3/5] Starting rsync synchronization..."
echo "  (This may take a few minutes for ~15MB of data)"
echo ""

rsync -avz --progress ${PLAYER2_USER}@${PLAYER2_IP}:${SOURCE_PATH} ${DEST_PATH}

if [ $? -eq 0 ]; then
    echo ""
    echo "[4/5] ✓ Sync completed successfully!"
else
    echo ""
    echo "[4/5] ✗ Sync failed"
    exit 1
fi

# Verify sync
echo ""
echo "[5/5] Verifying synchronized files..."
cd ${DEST_PATH}

if [ -f "qa_training_dataset.jsonl" ]; then
    echo "  ✓ Found qa_training_dataset.jsonl"
    EXAMPLES=$(wc -l < qa_training_dataset.jsonl)
    echo "    → ${EXAMPLES} training examples"
fi

if [ -f "train_qalm_production.py" ]; then
    echo "  ✓ Found train_qalm_production.py"
fi

if [ -d "qa_lab" ]; then
    echo "  ✓ Found qa_lab/ directory"
fi

echo ""
echo "=========================================="
echo "Sync Complete! Next steps:"
echo "=========================================="
echo "1. Install dependencies:"
echo "   pip install torch numpy pandas matplotlib scikit-learn tqdm"
echo ""
echo "2. Start QALM training:"
echo "   python train_qalm_production.py \\"
echo "       --dataset qa_training_dataset.jsonl \\"
echo "       --epochs 100 \\"
echo "       --batch-size 32 \\"
echo "       --hidden-size 512 \\"
echo "       --num-layers 8 \\"
echo "       --num-heads 8 \\"
echo "       --checkpoint-dir checkpoints/qalm_v1_medium"
echo ""
echo "3. Monitor training:"
echo "   tail -f qalm_training.log"
echo ""
echo "=========================================="
