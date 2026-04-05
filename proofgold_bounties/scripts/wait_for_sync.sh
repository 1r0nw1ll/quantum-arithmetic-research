#!/bin/bash
# Wait for Proofgold daemon to complete blockchain sync

PROOFGOLDCLI="/home/player2/signal_experiments/proofgold_bounties/data/proofgold-lava/client/bin/proofgoldcli"
MAX_WAIT=7200  # 2 hours max
INTERVAL=60    # Check every 60 seconds

elapsed=0

echo "Waiting for Proofgold blockchain sync to complete..."
echo "Checking every $INTERVAL seconds (max wait: $((MAX_WAIT/60)) minutes)"
echo ""

while [ $elapsed -lt $MAX_WAIT ]; do
    # Try getinfo and capture output
    output=$($PROOFGOLDCLI getinfo 2>&1)

    # Check if it contains the "out of sync" error
    if echo "$output" | grep -q "cannot find best validated header"; then
        echo "[$(date +%H:%M:%S)] Still syncing... (elapsed: $((elapsed/60))m)"
    else
        echo "[$(date +%H:%M:%S)] Sync appears complete!"
        echo ""
        echo "getinfo output:"
        echo "$output"
        echo ""
        echo "Proofgold is ready for operations."
        exit 0
    fi

    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

echo ""
echo "Timeout after $((MAX_WAIT/60)) minutes. Sync may still be in progress."
echo "Check manually with: $PROOFGOLDCLI getinfo"
exit 1
