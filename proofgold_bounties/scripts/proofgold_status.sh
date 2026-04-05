#!/bin/bash
# Show Proofgold node status

PROOFGOLD_DIR="$HOME/.proofgold"

echo "Proofgold Status"
echo "================"
echo

if [ -f "$PROOFGOLD_DIR/proofgold.log" ]; then
    echo "Recent log entries:"
    tail -20 "$PROOFGOLD_DIR/proofgold.log"
else
    echo "No log file found (node may not have started yet)"
fi

echo
echo "Database directories:"
ls -lh "$PROOFGOLD_DIR/" 2>/dev/null || echo "  (not created yet)"
