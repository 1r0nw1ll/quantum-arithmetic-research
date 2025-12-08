#!/bin/bash
# Monitor Player4 sync status
# Run: watch -n 5 ./monitor_player4_sync.sh

echo "=== Player4 Sync Monitor ==="
echo "Time: $(date)"
echo ""

# Check connection
CONN=$(netstat -tn | grep "192.168.4.31.*ESTABLISHED" | head -1)
if [ -n "$CONN" ]; then
    echo "✓ Player4 CONNECTED"
    echo "$CONN"
    BUFFER=$(echo "$CONN" | awk '{print $3}')
    echo "  Send buffer: $((BUFFER/1024))KB"
else
    echo "✗ Player4 not connected (sync may be complete)"
fi

echo ""
echo "To check if sync completed, run on player4:"
echo "  cd signal_experiments && ls -lh qa_training_dataset.jsonl"
