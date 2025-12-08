#!/bin/bash
# Monitor player4 activity and theorem discovery progress

echo "=== Player4 Activity Monitor ==="
echo "Time: $(date)"
echo ""

# Check network connection
echo "[1] Network Status:"
CONN=$(netstat -tn | grep "192.168.4.31.*ESTABLISHED")
if [ -n "$CONN" ]; then
    echo "  ✓ Player4 connected"
    echo "  $CONN"
else
    echo "  • No active connection (player4 may be working locally)"
fi

echo ""
echo "[2] Recent SSH Activity:"
journalctl -u ssh --since "10 minutes ago" 2>/dev/null | grep "192.168.4.31" | tail -3 || echo "  No recent SSH logs"

echo ""
echo "[3] Check for theorem discovery outputs:"
echo "  Looking for qa_discovery_workspace/ activity..."

# Check if discovery workspace exists and has recent activity
if [ -d "qa_discovery_workspace" ]; then
    echo "  ✓ Discovery workspace exists"
    RECENT=$(find qa_discovery_workspace -mmin -10 -type f 2>/dev/null | wc -l)
    if [ $RECENT -gt 0 ]; then
        echo "  ✓ $RECENT files modified in last 10 minutes"
        echo "  Recent files:"
        find qa_discovery_workspace -mmin -10 -type f 2>/dev/null | head -5
    else
        echo "  • No recent activity"
    fi
else
    echo "  • Workspace not yet created (theorem discovery not started)"
fi

echo ""
echo "[4] Log Files:"
if [ -f "theorem_discovery.log" ]; then
    echo "  ✓ theorem_discovery.log exists"
    LINES=$(wc -l < theorem_discovery.log)
    echo "    Lines: $LINES"
    echo "    Last 5 lines:"
    tail -5 theorem_discovery.log
else
    echo "  • No theorem_discovery.log yet"
fi

if [ -f "qalm_training.log" ]; then
    echo "  ✓ qalm_training.log exists"
    LINES=$(wc -l < qalm_training.log)
    echo "    Lines: $LINES"
else
    echo "  • No qalm_training.log yet"
fi

echo ""
echo "=========================================="
echo "To watch continuously: watch -n 10 $0"
echo "To check logs: tail -f theorem_discovery.log"
echo "=========================================="
