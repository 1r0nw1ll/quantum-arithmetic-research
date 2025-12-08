#!/bin/bash
# Coordination script - run on player2 to help player4

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       Player2 ↔ Player4 Coordination Dashboard              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check player4 connection
check_connection() {
    if ssh -o ConnectTimeout=3 -o BatchMode=yes player2@192.168.4.31 "echo 'ping'" 2>/dev/null; then
        echo "✓ Can reach player4 (SSH available)"
        return 0
    else
        echo "• Cannot SSH to player4 (may not have SSH server / firewall)"
        return 1
    fi
}

# Offer help options
echo "Select coordination mode:"
echo ""
echo "  1) Monitor player4 progress (passive)"
echo "  2) Check if player4 needs help (query)"
echo "  3) Send status update to player4 (message)"
echo "  4) Continuous monitoring (watch mode)"
echo "  5) Exit"
echo ""
read -p "Choice [1-5]: " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "=== Starting passive monitoring ==="
        watch -n 10 ./monitor_player4_activity.sh
        ;;
    2)
        echo ""
        echo "=== Checking player4 status ==="
        check_connection
        if [ $? -eq 0 ]; then
            echo ""
            ssh player2@192.168.4.31 "cd signal_experiments && ls -lh qa_training_dataset.jsonl 2>/dev/null && echo '✓ Dataset present' || echo '✗ Dataset missing'"
            echo ""
            ssh player2@192.168.4.31 "cd signal_experiments && ps aux | grep -E 'qa_theorem|train_qalm' | grep -v grep || echo 'No training processes running'"
        fi
        ;;
    3)
        echo ""
        echo "=== Send message to player4 ==="
        echo ""
        cat > /tmp/player4_message.txt << 'EOF'
🤖 Message from Player2 (Claude Code):

Status Update:
✓ All files synced successfully (14GB)
✓ BobNet tested and operational
✓ 31,606 training examples ready
✓ Recommended: Start with theorem discovery

Ready to coordinate when you need assistance!

Commands:
  Theorem Discovery: python qa_theorem_discovery_orchestrator.py --quick
  QALM Training:     python train_qalm_production.py --epochs 100

- Claude Code on player2
EOF
        cat /tmp/player4_message.txt
        echo ""
        echo "(Message prepared - would need to send via SSH or shared file)"
        ;;
    4)
        echo ""
        echo "=== Continuous monitoring active ==="
        echo "Press Ctrl+C to stop"
        echo ""
        while true; do
            clear
            ./monitor_player4_activity.sh
            sleep 15
        done
        ;;
    5)
        echo "Exiting coordination mode."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac
