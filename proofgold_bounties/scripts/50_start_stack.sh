#!/bin/bash
# 50_start_stack.sh - Start Litecoin daemon and Proofgold Lava client
# Safe to run multiple times (idempotent)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROOFGOLD_DIR="$HOME/.proofgold"
LAVA_BINARY="$(pwd)/scripts/proofgoldlava"
LOG_DIR="$(pwd)/data/logs"

echo "=========================================="
echo "Proofgold Lava: Start Stack"
echo "=========================================="
echo

# Create log directory
mkdir -p "$LOG_DIR"

# Check if litecoind service is already running
echo -n "Checking litecoind status... "
if systemctl is-active --quiet litecoind; then
    echo -e "${YELLOW}SKIP${NC} (already running)"
else
    echo "starting..."
    sudo systemctl start litecoind

    # Wait for litecoind to be ready (max 30 seconds)
    echo -n "  Waiting for litecoind RPC to be ready... "
    for i in {1..30}; do
        if $(pwd)/scripts/litecoin_rpc_test.sh &>/dev/null; then
            echo -e "${GREEN}OK${NC}"
            break
        fi
        sleep 1
    done

    if ! $(pwd)/scripts/litecoin_rpc_test.sh &>/dev/null; then
        echo -e "${RED}TIMEOUT${NC}"
        echo "litecoind started but RPC not responding after 30 seconds"
        echo "Check logs: journalctl -u litecoind -f"
        exit 1
    fi
fi

# Show brief litecoind status
echo
echo "Litecoin node status:"
source env/.env
litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" getblockchaininfo | grep -E '"chain"|"blocks"|"headers"|"verificationprogress"' || true
echo

# Check if Proofgold Lava binary exists
if [ ! -x "$LAVA_BINARY" ]; then
    echo -e "${RED}ERROR: Proofgold Lava binary not found or not executable${NC}"
    echo "Expected: $LAVA_BINARY"
    echo "Run ./scripts/30_build_lava.sh first"
    exit 1
fi

# Check if Proofgold is already running
echo -n "Checking Proofgold Lava status... "
if pgrep -f "proofgoldlava" >/dev/null; then
    echo -e "${YELLOW}WARNING${NC} (already running)"
    echo
    echo "Current Proofgold process:"
    ps aux | grep proofgoldlava | grep -v grep
    echo
    read -p "Kill and restart? [y/N]: " RESTART
    if [[ "$RESTART" =~ ^[Yy]$ ]]; then
        echo "Stopping existing Proofgold process..."
        pkill -f "proofgoldlava" || true
        sleep 2
    else
        echo "Keeping existing process running"
        exit 0
    fi
fi

# Ask user how to run Proofgold
echo
echo "Start Proofgold Lava:"
echo "  1) Daemon mode (runs in background, recommended)"
echo "  2) Foreground mode (runs in this terminal, for debugging)"
echo "  3) tmux session (background, with easy reattachment)"
echo
read -p "Select mode [1-3] (default: 1): " RUN_MODE
RUN_MODE=${RUN_MODE:-1}

case $RUN_MODE in
    1)
        # Daemon mode
        echo "Starting Proofgold Lava in daemon mode..."

        # Create a systemd user service
        mkdir -p "$HOME/.config/systemd/user"

        cat > "$HOME/.config/systemd/user/proofgoldlava.service" <<EOF
[Unit]
Description=Proofgold Lava Client
After=network.target

[Service]
Type=simple
ExecStart=$LAVA_BINARY -datadir=$PROOFGOLD_DIR
Restart=on-failure
RestartSec=10
StandardOutput=append:$LOG_DIR/proofgoldlava.log
StandardError=append:$LOG_DIR/proofgoldlava.log

[Install]
WantedBy=default.target
EOF

        systemctl --user daemon-reload
        systemctl --user enable proofgoldlava
        systemctl --user start proofgoldlava

        echo -e "${GREEN}OK${NC}"
        echo
        echo "Proofgold Lava running as systemd user service"
        echo "  Status:  systemctl --user status proofgoldlava"
        echo "  Logs:    journalctl --user -u proofgoldlava -f"
        echo "  Stop:    systemctl --user stop proofgoldlava"
        echo "  Restart: systemctl --user restart proofgoldlava"
        ;;

    2)
        # Foreground mode
        echo "Starting Proofgold Lava in foreground..."
        echo "Press Ctrl+C to stop"
        echo
        $LAVA_BINARY -datadir="$PROOFGOLD_DIR"
        ;;

    3)
        # tmux mode
        if ! command -v tmux &>/dev/null; then
            echo -e "${RED}ERROR: tmux not installed${NC}"
            echo "Install with: sudo apt-get install tmux"
            exit 1
        fi

        SESSION_NAME="proofgold"

        # Check if session already exists
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo -e "${YELLOW}tmux session '$SESSION_NAME' already exists${NC}"
            echo "Attach with: tmux attach -t $SESSION_NAME"
            exit 0
        fi

        echo "Starting Proofgold Lava in tmux session '$SESSION_NAME'..."
        tmux new-session -d -s "$SESSION_NAME" "$LAVA_BINARY -datadir=$PROOFGOLD_DIR"

        echo -e "${GREEN}OK${NC}"
        echo
        echo "Proofgold Lava running in tmux session"
        echo "  Attach:  tmux attach -t $SESSION_NAME"
        echo "  Detach:  Ctrl+B, then D"
        echo "  Kill:    tmux kill-session -t $SESSION_NAME"
        ;;

    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

# Give Proofgold a moment to start
sleep 3

# Check if it's running
echo
echo -n "Verifying Proofgold process... "
if pgrep -f "proofgoldlava" >/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Process not found. Check logs:"
    if [ -f "$LOG_DIR/proofgoldlava.log" ]; then
        tail -20 "$LOG_DIR/proofgoldlava.log"
    fi
    exit 1
fi

echo
echo "=========================================="
echo -e "${GREEN}Stack started successfully!${NC}"
echo "=========================================="
echo
echo "Services running:"
echo "  Litecoin:  systemctl status litecoind"
echo "  Proofgold: (mode $RUN_MODE)"
echo
echo "Logs:"
echo "  Litecoin:  journalctl -u litecoind -f"
echo "  Proofgold: $LOG_DIR/proofgoldlava.log"
echo
echo "Next steps:"
echo "  ./scripts/60_healthcheck.sh (verify sync)"
