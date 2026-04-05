#!/bin/bash
# 60_healthcheck.sh - Comprehensive health check for the Proofgold stack
# Safe to run multiple times

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

ENV_FILE="$(pwd)/env/.env"
PROOFGOLD_DIR="$HOME/.proofgold"
LAVA_BINARY="$(pwd)/scripts/proofgoldlava"

echo "=========================================="
echo "Proofgold Stack Health Check"
echo "=========================================="
echo

# Load credentials
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
else
    echo -e "${RED}✗ Environment file not found${NC}"
    exit 1
fi

# Track overall health
HEALTH_OK=true

# 1. Check Litecoin daemon
echo -e "${BLUE}[1/6] Litecoin Node${NC}"
echo -n "  Service status: "
if systemctl is-active --quiet litecoind; then
    echo -e "${GREEN}RUNNING${NC}"
else
    echo -e "${RED}STOPPED${NC}"
    HEALTH_OK=false
fi

echo -n "  RPC connectivity: "
if litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" getnetworkinfo &>/dev/null; then
    echo -e "${GREEN}OK${NC}"

    # Get detailed blockchain info
    BLOCKCHAIN_INFO=$(litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" getblockchaininfo 2>/dev/null)

    CHAIN=$(echo "$BLOCKCHAIN_INFO" | jq -r '.chain' 2>/dev/null || echo "unknown")
    BLOCKS=$(echo "$BLOCKCHAIN_INFO" | jq -r '.blocks' 2>/dev/null || echo "unknown")
    HEADERS=$(echo "$BLOCKCHAIN_INFO" | jq -r '.headers' 2>/dev/null || echo "unknown")
    PROGRESS=$(echo "$BLOCKCHAIN_INFO" | jq -r '.verificationprogress' 2>/dev/null || echo "0")
    PROGRESS_PCT=$(echo "$PROGRESS * 100" | bc 2>/dev/null | cut -d. -f1 || echo "0")

    echo "  Network: $CHAIN"
    echo "  Blocks: $BLOCKS / $HEADERS"
    echo "  Sync progress: ${PROGRESS_PCT}%"

    if [ "$BLOCKS" != "$HEADERS" ]; then
        echo -e "  ${YELLOW}⚠ Still syncing...${NC}"
    else
        echo -e "  ${GREEN}✓ Fully synced${NC}"
    fi

    # Get connection count
    CONNECTIONS=$(litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" getconnectioncount 2>/dev/null || echo "0")
    echo "  Peer connections: $CONNECTIONS"

else
    echo -e "${RED}FAILED${NC}"
    HEALTH_OK=false
fi

echo

# 2. Check Proofgold Lava process
echo -e "${BLUE}[2/6] Proofgold Lava Client${NC}"
echo -n "  Process status: "
if pgrep -f "proofgoldlava" >/dev/null; then
    echo -e "${GREEN}RUNNING${NC}"

    PID=$(pgrep -f "proofgoldlava" | head -1)
    UPTIME=$(ps -o etime= -p $PID | tr -d ' ')
    MEM=$(ps -o rss= -p $PID | awk '{printf "%.1f MB", $1/1024}')

    echo "  PID: $PID"
    echo "  Uptime: $UPTIME"
    echo "  Memory: $MEM"
else
    echo -e "${RED}STOPPED${NC}"
    HEALTH_OK=false
fi

echo

# 3. Check Proofgold data directory
echo -e "${BLUE}[3/6] Proofgold Data Directory${NC}"
echo -n "  Directory exists: "
if [ -d "$PROOFGOLD_DIR" ]; then
    echo -e "${GREEN}OK${NC}"

    SIZE=$(du -sh "$PROOFGOLD_DIR" 2>/dev/null | cut -f1)
    echo "  Data size: $SIZE"

    # Check for key subdirectories
    echo -n "  Database: "
    if [ -d "$PROOFGOLD_DIR/db" ]; then
        DB_SIZE=$(du -sh "$PROOFGOLD_DIR/db" 2>/dev/null | cut -f1 || echo "0")
        echo -e "${GREEN}OK${NC} ($DB_SIZE)"
    else
        echo -e "${YELLOW}NOT CREATED${NC}"
    fi

    echo -n "  Blocks: "
    if [ -d "$PROOFGOLD_DIR/blocks" ]; then
        BLOCKS_SIZE=$(du -sh "$PROOFGOLD_DIR/blocks" 2>/dev/null | cut -f1 || echo "0")
        echo -e "${GREEN}OK${NC} ($BLOCKS_SIZE)"
    else
        echo -e "${YELLOW}NOT CREATED${NC}"
    fi

else
    echo -e "${RED}MISSING${NC}"
    HEALTH_OK=false
fi

echo

# 4. Check Proofgold logs
echo -e "${BLUE}[4/6] Proofgold Logs${NC}"
if [ -f "$PROOFGOLD_DIR/proofgold.log" ]; then
    echo "  Recent log entries (last 5 lines):"
    tail -5 "$PROOFGOLD_DIR/proofgold.log" | sed 's/^/    /'

    # Check for common error patterns
    echo -n "  Error check: "
    ERROR_COUNT=$(grep -i "error\|fail\|exception" "$PROOFGOLD_DIR/proofgold.log" 2>/dev/null | wc -l)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}$ERROR_COUNT errors in log${NC}"
    else
        echo -e "${GREEN}No errors detected${NC}"
    fi
else
    echo -e "  ${YELLOW}No log file yet (node may be starting)${NC}"
fi

echo

# 5. Check Proofgold sync status (if we can query it)
echo -e "${BLUE}[5/6] Proofgold Sync Status${NC}"

# Try to query Proofgold status (this depends on the actual Lava interface)
# Common commands might be: proofgoldlava status, or a RPC-style query
# For now, we'll check if the node is responding

echo -n "  Checking if node responds... "

# Try different potential status commands
if [ -x "$LAVA_BINARY" ]; then
    # Try --status flag
    if timeout 5 $LAVA_BINARY --status &>/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        $LAVA_BINARY --status 2>&1 | head -10 | sed 's/^/    /'
    # Try status subcommand
    elif timeout 5 $LAVA_BINARY status &>/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        $LAVA_BINARY status 2>&1 | head -10 | sed 's/^/    /'
    else
        echo -e "${YELLOW}Cannot query (interface unknown)${NC}"
        echo "  This is normal if the node doesn't have a status command"
    fi
else
    echo -e "${RED}Binary not found${NC}"
    HEALTH_OK=false
fi

echo

# 6. Overall system health
echo -e "${BLUE}[6/6] System Resources${NC}"

# Disk space
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
DISK_USE_PCT=$(df -h . | tail -1 | awk '{print $5}')
echo "  Disk available: $DISK_AVAIL (${DISK_USE_PCT} used)"

# Memory
MEM_FREE=$(free -h | awk '/Mem:/ {print $7}')
MEM_TOTAL=$(free -h | awk '/Mem:/ {print $2}')
echo "  Memory free: $MEM_FREE / $MEM_TOTAL"

# Load average
LOAD=$(uptime | awk -F'load average:' '{print $2}' | xargs)
echo "  Load average: $LOAD"

echo
echo "=========================================="

if [ "$HEALTH_OK" = true ]; then
    echo -e "${GREEN}✓ Health check PASSED${NC}"
    echo "=========================================="
    echo
    echo "Your Proofgold stack is operational!"
    echo
    echo "Next steps:"
    echo "  - Wait for Litecoin to fully sync (may take hours)"
    echo "  - Wait for Proofgold to sync (may take 1-2 hours)"
    echo "  - Then proceed to: ./scripts/70_bounty_workflow.sh"
else
    echo -e "${RED}✗ Health check FAILED${NC}"
    echo "=========================================="
    echo
    echo "Issues detected. Please review the output above."
    echo
    echo "Troubleshooting:"
    echo "  - Check Litecoin logs: journalctl -u litecoind -f"
    echo "  - Check Proofgold logs: tail -f $PROOFGOLD_DIR/proofgold.log"
    echo "  - Restart services: ./scripts/50_start_stack.sh"
    exit 1
fi
