#!/bin/bash
# 40_config_proofgold.sh - Configure Proofgold datadir and config file
# Safe to run multiple times (idempotent)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ENV_FILE="$(pwd)/env/.env"
PROOFGOLD_DIR="$HOME/.proofgold"
CONFIG_FILE="$PROOFGOLD_DIR/proofgold.conf"

echo "=========================================="
echo "Proofgold Lava: Configuration"
echo "=========================================="
echo

# Load environment variables
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}ERROR: Environment file not found: $ENV_FILE${NC}"
    echo "Run ./scripts/20_setup_litecoin.sh first"
    exit 1
fi

source "$ENV_FILE"

# Verify required variables
if [ -z "${LITECOIN_RPC_USER:-}" ] || [ -z "${LITECOIN_RPC_PASS:-}" ]; then
    echo -e "${RED}ERROR: Litecoin RPC credentials not set in $ENV_FILE${NC}"
    exit 1
fi

# Create Proofgold data directory
echo -n "Creating Proofgold data directory... "
mkdir -p "$PROOFGOLD_DIR"
echo -e "${GREEN}OK${NC}"

# Ask user about network mode
echo
echo "Network configuration:"
echo "  1) Bootstrap mode (faster, ~1-2 hours, uses default peers)"
echo "  2) Independent bootstrap (slower, more secure, doesn't rely on peers)"
echo
read -p "Select mode [1-2] (default: 1): " NETWORK_MODE
NETWORK_MODE=${NETWORK_MODE:-1}

if [ "$NETWORK_MODE" = "2" ]; then
    INDEPENDENT_BOOTSTRAP=1
else
    INDEPENDENT_BOOTSTRAP=0
fi

# Ask about listening for connections
echo
read -p "Listen for incoming peer connections? [y/N]: " LISTEN_CHOICE
if [[ "$LISTEN_CHOICE" =~ ^[Yy]$ ]]; then
    echo -n "Enter IP to bind to (default: 0.0.0.0 for all interfaces): "
    read LISTEN_IP
    LISTEN_IP=${LISTEN_IP:-0.0.0.0}
    LISTEN_PORT=21805  # Default Proofgold port
else
    LISTEN_IP=""
    LISTEN_PORT=""
fi

# Create proofgold.conf
echo
echo -n "Creating Proofgold configuration... "

cat > "$CONFIG_FILE" <<EOF
# Proofgold Configuration
# Generated on $(date)

# Litecoin RPC connection
ltcrpcuser=$LITECOIN_RPC_USER
ltcrpcpass=$LITECOIN_RPC_PASS
ltcrpchost=127.0.0.1
ltcrpcport=9332

# Data directory
datadir=$PROOFGOLD_DIR

# Network settings
independentbootstrap=$INDEPENDENT_BOOTSTRAP
EOF

# Add listening configuration if enabled
if [ -n "$LISTEN_IP" ]; then
    cat >> "$CONFIG_FILE" <<EOF

# Peer listening
ip=$LISTEN_IP
port=$LISTEN_PORT
maxconns=20
EOF
fi

# Add logging settings
cat >> "$CONFIG_FILE" <<EOF

# Logging
logfile=$PROOFGOLD_DIR/proofgold.log
loglevel=info
EOF

chmod 600 "$CONFIG_FILE"
echo -e "${GREEN}OK${NC}"

# Create database directories
echo -n "Creating database directories... "
mkdir -p "$PROOFGOLD_DIR/db"
mkdir -p "$PROOFGOLD_DIR/blocks"
echo -e "${GREEN}OK${NC}"

# Create a helper script to show Proofgold status
cat > "$(pwd)/scripts/proofgold_status.sh" <<'EOF'
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
EOF

chmod +x "$(pwd)/scripts/proofgold_status.sh"

# Update .env with Proofgold directory
if ! grep -q "PROOFGOLD_DIR" "$ENV_FILE"; then
    echo "PROOFGOLD_DIR=\"$PROOFGOLD_DIR\"" >> "$ENV_FILE"
fi

echo
echo "=========================================="
echo -e "${GREEN}Proofgold configured successfully!${NC}"
echo "=========================================="
echo
echo "Configuration:"
echo "  Data directory: $PROOFGOLD_DIR"
echo "  Config file: $CONFIG_FILE (mode 600)"
echo "  Bootstrap mode: $([ $INDEPENDENT_BOOTSTRAP -eq 1 ] && echo 'independent' || echo 'default')"
if [ -n "$LISTEN_IP" ]; then
    echo "  Listening on: $LISTEN_IP:$LISTEN_PORT"
else
    echo "  Listening: disabled (outbound only)"
fi
echo
echo "Directories created:"
echo "  $PROOFGOLD_DIR/db (blockchain database)"
echo "  $PROOFGOLD_DIR/blocks (block storage)"
echo
echo "Helper scripts:"
echo "  ./scripts/proofgold_status.sh - View node status and logs"
echo
echo "Next steps:"
echo "  ./scripts/50_start_stack.sh"
