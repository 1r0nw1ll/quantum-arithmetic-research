#!/bin/bash
# 20_setup_litecoin.sh - Install and configure Litecoin node with RPC
# Safe to run multiple times (idempotent)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LITECOIN_VERSION="0.21.2.2"  # Latest stable as of Jan 2026
LITECOIN_DIR="$HOME/.litecoin"
DATA_DIR="$(pwd)/data/litecoin"
ENV_FILE="$(pwd)/env/.env"

echo "=========================================="
echo "Proofgold Lava: Litecoin Node Setup"
echo "=========================================="
echo

# Load or generate RPC credentials
if [ -f "$ENV_FILE" ]; then
    echo "Loading credentials from $ENV_FILE"
    source "$ENV_FILE"
else
    echo "Generating new RPC credentials..."
    mkdir -p "$(pwd)/env"

    # Generate strong random password
    RPC_USER="proofgold_$(openssl rand -hex 4)"
    RPC_PASS="$(openssl rand -base64 32 | tr -d '/+=' | head -c 32)"

    cat > "$ENV_FILE" <<EOF
# Proofgold Lava Environment Configuration
# Generated on $(date)

# Litecoin RPC Credentials (KEEP THESE SECRET!)
LITECOIN_RPC_USER="$RPC_USER"
LITECOIN_RPC_PASS="$RPC_PASS"

# Data directories
LITECOIN_DIR="$LITECOIN_DIR"
PROOFGOLD_DIR="$HOME/.proofgold"

# Network settings
LITECOIN_NETWORK="mainnet"  # or "testnet" for testing
EOF

    echo -e "${GREEN}Credentials saved to $ENV_FILE${NC}"
    echo -e "${YELLOW}IMPORTANT: Keep this file secret!${NC}"
fi

# Reload to ensure we have the vars
source "$ENV_FILE"

# Check if litecoind is already installed
echo -n "Checking for litecoind... "
if command -v litecoind &>/dev/null; then
    INSTALLED_VERSION=$(litecoind --version | head -1 | awk '{print $NF}')
    echo -e "${YELLOW}SKIP${NC} (already installed: $INSTALLED_VERSION)"
else
    echo "not found, installing..."

    # Install from official binaries
    ARCH=$(uname -m)
    if [ "$ARCH" == "x86_64" ]; then
        LITECOIN_ARCH="x86_64-linux-gnu"
    elif [ "$ARCH" == "aarch64" ]; then
        LITECOIN_ARCH="aarch64-linux-gnu"
    else
        echo -e "${RED}Unsupported architecture: $ARCH${NC}"
        exit 1
    fi

    LITECOIN_URL="https://download.litecoin.org/litecoin-${LITECOIN_VERSION}/linux/litecoin-${LITECOIN_VERSION}-${LITECOIN_ARCH}.tar.gz"

    echo "  Downloading Litecoin $LITECOIN_VERSION for $LITECOIN_ARCH..."
    wget -q --show-progress "$LITECOIN_URL" -O /tmp/litecoin.tar.gz

    echo "  Extracting..."
    tar -xzf /tmp/litecoin.tar.gz -C /tmp

    echo "  Installing to /usr/local/bin (requires sudo)..."
    sudo install -m 0755 -o root -g root -t /usr/local/bin \
        /tmp/litecoin-${LITECOIN_VERSION}/bin/litecoind \
        /tmp/litecoin-${LITECOIN_VERSION}/bin/litecoin-cli

    rm -rf /tmp/litecoin.tar.gz /tmp/litecoin-${LITECOIN_VERSION}
    echo -e "${GREEN}OK${NC}"
fi

# Create Litecoin data directory
echo -n "Creating Litecoin data directory... "
mkdir -p "$LITECOIN_DIR"
echo -e "${GREEN}OK${NC}"

# Create litecoin.conf
echo -n "Creating litecoin.conf... "
cat > "$LITECOIN_DIR/litecoin.conf" <<EOF
# Litecoin Configuration for Proofgold Lava
# Generated on $(date)

# Server mode (required for RPC)
server=1
daemon=1

# RPC credentials (SECURITY CRITICAL)
rpcuser=$LITECOIN_RPC_USER
rpcpassword=$LITECOIN_RPC_PASS

# Lock RPC to localhost only (SECURITY CRITICAL)
rpcallowip=127.0.0.1
rpcbind=127.0.0.1

# RPC port (default: 9332 for mainnet, 19332 for testnet)
rpcport=9332

# Network
$([ "$LITECOIN_NETWORK" = "testnet" ] && echo "testnet=1" || echo "# mainnet (default)")

# Performance tuning
maxconnections=125
dbcache=450

# Logging
debug=0
EOF

chmod 600 "$LITECOIN_DIR/litecoin.conf"
echo -e "${GREEN}OK${NC}"

# Create systemd service
echo -n "Creating systemd service... "
sudo tee /etc/systemd/system/litecoind.service > /dev/null <<EOF
[Unit]
Description=Litecoin Core Daemon
After=network.target

[Service]
Type=forking
User=$USER
Group=$USER
ExecStart=/usr/local/bin/litecoind -conf=$LITECOIN_DIR/litecoin.conf -datadir=$LITECOIN_DIR
ExecStop=/usr/local/bin/litecoin-cli -conf=$LITECOIN_DIR/litecoin.conf stop
Restart=on-failure
RestartSec=10

# Security hardening
PrivateTmp=true
ProtectSystem=full
NoNewPrivileges=true
PrivateDevices=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo -e "${GREEN}OK${NC}"

# Enable service (but don't start yet)
echo -n "Enabling litecoind service... "
sudo systemctl enable litecoind &>/dev/null
echo -e "${GREEN}OK${NC}"

# Create RPC test script
echo -n "Creating RPC test script... "
cat > "$(pwd)/scripts/litecoin_rpc_test.sh" <<'EOF'
#!/bin/bash
# Test Litecoin RPC connectivity

source "$(dirname $0)/../env/.env"

echo "Testing Litecoin RPC..."
litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" getblockchaininfo
EOF

chmod +x "$(pwd)/scripts/litecoin_rpc_test.sh"
echo -e "${GREEN}OK${NC}"

echo
echo "=========================================="
echo -e "${GREEN}Litecoin node configured!${NC}"
echo "=========================================="
echo
echo "Configuration:"
echo "  Data directory: $LITECOIN_DIR"
echo "  RPC user: $LITECOIN_RPC_USER"
echo "  RPC bind: 127.0.0.1:9332"
echo "  Network: $LITECOIN_NETWORK"
echo
echo "Security notes:"
echo "  - RPC is locked to localhost only"
echo "  - Credentials stored in: $ENV_FILE"
echo "  - Config file: $LITECOIN_DIR/litecoin.conf (mode 600)"
echo
echo "Service management:"
echo "  Start:   sudo systemctl start litecoind"
echo "  Stop:    sudo systemctl stop litecoind"
echo "  Status:  sudo systemctl status litecoind"
echo "  Logs:    journalctl -u litecoind -f"
echo
echo "Next steps:"
echo "  ./scripts/30_build_lava.sh"
