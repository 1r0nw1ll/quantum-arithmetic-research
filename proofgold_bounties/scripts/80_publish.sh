#!/bin/bash
# 80_publish.sh - Complete publishing workflow
# commit → wait confirmations → publish → collect bounties

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

LAVA_BINARY="$(pwd)/scripts/proofgoldlava"
DRAFTS_DIR="$(pwd)/drafts"
ENV_FILE="$(pwd)/env/.env"

REQUIRED_CONFIRMATIONS=12  # Per Proofgold docs

echo "=========================================="
echo "Proofgold Publishing Workflow"
echo "=========================================="
echo

# Check if Proofgold is running
if ! pgrep -f "proofgoldlava" >/dev/null; then
    echo -e "${RED}ERROR: Proofgold Lava is not running${NC}"
    echo "Start it first: ./scripts/50_start_stack.sh"
    exit 1
fi

# Load environment
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Select draft to publish
echo "Available drafts:"
DRAFTS=($(ls -1 "$DRAFTS_DIR"/*.pfg 2>/dev/null || echo ""))

if [ ${#DRAFTS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No drafts found in $DRAFTS_DIR${NC}"
    echo "Create one with: ./scripts/70_bounty_workflow.sh"
    exit 1
fi

i=1
for draft in "${DRAFTS[@]}"; do
    echo "  $i) $(basename $draft)"
    ((i++))
done

echo
read -p "Select draft number to publish: " DRAFT_NUM

SELECTED_DRAFT="${DRAFTS[$((DRAFT_NUM-1))]}"

if [ ! -f "$SELECTED_DRAFT" ]; then
    echo -e "${RED}Invalid selection${NC}"
    exit 1
fi

echo
echo "Selected: $(basename $SELECTED_DRAFT)"
echo

# Check wallet balance before proceeding
echo "Checking wallet balance..."
WALLET_CHECK=$(timeout 10 $LAVA_BINARY printassets 2>&1 || echo "WALLET_ERROR")

if echo "$WALLET_CHECK" | grep -qi "error\|fail\|not found\|no assets\|0.0*\s*LTC"; then
    echo -e "${RED}⚠ Wallet balance is zero or very low${NC}"
    echo
    echo "Publishing requires a small amount of Litecoin for transaction fees (~0.005-0.01 LTC)."
    echo
    echo "To get started, you have two options:"
    echo
    echo "  Option 1 (Recommended):"
    echo "    Ask an existing Proofgold user to send ~0.005 LTC to your address."
    echo "    This is the normal bootstrap mechanism for new Proofgold users."
    echo
    echo "  Option 2:"
    echo "    Purchase a small amount of LTC (~\$1 USD) from an exchange and transfer."
    echo
    echo "Your Proofgold receive address:"

    # Try to get receiving address
    RECV_ADDR=$(timeout 10 $LAVA_BINARY getaddress 2>&1 || timeout 10 $LAVA_BINARY newaddress 2>&1 || echo "Run: ./scripts/proofgoldlava getaddress")
    echo "  $RECV_ADDR"
    echo
    echo "After receiving funds, wait for 1 confirmation, then re-run this script."
    echo
    read -p "Do you already have LTC and want to proceed anyway? [y/N]: " FORCE_PROCEED

    if [[ ! "$FORCE_PROCEED" =~ ^[Yy]$ ]]; then
        echo "Publishing cancelled. Get LTC first, then try again."
        exit 0
    fi
else
    echo -e "${GREEN}Wallet balance OK${NC}"
    echo "$WALLET_CHECK" | head -5
    echo
fi

# Confirm publication
echo -e "${YELLOW}WARNING: Publishing costs LTC for transaction fees!${NC}"
echo -e "${YELLOW}This transaction will be broadcast to the Litecoin network.${NC}"
echo
read -p "Proceed with publication? [y/N]: " CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# ======================================
# Step 1: Add nonce
# ======================================
echo
echo -e "${BLUE}Step 1/5: Adding nonce${NC}"
echo "Running: addnonce"

if timeout 30 $LAVA_BINARY addnonce "$SELECTED_DRAFT" 2>&1; then
    echo -e "${GREEN}✓ Nonce added${NC}"
else
    echo -e "${RED}✗ Failed to add nonce${NC}"
    exit 1
fi

# ======================================
# Step 2: Add publisher
# ======================================
echo
echo -e "${BLUE}Step 2/5: Adding publisher${NC}"
echo "Running: addpublisher"

if timeout 30 $LAVA_BINARY addpublisher "$SELECTED_DRAFT" 2>&1; then
    echo -e "${GREEN}✓ Publisher added${NC}"
else
    echo -e "${RED}✗ Failed to add publisher${NC}"
    exit 1
fi

# ======================================
# Step 3: Commit draft
# ======================================
echo
echo -e "${BLUE}Step 3/5: Committing draft to blockchain${NC}"
echo "Running: commitdraft"
echo "This creates a commitment transaction on Litecoin..."

COMMIT_OUTPUT=$(timeout 60 $LAVA_BINARY commitdraft "$SELECTED_DRAFT" 2>&1) || {
    echo -e "${RED}✗ Commit failed${NC}"
    echo "$COMMIT_OUTPUT"
    exit 1
}

echo "$COMMIT_OUTPUT"
echo -e "${GREEN}✓ Draft committed${NC}"

# Try to extract transaction ID from output (format may vary)
COMMIT_TXID=$(echo "$COMMIT_OUTPUT" | grep -oP 'txid: \K[a-f0-9]{64}' || echo "")

if [ -n "$COMMIT_TXID" ]; then
    echo "Commitment TXID: $COMMIT_TXID"
fi

# ======================================
# Step 4: Wait for confirmations
# ======================================
echo
echo -e "${BLUE}Step 4/5: Waiting for $REQUIRED_CONFIRMATIONS confirmations${NC}"
echo "This may take 30+ minutes (Litecoin block time ~2.5 min)"
echo

CONFIRMATIONS=0
START_TIME=$(date +%s)

while [ "$CONFIRMATIONS" -lt "$REQUIRED_CONFIRMATIONS" ]; do
    # Query Litecoin for confirmations
    if [ -n "$COMMIT_TXID" ]; then
        # If we have the txid, query it directly
        TX_INFO=$(litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" \
                  gettransaction "$COMMIT_TXID" 2>/dev/null || echo "")

        if [ -n "$TX_INFO" ]; then
            CONFIRMATIONS=$(echo "$TX_INFO" | jq -r '.confirmations // 0' 2>/dev/null || echo "0")
        fi
    else
        # Fallback: estimate based on current block height
        # This is less precise but works if we can't get txid
        CURRENT_BLOCKS=$(litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" \
                        getblockcount 2>/dev/null || echo "0")

        # We'd need to know the commit block height for this to work
        # For now, just show elapsed time
        ELAPSED=$(($(date +%s) - START_TIME))
        echo -ne "\r  Elapsed: ${ELAPSED}s | Checking confirmations... "
    fi

    if [ "$CONFIRMATIONS" -gt 0 ]; then
        PROGRESS=$((CONFIRMATIONS * 100 / REQUIRED_CONFIRMATIONS))
        echo -ne "\r  Confirmations: $CONFIRMATIONS / $REQUIRED_CONFIRMATIONS [$PROGRESS%]    "
    fi

    if [ "$CONFIRMATIONS" -lt "$REQUIRED_CONFIRMATIONS" ]; then
        sleep 30  # Check every 30 seconds
    fi
done

echo
echo -e "${GREEN}✓ $REQUIRED_CONFIRMATIONS confirmations received${NC}"

# ======================================
# Step 5: Publish draft
# ======================================
echo
echo -e "${BLUE}Step 5/5: Publishing draft to blockchain${NC}"
echo "Running: publishdraft"

PUBLISH_OUTPUT=$(timeout 60 $LAVA_BINARY publishdraft "$SELECTED_DRAFT" 2>&1) || {
    echo -e "${RED}✗ Publish failed${NC}"
    echo "$PUBLISH_OUTPUT"
    exit 1
}

echo "$PUBLISH_OUTPUT"
echo -e "${GREEN}✓ Draft published!${NC}"

# ======================================
# Bonus: Collect bounties
# ======================================
echo
echo -e "${BLUE}Bonus: Attempting to collect bounties${NC}"
echo "Running: collectbounties"

COLLECT_OUTPUT=$(timeout 60 $LAVA_BINARY collectbounties 2>&1) || {
    echo -e "${YELLOW}Warning: collectbounties command failed${NC}"
    echo "$COLLECT_OUTPUT"
}

if echo "$COLLECT_OUTPUT" | grep -qi "success\|collected\|claimed"; then
    echo -e "${GREEN}✓ Bounties collected!${NC}"
    echo "$COLLECT_OUTPUT"
elif echo "$COLLECT_OUTPUT" | grep -qi "no bounties\|nothing to collect"; then
    echo -e "${YELLOW}No bounties to collect (yet)${NC}"
else
    echo "$COLLECT_OUTPUT"
fi

echo
echo "=========================================="
echo -e "${GREEN}Publication Complete!${NC}"
echo "=========================================="
echo
echo "Your proof has been published to the Proofgold blockchain."
echo
echo "Next steps:"
echo "  1. Check Proofgold explorer for your document"
echo "  2. If bounty eligible, check wallet for rewards"
echo "  3. Create more drafts: ./scripts/70_bounty_workflow.sh"
echo
echo "To check your assets:"
echo "  $LAVA_BINARY printassets"
echo

# Save publication record
RECORD_FILE="$DRAFTS_DIR/publications.log"
cat >> "$RECORD_FILE" <<EOF
$(date): Published $(basename $SELECTED_DRAFT)
  Commit TXID: ${COMMIT_TXID:-unknown}
  Confirmations: $REQUIRED_CONFIRMATIONS
  Status: SUCCESS
---
EOF

echo "Publication logged to: $RECORD_FILE"
