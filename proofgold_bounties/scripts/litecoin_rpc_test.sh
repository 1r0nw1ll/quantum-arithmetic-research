#!/bin/bash
# Test Litecoin RPC connectivity

source "$(dirname $0)/../env/.env"

echo "Testing Litecoin RPC..."
litecoin-cli -rpcuser="$LITECOIN_RPC_USER" -rpcpassword="$LITECOIN_RPC_PASS" getblockchaininfo
