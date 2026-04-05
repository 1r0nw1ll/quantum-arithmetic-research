#!/bin/bash
# Helper script to look up proposition details once we have the ID

PROP_ID=$1

if [ -z "$PROP_ID" ]; then
    echo "Usage: $0 <proposition_id>"
    echo "Example: $0 PUPxxxx..."
    exit 1
fi

echo "Looking up proposition: $PROP_ID"
echo "This would query the local Proofgold database or web API"
echo ""
echo "For now, use the web explorer:"
echo "https://proofgold.org/explorer"
