#!/bin/bash
# qa-collab MCP launcher — auto-detects REPO from this script's location.
# Used by .mcp.json so the same config works on Linux and macOS without
# hardcoded /home/player2 vs /Users/player3 paths.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/.venv/bin/python" "$REPO/qa_lab/qa_mcp_servers/qa-collab/server.py" "$@"
