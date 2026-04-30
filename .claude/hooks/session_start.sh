#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" 2>/dev/null || exit 0

echo "=== QA Session Start ==="
echo "Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "Uncommitted: $(git status --short 2>/dev/null | wc -l) files"

# Days since last commit
LAST=$(git log -1 --format=%ct 2>/dev/null || echo 0)
NOW=$(date +%s)
DAYS=$(( (NOW - LAST) / 86400 ))
if [ "$DAYS" -gt 7 ]; then
  echo "WARNING: $DAYS days since last commit (hygiene rule: max 7)"
fi

# Remind about OB protocol
echo ""
echo "OB protocol: run mcp__open-brain__recent_thoughts (3 days) before starting work"
