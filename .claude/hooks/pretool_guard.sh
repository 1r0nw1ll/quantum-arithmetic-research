#!/usr/bin/env bash
# PreToolUse hook: legacy path/asset guard.
#
# Hard blocks live here for off-limits directories and generated PNGs.
# Cert-adjacent collab-marker enforcement now lives in
# llm_qa_wrapper/cert_gate_hook.py, which runs first and exits 2.
set -euo pipefail
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARKER="/tmp/qa_collab_session_registered"

# Resolve to relative path from repo root
REL=$(realpath --relative-to="$REPO" "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# ── Hard blocks for off-limits locations ──────────────────────────────
# Resolve to canonical absolute path to defeat relative-path bypass (../Desktop/qa_finance/)
CANON=$(realpath -m "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

case "$CANON" in
  /home/player2/Desktop/qa_finance/*|/Users/player3/Desktop/qa_finance/*|"$HOME/Desktop/qa_finance/"*)
    echo "BLOCKED: qa_finance is frozen (hashes in FROZEN_HASHES_30_42.txt)" >&2
    exit 2
    ;;
esac

case "$REL" in
  Documents/wildberger_corpus/*.[pP][dD][fF]|Documents/haramein_rsf/*.[pP][dD][fF])
    # Narrow Phase 4.5 primary-source ingress exception: canonical PDFs only.
    ;;
  archive/*|Documents/*|QAnotes/*)
    echo "BLOCKED: $REL is in a protected directory (Do Not Touch)" >&2
    exit 2
    ;;
  *.[pP][nN][gG])
    echo "BLOCKED: $REL — PNG outputs are generated, not hand-edited" >&2
    exit 2
    ;;
esac

# ── Collab-bus heartbeat for cert-adjacent edits ─────────────────────
# Informational only — never blocks.
case "$REL" in
  qa_alphageometry_ptolemy/*|tools/qa_*|tools/tests/*|.claude/hooks/*|qa_pim/*|qa_graph/*|qa_observer/*|qa_arithmetic/*|qa_guardrail/*)
    if [ ! -s "$MARKER" ]; then
      echo "Note: editing cert-adjacent path $REL without collab bus session marker"
    fi
    if ! ss -tlnp 2>/dev/null | grep -q ":5555 "; then
      echo "Note: collab bus port 5555 not listening"
    fi
    exit 0
    ;;
esac

exit 0
