#!/usr/bin/env bash
set -euo pipefail
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

[ -z "$FILE_PATH" ] && exit 0

# Resolve to relative path from repo root
REL=$(realpath --relative-to=/home/player2/signal_experiments "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# Check absolute path for locations outside the repo
case "$FILE_PATH" in
  /home/player2/Desktop/qa_finance/*)
    echo "BLOCKED: qa_finance is frozen (hashes in FROZEN_HASHES_30_42.txt)" >&2
    exit 2
    ;;
esac

# Check relative path for locations inside the repo
case "$REL" in
  archive/*|Documents/*|QAnotes/*)
    echo "BLOCKED: $REL is in a protected directory (Do Not Touch)" >&2
    exit 2
    ;;
  *.png)
    echo "BLOCKED: $REL — PNG outputs are generated, not hand-edited" >&2
    exit 2
    ;;
  *)
    exit 0
    ;;
esac
