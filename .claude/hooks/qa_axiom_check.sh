#!/bin/bash
# PostToolUse hook: run QA axiom linter on edited .py files
#
# HOOK DESIGN PRINCIPLES (Lesson 8):
# 1. Skip files that can't contain QA violations (non-.py, configs, tests)
# 2. Only BLOCK on errors, not warnings (warnings are often false positives
#    in observer scripts where 'd' and 'a' aren't QA variables)
# 3. Show warnings but don't block — let the developer decide
# 4. Skip files outside the repo (temp files, venv, etc.)

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Skip non-Python files
[[ -z "$FILE_PATH" || "$FILE_PATH" != *.py ]] && exit 0

# Skip files outside the repo
REPO="/home/player2/signal_experiments"
[[ "$FILE_PATH" != "$REPO"/* ]] && exit 0

# Skip directories that don't contain QA logic
REL="${FILE_PATH#$REPO/}"
case "$REL" in
  .claude/*|archive/*|Documents/*|QAnotes/*|qa_lab/*|*venv*|*__pycache__*|tools/qa_axiom_linter.py|tools/tests/linter_fixtures/*)
    # qa_lab/* is in the linter's _EXCLUDE_DIRS — mirror that here so
    # bulk scans and per-file hook checks agree.
    # tools/tests/linter_fixtures/* contains INTENTIONALLY bad fixtures that
    # the linter test harness exercises. Skipping here so edits to the
    # fixtures themselves don't block; they are still scanned when the
    # test harness test_qa_axiom_linter_firewall.py runs.
    exit 0
    ;;
esac

# Run the linter
cd "$REPO"
PYTHON="${REPO}/.venv/bin/python"
[ ! -x "$PYTHON" ] && PYTHON=python3
RESULT=$($PYTHON tools/qa_axiom_linter.py "$FILE_PATH" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  # Check if there are actual ERRORS (not just warnings)
  ERROR_COUNT=$(echo "$RESULT" | grep -oP '\d+ error' | grep -oP '\d+')
  if [ "${ERROR_COUNT:-0}" -gt 0 ]; then
    echo "QA AXIOM VIOLATION in $REL:"
    echo "$RESULT"
    echo ""
    echo "Fix before proceeding. See QA_AXIOMS_BLOCK.md"
    exit 2  # BLOCK
  else
    # Warnings only — show but don't block
    echo "QA linter warnings in $REL (not blocking):"
    echo "$RESULT"
    exit 0
  fi
fi

exit 0
