#!/usr/bin/env bash
set -euo pipefail
cd /home/player2/signal_experiments 2>/dev/null || exit 0

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
DIRTY=$(git diff --quiet --ignore-submodules HEAD -- 2>/dev/null && echo "clean" || echo "dirty")
WT=$(git rev-parse --git-dir 2>/dev/null | grep -q "/worktrees/" && echo "wt" || echo "")

# Count cert families (fast — just count dirs with mapping_protocol*)
CERTS=$(find qa_alphageometry_ptolemy -maxdepth 2 -name 'mapping_protocol*.json' 2>/dev/null | wc -l)

# Last axiom linter result
LINT="ok"
[ -f .claude/.lint_fail ] && LINT="LINT!"

printf "%s %s | %s certs | %s%s" "$BRANCH" "$DIRTY" "$CERTS" "$LINT" "${WT:+ | $WT}"
