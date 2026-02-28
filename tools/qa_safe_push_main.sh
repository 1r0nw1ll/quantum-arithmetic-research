#!/usr/bin/env bash
set -euo pipefail

# Guarded push helper for multi-agent environments.
# Usage:
#   tools/qa_safe_push_main.sh <allowed_path_1> [allowed_path_2 ...]
# Example:
#   tools/qa_safe_push_main.sh .github/workflows/qa-ci.yml

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
qa_safe_push_main.sh

Pushes HEAD to origin/main only after safety checks:
1) No git index lock.
2) No unresolved merge/rebase state.
3) Clean working tree outside staged changes.
4) Staged files are limited to explicit allowed path prefixes.
5) HEAD contains latest origin/main (no stale base).

Usage:
  tools/qa_safe_push_main.sh <allowed_path_1> [allowed_path_2 ...]
EOF
  exit 0
fi

if [[ "$#" -lt 1 ]]; then
  echo "ERROR: provide at least one allowed path/prefix." >&2
  echo "Usage: tools/qa_safe_push_main.sh <allowed_path_1> [allowed_path_2 ...]" >&2
  exit 2
fi

if ! git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo "ERROR: not inside a git repository." >&2
  exit 2
fi
cd "$git_root"

if [[ -f .git/index.lock ]]; then
  echo "ERROR: .git/index.lock present. Another git process may be active." >&2
  exit 3
fi

if [[ -f .git/MERGE_HEAD || -d .git/rebase-merge || -d .git/rebase-apply ]]; then
  echo "ERROR: repository is in merge/rebase state. Resolve it before pushing." >&2
  exit 3
fi

if ! git diff --quiet; then
  echo "ERROR: unstaged tracked changes detected. Stage or discard before push." >&2
  git status --short >&2
  exit 4
fi

if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo "ERROR: untracked files detected. Use a clean/sandbox tree for agent pushes." >&2
  git status --short >&2
  exit 4
fi

if git diff --cached --quiet; then
  echo "ERROR: no staged changes to push." >&2
  exit 4
fi

mapfile -t staged_files < <(git diff --cached --name-only)

for file in "${staged_files[@]}"; do
  allowed_match=false
  for allowed in "$@"; do
    # Allow exact file, directory prefix, and glob/pattern matching.
    if [[ "$file" == "$allowed" || "$file" == "$allowed/"* || "$file" == $allowed ]]; then
      allowed_match=true
      break
    fi
  done
  if [[ "$allowed_match" == false ]]; then
    echo "ERROR: staged path not in allowlist: $file" >&2
    echo "Allowed prefixes: $*" >&2
    exit 5
  fi
done

echo "[qa-safe-push] Fetching origin/main..."
git fetch origin main

origin_main_sha="$(git rev-parse origin/main)"
merge_base_sha="$(git merge-base HEAD origin/main)"

if [[ "$merge_base_sha" != "$origin_main_sha" ]]; then
  echo "ERROR: HEAD does not include latest origin/main." >&2
  echo "origin/main: $origin_main_sha" >&2
  echo "merge-base : $merge_base_sha" >&2
  echo "Rebase or merge origin/main, then retry." >&2
  exit 6
fi

echo "[qa-safe-push] Pushing HEAD to origin/main..."
git push origin HEAD:main
echo "[qa-safe-push] Push complete."
