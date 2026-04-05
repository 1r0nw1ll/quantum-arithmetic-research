#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
venv_dir="${repo_root}/.venv_player5"

require_path() {
  local rel="$1"
  if [[ ! -e "${repo_root}/${rel}" ]]; then
    echo "[player5-bootstrap] missing required path: ${rel}" >&2
    exit 1
  fi
}

require_path "AGENTS.md"
require_path "MEMORY.md"
require_path "CONSTITUTION.md"
require_path "docs/specs/PROJECT_SPEC.md"
require_path "qa_lab/open_brain_bootstrap.py"

echo "[player5-bootstrap] repo root: ${repo_root}"

if [[ ! -d "${venv_dir}" ]]; then
  echo "[player5-bootstrap] creating ${venv_dir}"
  python3 -m venv "${venv_dir}"
else
  echo "[player5-bootstrap] using existing ${venv_dir}"
fi

echo "[player5-bootstrap] key docs:"
echo "  - AGENTS.md"
echo "  - MEMORY.md"
echo "  - CONSTITUTION.md"
echo "  - docs/specs/PROJECT_SPEC.md"
echo "  - docs/specs/VISION.md"
echo "  - CLAUDE.md"

echo "[player5-bootstrap] verifying Open Brain connectivity"
(
  cd "${repo_root}"
  python3 qa_lab/open_brain_bootstrap.py --limit 5 --since-days 14
)

cat <<EOF
[player5-bootstrap] next commands:
  cd "${repo_root}"
  source .venv_player5/bin/activate
  python3 qa_alphageometry_ptolemy/qa_meta_validator.py
EOF
