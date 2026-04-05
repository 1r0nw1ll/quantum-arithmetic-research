#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SOURCE_REPO DEST_REPO" >&2
  exit 1
fi

source_repo="$1"
dest_repo="$2"

if [[ ! -d "${source_repo}" ]]; then
  echo "[clone-from-usb] missing source repo: ${source_repo}" >&2
  exit 1
fi

mkdir -p "${dest_repo}"

rsync -a --delete \
  --exclude '.venv/' \
  --exclude '.venv_player5/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude 'qa_lab/.venv/' \
  --exclude 'qa_lab/qa_venv/' \
  --exclude 'codex_on_QA/.venv/' \
  "${source_repo}/" "${dest_repo}/"

cat <<EOF
[clone-from-usb] cloned repo to ${dest_repo}
[clone-from-usb] next:
  cd "${dest_repo}"
  bash player5_fresh_metal_pack/bootstrap_player5.sh
EOF
