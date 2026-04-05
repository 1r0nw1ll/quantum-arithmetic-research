#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 KEY_FILE [HOME_DIR]" >&2
  exit 1
fi

key_file="$1"
home_dir="${2:-$HOME}"
target="${home_dir}/.open_brain_mcp_key"

if [[ ! -f "${key_file}" ]]; then
  echo "[open-brain-key] missing key file: ${key_file}" >&2
  exit 1
fi

mkdir -p "${home_dir}"
cp "${key_file}" "${target}"
chmod 600 "${target}"

echo "[open-brain-key] installed ${target}"
