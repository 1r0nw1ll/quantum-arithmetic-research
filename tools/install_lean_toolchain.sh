#!/usr/bin/env bash
set -euo pipefail

toolchain="leanprover/lean4:v4.31.0"

if ! command -v elan >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh |
    sh -s -- -y --default-toolchain none
  export PATH="$HOME/.elan/bin:$PATH"
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
  printf '%s\n' "$HOME/.elan/bin" >> "$GITHUB_PATH"
fi

elan toolchain install "$toolchain"
elan default "$toolchain"
lean --version
