#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Pinned Math Compiler bootstrap currently supports macOS only." >&2
  exit 2
fi

brew install elan coq
brew install --cask isabelle

elan toolchain install leanprover/lean4:v4.31.0
elan default leanprover/lean4:v4.31.0

isabelle_archive="$(
  find "$(brew --cache)" -type f -name '*Isabelle2025-2_macos.tar.gz' -print -quit
)"
if [[ -z "$isabelle_archive" ]]; then
  echo "Pinned Isabelle2025-2 archive not found in Homebrew cache." >&2
  exit 3
fi
echo "8f187496e295f169952e944745af9e4ae00c9c1cd2ed4cadbcf7d898e444913e  $isabelle_archive" | shasum -a 256 -c -

lean --version
coqc --version
isabelle version

python3 qa_alphageometry_ptolemy/qa_math_compiler/verify_live_kernels.py
