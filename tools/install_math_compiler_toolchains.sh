#!/usr/bin/env bash
set -euo pipefail

report_failure() {
  local exit_code=$?
  local line_number=$1
  local command=$2
  printf '::error file=%s,line=%s::Installer command failed (exit %s): %s\n' \
    "${BASH_SOURCE[0]}" "$line_number" "$exit_code" "$command" >&2
  exit "$exit_code"
}
trap 'report_failure "$LINENO" "$BASH_COMMAND"' ERR

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Pinned Math Compiler bootstrap currently supports macOS only." >&2
  exit 2
fi

lean_toolchain="leanprover/lean4:v4.31.0"
rocq_version="9.1.1"
isabelle_version="Isabelle2025-2"
isabelle_sha256="8f187496e295f169952e944745af9e4ae00c9c1cd2ed4cadbcf7d898e444913e"

# Use canonical Homebrew names. `coq` is only an old name for `rocq`, and
# relying on that redirect has failed on clean runners.
brew install elan-init
brew install rocq
brew fetch --cask isabelle
brew install --cask isabelle

elan toolchain install "$lean_toolchain"
elan default "$lean_toolchain"

isabelle_archive="$(
  find "$(brew --cache)" -type f \
    -name "*${isabelle_version}_macos.tar.gz" \
    -print -quit
)"
if [[ -z "$isabelle_archive" ]]; then
  echo "Pinned ${isabelle_version} archive not found in Homebrew cache." >&2
  exit 3
fi
echo "${isabelle_sha256}  ${isabelle_archive}" | shasum -a 256 -c -

lean_version="$(lean --version)"
rocq_version_output="$(rocq --version)"
isabelle_version_output="$(isabelle version)"

grep -F "Lean (version 4.31.0," <<<"$lean_version"
grep -F "The Rocq Prover, version ${rocq_version}" <<<"$rocq_version_output"
grep -Fx "$isabelle_version" <<<"$isabelle_version_output"

printf '%s\n' "$lean_version"
printf '%s\n' "$rocq_version_output"
printf '%s\n' "$isabelle_version_output"

python3 qa_alphageometry_ptolemy/qa_math_compiler/verify_live_kernels.py
python3 qa_alphageometry_ptolemy/qa_math_compiler/compile_kernel_traces.py
