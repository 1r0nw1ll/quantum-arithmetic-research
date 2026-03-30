#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${QA_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  qa_discovery_pipeline/run_overnight.sh --plan <plan.json> [options]

Options:
  --out-root <dir>            Root directory for outputs (default: artifacts/discovery_runs)
  --label <name>              Output label (default: plan basename)
  --allow-fail                Pass --allow_fail to ci_check.py
  --episode-timeout-s <secs>  Forwarded to run_batch.py (default: 30; 0 disables)
  --toolchain-id <id>         Forwarded to run_batch.py (default: lean4.12.0)
  --prev-bundle-hash <hash>   Forwarded to run_batch.py (default: 64 zeros)
  --created-utc <ts>          Forwarded to run_batch.py (default: now)
  --summarize-harmonic        Run qa_harmonic_obstruction/summarize_sweep.py after CI

Run from anywhere; the script will cd into qa_alphageometry_ptolemy/.
EOF
}

PLAN=""
OUT_ROOT="artifacts/discovery_runs"
LABEL=""
ALLOW_FAIL=0
EPISODE_TIMEOUT_S="30"
TOOLCHAIN_ID="lean4.12.0"
PREV_BUNDLE_HASH="$(python3 - <<'PY'
print("0" * 64)
PY
)"
CREATED_UTC=""
SUMMARIZE_HARMONIC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      PLAN="${2:-}"; shift 2;;
    --out-root)
      OUT_ROOT="${2:-}"; shift 2;;
    --label)
      LABEL="${2:-}"; shift 2;;
    --allow-fail)
      ALLOW_FAIL=1; shift 1;;
    --episode-timeout-s)
      EPISODE_TIMEOUT_S="${2:-}"; shift 2;;
    --toolchain-id)
      TOOLCHAIN_ID="${2:-}"; shift 2;;
    --prev-bundle-hash)
      PREV_BUNDLE_HASH="${2:-}"; shift 2;;
    --created-utc)
      CREATED_UTC="${2:-}"; shift 2;;
    --summarize-harmonic)
      SUMMARIZE_HARMONIC=1; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2;;
  esac
done

if [[ -z "${PLAN}" ]]; then
  echo "Missing --plan" >&2
  usage
  exit 2
fi

if [[ -z "${LABEL}" ]]; then
  base="$(basename "${PLAN}")"
  LABEL="${base%.json}"
fi

ts="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/${LABEL}_${ts}"
mkdir -p "${OUT_DIR}"

echo "QA_ROOT=${QA_ROOT}"
echo "PLAN=${PLAN}"
echo "OUT_DIR=${OUT_DIR}"

run_batch_args=(--plan "${PLAN}" --out_dir "${OUT_DIR}" --toolchain_id "${TOOLCHAIN_ID}" --episode-timeout-s "${EPISODE_TIMEOUT_S}" --prev_bundle_hash "${PREV_BUNDLE_HASH}")
if [[ -n "${CREATED_UTC}" ]]; then
  run_batch_args+=(--created_utc "${CREATED_UTC}")
fi

echo "Running: python3 qa_discovery_pipeline/run_batch.py ${run_batch_args[*]}"
PYTHONUNBUFFERED=1 python3 qa_discovery_pipeline/run_batch.py "${run_batch_args[@]}" 2>&1 | tee "${OUT_DIR}/run_batch.log"

ci_args=(--out_dir "${OUT_DIR}")
if [[ "${ALLOW_FAIL}" -eq 1 ]]; then
  ci_args+=(--allow_fail)
fi

echo "Running: python3 qa_discovery_pipeline/ci_check.py ${ci_args[*]}"
PYTHONUNBUFFERED=1 python3 qa_discovery_pipeline/ci_check.py "${ci_args[@]}" 2>&1 | tee "${OUT_DIR}/ci_check.log"

if [[ "${SUMMARIZE_HARMONIC}" -eq 1 ]]; then
  echo "Running: python3 qa_harmonic_obstruction/summarize_sweep.py --out_dir ${OUT_DIR}"
  PYTHONUNBUFFERED=1 python3 qa_harmonic_obstruction/summarize_sweep.py --out_dir "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/summarize.log"
fi

echo "DONE out_dir=${OUT_DIR}"

