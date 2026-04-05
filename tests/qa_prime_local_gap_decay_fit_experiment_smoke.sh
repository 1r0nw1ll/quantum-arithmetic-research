#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_prime_local_gap_decay_fit.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/experiments/qa_prime_local_gap_decay_fit_experiment.py" \
  --endpoints 100,250,500 \
  --out "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["tested_endpoints"] == [100, 250, 500]
assert payload["result"] == "PASS"
assert payload["best_generic_fit"] in {"half_plus_c_over_log_n", "affine_in_inv_log_n"}
assert payload["fits"]["structural_exact_formula"]["rmse"] < 1e-12
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_prime_local_gap_decay_fit_experiment_smoke: ok"
