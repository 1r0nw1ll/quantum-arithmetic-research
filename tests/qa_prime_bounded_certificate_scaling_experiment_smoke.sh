#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_prime_bounded_certificate_scaling.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/experiments/qa_prime_bounded_certificate_scaling_experiment.py" \
  --endpoints 100,250 \
  --out "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["tested_endpoints"] == [100, 250]
assert payload["result"] == "PASS"
assert all(row["matches_prediction"] for row in payload["rows"])
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_prime_bounded_certificate_scaling_experiment_smoke: ok"
