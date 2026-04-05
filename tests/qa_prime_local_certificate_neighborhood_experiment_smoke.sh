#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_prime_local_certificate_neighborhood.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/experiments/qa_prime_local_certificate_neighborhood_experiment.py" \
  --start 2 \
  --end 100 \
  --out "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["interval"] == {"start": 2, "end": 100}
assert payload["result"] == "PASS"
assert payload["summary"]["full_horizon_exact_fraction_overall"] == 1.0
assert payload["summary"]["first_radius_with_composite_advantage"] is not None
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_prime_local_certificate_neighborhood_experiment_smoke: ok"
