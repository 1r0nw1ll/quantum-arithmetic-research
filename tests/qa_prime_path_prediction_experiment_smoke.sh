#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_prime_path_prediction.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/experiments/qa_prime_path_prediction_experiment.py" \
  --start 2 \
  --end 60 \
  --out "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["interval"] == {"start": 2, "end": 60}
assert payload["result"] in {"PASS", "PARTIAL", "FAIL"}
assert payload["configurations"]
assert "prime_coverage_ratio" in payload["best_configuration"]
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_prime_path_prediction_experiment_smoke: ok"
