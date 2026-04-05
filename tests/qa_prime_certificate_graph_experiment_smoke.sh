#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_prime_certificate_graph_experiment.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/experiments/qa_prime_certificate_graph_experiment.py" \
  --start 2 \
  --end 100 \
  --out "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["interval"] == {"start": 2, "end": 100}
assert payload["result"] == "PASS"
assert payload["summary"]["prime_terminal_exact"] is True
assert payload["summary"]["composite_positive_depth_exact"] is True
assert payload["summary"]["semiprime_depth_exact"] is True
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_prime_certificate_graph_experiment_smoke: ok"
