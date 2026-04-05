#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_prime_bounded_certificate_sweep.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/experiments/qa_prime_bounded_certificate_sweep.py" \
  --start 2 \
  --end 100 \
  --prime-caps 2,3,5,7 \
  --out "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["interval"] == {"start": 2, "end": 100}
assert payload["result"] == "PASS"
assert payload["minimal_pass_prime_max"] == 7
assert payload["runs"]
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_prime_bounded_certificate_sweep_smoke: ok"
