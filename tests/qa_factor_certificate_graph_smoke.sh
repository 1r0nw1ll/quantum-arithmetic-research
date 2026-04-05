#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_factor_certificate_graph.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/tools/qa_factor_certificate_graph.py" --start 2 --end 30 --focus 12 > "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["summary"]["prime_count"] == 10
assert payload["summary"]["semiprime_count"] == 10
assert payload["focus"]["n"] == 12
assert payload["focus"]["decomposition_depth_to_prime_terminal"] == 1
assert payload["focus"]["smallest_prime_chain"] == [12, 6, 3]
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_factor_certificate_graph_smoke: ok"
