#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_interval_stats.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/tools/qa_interval_stats.py" --start 1 --end 30 > "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["range"] == {"start": 1, "end": 30}
assert payload["counts"]["primes"] == 10
assert payload["counts"]["semiprimes"] == 10
assert payload["counts"]["prime_powers"] == 6
assert payload["prime_counting"]["pi_interval"] == 10
assert payload["constellations"]["twin_prime_pairs"] == [[3, 5], [5, 7], [11, 13], [17, 19]]
assert payload["qa_overlay"]["mod_24"]["obstructed_prime_residues"] == [3, 7]
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_interval_stats_smoke: ok"
