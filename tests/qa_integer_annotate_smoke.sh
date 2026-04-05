#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t qa_integer_annotate.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/tools/qa_integer_annotate.py" --n 221 > "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["n"] == 221
assert payload["classification"]["is_semiprime"] is True
assert payload["classification"]["is_prime"] is False
assert payload["factorization"]["prime_powers"] == [{"p": 13, "e": 1}, {"p": 17, "e": 1}]
assert payload["factorization"]["Omega"] == 2
assert payload["residues"]["mod_24"] == 5
assert payload["qa_overlay"]["mod_24"]["prime_residue_candidate"] is True
assert len(payload["canonical_hash"]) == 64
PY

echo "qa_integer_annotate_smoke: ok"
