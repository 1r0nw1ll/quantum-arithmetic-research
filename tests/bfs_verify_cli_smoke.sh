#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_FILE="$(mktemp -t bfs_verify.XXXXXX.json)"
GRAPH_FILE="$(mktemp -t bfs_graph.XXXXXX.json)"
SEMI_JSON_FILE="$(mktemp -t bfs_verify_semiprime.XXXXXX.json)"

cleanup() {
  rm -f "$JSON_FILE" "$GRAPH_FILE" "$SEMI_JSON_FILE"
}
trap cleanup EXIT

python "$ROOT_DIR/bfs_verify.py" \
  --source 1 1 \
  --target 1 2 \
  --prime-target norm \
  --limit 2 \
  --emit-graph "$GRAPH_FILE" \
  > "$JSON_FILE"

python "$ROOT_DIR/bfs_verify.py" \
  --source 2 2 \
  --semiprime-target norm \
  --limit 2 \
  > "$SEMI_JSON_FILE"

python - "$JSON_FILE" "$GRAPH_FILE" "$SEMI_JSON_FILE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
graph = json.load(open(sys.argv[2], "r", encoding="utf-8"))
semi_payload = json.load(open(sys.argv[3], "r", encoding="utf-8"))

assert payload["target_query"]["reachable"] is True
assert payload["target_query"]["shortest_steps"] == 1
assert payload["prime_query"]["reachable_prime_targets"][0]["prime"] == 23
assert payload["prime_query"]["obstructed_prime_residues"] == [3, 7]

assert semi_payload["semiprime_query"]["reachable_semiprime_targets"][0]["semiprime"] == 4

assert graph["source"] == [1, 1]
assert graph["node_count"] >= 1
assert graph["edge_count"] >= 1
PY

echo "bfs_verify_cli_smoke: ok"
