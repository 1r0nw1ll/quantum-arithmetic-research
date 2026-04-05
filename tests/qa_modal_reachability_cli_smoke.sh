#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_FILE="$(mktemp -t qa_modal_reachability_spec.XXXXXX.yaml)"
JSON_FILE="$(mktemp -t qa_modal_reachability_json.XXXXXX)"

cleanup() {
  rm -f "$SPEC_FILE" "$JSON_FILE"
}
trap cleanup EXIT

cat > "$SPEC_FILE" <<'YAML'
version: "0.1"
states:
  Source:
    id: S_SRC
    capabilities: {}
  Target:
    id: S_TGT
    capabilities: {}
generators:
  may_fail_bridge:
    id: gen_may_fail_bridge
    domain: Source
    codomain: Target
    may_fail:
      - PHYSICS_MISMATCH
failures:
  PHYSICS_MISMATCH:
    description: May-fail edge
    terminal: true
  NON_IDENTIFIABLE:
    description: Projection only
    terminal: true
  GENERATOR_INSUFFICIENT:
    description: Missing caps
    terminal: true
  UNREACHABLE:
    description: No path
    terminal: true
certificates:
  RETURN_CONSTRUCTED:
    fields:
      source_state: string
      target_state: string
      path:
        type: list
        elements: generator_id
      preconditions_met: list
      invariants_preserved: list
      error_bounds:
        type: numeric
      notes: optional
  CYCLE_IMPOSSIBLE:
    fields:
      attempted_goal: string
      required_missing_information: list
      fail_type: failure_id
      invariant_difference: list
      notes: optional
YAML

python "$ROOT_DIR/qa_modal_reachability.py" \
  --spec "$SPEC_FILE" \
  --source S_SRC \
  --target S_TGT \
  > "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert data.get("target_state") == "S_TGT"
assert "path" in data
PY

python "$ROOT_DIR/qa_modal_reachability.py" \
  --spec "$SPEC_FILE" \
  --source S_SRC \
  --target S_TGT \
  --avoid-may-fail \
  > "$JSON_FILE"

python - "$JSON_FILE" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert data.get("fail_type") == "UNREACHABLE"
PY

echo "qa_modal_reachability_cli_smoke: ok"
