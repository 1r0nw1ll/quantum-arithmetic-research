#!/usr/bin/env python3
# noqa: DECL-1 (infrastructure — orbit-pack validator, not an empirical QA script)
"""
Validate the QA Orbit corpus: registry integrity + pack structural checks.

Usage:
  python validate_qa_orbit_ingest.py          — run self-test
  python validate_qa_orbit_ingest.py --ci     — exit 0 on PASS, 1 on FAIL
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
PACK_DIR = ROOT.parent / "qa_orbit_pack_v1"
REGISTRY_PATH = ROOT / "qa_orbit_registry.v1.json"
QALEAN_PATH = ROOT / "QAOrbits.lean"

REGISTRY_SCHEMA_ID = "QA_ORBIT_REGISTRY.v1"
EXPECTED_ENTRY_COUNT = 5
EXPECTED_THEOREM_NAMES = {
    "qa_cfgpythag",
    "qa_singularity_fixed",
    "qa_satellite_period_8",
    "qa_cosmos_period_24",
    "qa_t_period_divides_24",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _check(name: str, ok: bool, reason: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": name, "ok": ok}
    if not ok and reason:
        result["reason"] = reason
    return result


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------

def validate_registry() -> List[Dict[str, Any]]:
    checks = []

    if not REGISTRY_PATH.exists():
        return [_check("registry_exists", False, f"{REGISTRY_PATH} not found")]

    try:
        reg = _load_json(REGISTRY_PATH)
    except Exception as exc:
        return [_check("registry_parse", False, str(exc))]

    checks.append(_check("registry_schema_id", reg.get("schema_id") == REGISTRY_SCHEMA_ID,
                          f"got {reg.get('schema_id')!r}"))

    entries = reg.get("entries", [])
    checks.append(_check("registry_entry_count",
                          len(entries) == EXPECTED_ENTRY_COUNT,
                          f"expected {EXPECTED_ENTRY_COUNT}, got {len(entries)}"))

    seen_names = set()
    seen_ids = set()
    for entry in entries:
        name = entry.get("theorem_name", "")
        eid = entry.get("example_id", "")
        seen_names.add(name)
        seen_ids.add(eid)

    missing_names = EXPECTED_THEOREM_NAMES - seen_names
    checks.append(_check("registry_theorem_names_complete", not missing_names,
                          f"missing: {sorted(missing_names)}"))

    checks.append(_check("registry_example_ids_unique",
                          len(seen_ids) == len(entries),
                          "duplicate example_ids"))

    metrics = reg.get("metrics", {})
    declared = metrics.get("entry_count", -1)
    checks.append(_check("registry_metrics_entry_count",
                          declared == EXPECTED_ENTRY_COUNT,
                          f"declared {declared}, expected {EXPECTED_ENTRY_COUNT}"))

    tactic_sum = (metrics.get("ring_count", 0) + metrics.get("rfl_count", 0)
                  + metrics.get("decide_count", 0) + metrics.get("native_decide_count", 0))
    checks.append(_check("registry_metrics_tactic_sum",
                          tactic_sum == EXPECTED_ENTRY_COUNT,
                          f"tactic counts sum to {tactic_sum}, expected {EXPECTED_ENTRY_COUNT}"))

    return checks


# ---------------------------------------------------------------------------
# QAOrbits.lean source checks
# ---------------------------------------------------------------------------

def validate_lean_source() -> List[Dict[str, Any]]:
    checks = []

    if not QALEAN_PATH.exists():
        return [_check("lean_source_exists", False, f"{QALEAN_PATH} not found")]

    checks.append(_check("lean_source_exists", True))

    src = QALEAN_PATH.read_text(encoding="utf-8")

    # imports must be first lines
    checks.append(_check("lean_imports_first",
                          src.startswith("import"),
                          "imports must be the first content in the file"))

    for name in EXPECTED_THEOREM_NAMES:
        checks.append(_check(f"lean_defines_{name}",
                              f"theorem {name}" in src,
                              f"theorem {name} not found in QAOrbits.lean"))

    return checks


# ---------------------------------------------------------------------------
# Pack structural checks
# ---------------------------------------------------------------------------

def validate_pack() -> List[Dict[str, Any]]:
    checks = []

    if not PACK_DIR.exists():
        return [_check("pack_exists", False, f"{PACK_DIR} not found")]

    checks.append(_check("pack_exists", True))

    for fname in ["README.md", "index.json", "corpus.json"]:
        path = PACK_DIR / fname
        checks.append(_check(f"pack_{fname.replace('.', '_').replace('/', '_')}",
                              path.exists(), f"{path} not found"))

    if not (PACK_DIR / "index.json").exists():
        return checks

    try:
        index = _load_json(PACK_DIR / "index.json")
    except Exception as exc:
        checks.append(_check("pack_index_parse", False, str(exc)))
        return checks

    checks.append(_check("pack_index_schema_id",
                          index.get("schema_id") == "QA_MATH_COMPILER_DEMO_PACK_SCHEMA.v1",
                          f"got {index.get('schema_id')!r}"))
    checks.append(_check("pack_index_example_count",
                          index.get("example_count") == EXPECTED_ENTRY_COUNT,
                          f"expected {EXPECTED_ENTRY_COUNT}, got {index.get('example_count')}"))

    examples_dir = PACK_DIR / "examples"
    for example in index.get("examples", []):
        eid = example.get("id", "")
        ex_dir = examples_dir / eid
        for fname in ["claim.txt", "task.json", "trace.json", "replay.json", "pair.json", "README.md"]:
            path = ex_dir / fname
            checks.append(_check(f"pack_example_{eid}_{fname.replace('.', '_')}",
                                  path.exists(), f"{path} not found"))

    if (PACK_DIR / "corpus.json").exists():
        try:
            corpus = _load_json(PACK_DIR / "corpus.json")
            checks.append(_check("corpus_schema_id",
                                  corpus.get("schema_id") == "QA_CERTIFIED_PROOF_CORPUS_SCHEMA.v1",
                                  f"got {corpus.get('schema_id')!r}"))
            metrics = corpus.get("metrics", {})
            checks.append(_check("corpus_all_certified",
                                  metrics.get("certified_count") == EXPECTED_ENTRY_COUNT,
                                  f"certified_count={metrics.get('certified_count')}, expected {EXPECTED_ENTRY_COUNT}"))
            checks.append(_check("corpus_replay_rate_1",
                                  metrics.get("replay_success_rate") == 1.0,
                                  f"replay_success_rate={metrics.get('replay_success_rate')}"))
        except Exception as exc:
            checks.append(_check("corpus_parse", False, str(exc)))

    return checks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> Dict[str, Any]:
    all_checks: List[Dict[str, Any]] = []
    all_checks += validate_registry()
    all_checks += validate_lean_source()
    all_checks += validate_pack()

    ok = all(c["ok"] for c in all_checks)
    return {"ok": ok, "checks": all_checks}


if __name__ == "__main__":
    ci_mode = "--ci" in sys.argv
    result = self_test()
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        failures = [c for c in result["checks"] if not c["ok"]]
        print(f"\nFAIL: {len(failures)} check(s) failed", file=sys.stderr)
        for c in failures:
            print(f"  - {c['name']}: {c.get('reason', '')}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nPASS: {len(result['checks'])} checks", file=sys.stderr)
        sys.exit(0)
