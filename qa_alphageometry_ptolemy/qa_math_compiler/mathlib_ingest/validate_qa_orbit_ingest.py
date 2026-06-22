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
QAPARTITION_PATH = ROOT / "QAOrbitPartition.lean"
QAINVARIANCE_PATH = ROOT / "QAOrbitInvariance.lean"
QAFIBMATRIX_PATH = ROOT / "QAFibMatrix.lean"
QAFIBMATRIXGROUP_PATH = ROOT / "QAFibMatrixGroup.lean"

REGISTRY_SCHEMA_ID = "QA_ORBIT_REGISTRY.v1"
EXPECTED_ENTRY_COUNT = 32
EXPECTED_THEOREM_NAMES = {
    # QAOrbits.lean (5)
    "qa_cfgpythag",
    "qa_singularity_fixed",
    "qa_satellite_period_8",
    "qa_cosmos_period_24",
    "qa_t_period_divides_24",
    # QAOrbitPartition.lean (6)
    "qa_cosmos_card",
    "qa_orbit_partition",
    "qa_cosmos_period_exact",
    "qa_satellite_period_exact",
    "qa_singularity_unique",
    "qa_pisano_9_exact",
    # QAOrbitInvariance.lean (7)
    "qa_cosmos_invariant",
    "qa_satellite_invariant",
    "qa_cosmos_orbit1_card",
    "qa_cosmos_orbit12_disjoint",
    "qa_cosmos_suborbit_union",
    "qa_cosmos_step_injective",
    "qa_cosmos_reps_distinct",
    # QAFibMatrix.lean (7)
    "fib_mat_pow_24",
    "fib_mat_order_exact",
    "fib_mat_det",
    "fib_mat_det_ne_zero",
    "fib_mat_action",
    "fib_mat_iter",
    "fib_mat_pisano_9",
    # QAFibMatrixGroup.lean (7)
    "fib_mat_unit_pow_24",
    "fib_mat_unit_order_exact",
    "fib_mat_unit_pow_12_ne_one",
    "fib_mat_unit_pow_8_ne_one",
    "fib_mat_unit_orderOf",
    "fib_mat_zpowers_card",
    "fib_mat_zpowers_isCyclic",
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
                  + metrics.get("decide_count", 0) + metrics.get("native_decide_count", 0)
                  + metrics.get("simp_count", 0) + metrics.get("rw_count", 0)
                  + metrics.get("exact_count", 0))
    checks.append(_check("registry_metrics_tactic_sum",
                          tactic_sum == EXPECTED_ENTRY_COUNT,
                          f"tactic counts sum to {tactic_sum}, expected {EXPECTED_ENTRY_COUNT}"))

    return checks


# ---------------------------------------------------------------------------
# QAOrbits.lean source checks
# ---------------------------------------------------------------------------

def validate_lean_source() -> List[Dict[str, Any]]:
    checks = []

    # ── QAOrbits.lean ────────────────────────────────────────────────────────
    if not QALEAN_PATH.exists():
        return [_check("lean_source_exists", False, f"{QALEAN_PATH} not found")]

    checks.append(_check("lean_source_exists", True))

    src = QALEAN_PATH.read_text(encoding="utf-8")

    checks.append(_check("lean_imports_first",
                          src.startswith("import"),
                          "imports must be the first content in the file"))

    orbits_names = {
        "qa_cfgpythag", "qa_singularity_fixed", "qa_satellite_period_8",
        "qa_cosmos_period_24", "qa_t_period_divides_24",
    }
    for name in orbits_names:
        checks.append(_check(f"lean_defines_{name}",
                              f"theorem {name}" in src,
                              f"theorem {name} not found in QAOrbits.lean"))

    # ── QAOrbitPartition.lean ─────────────────────────────────────────────────
    if not QAPARTITION_PATH.exists():
        checks.append(_check("lean_partition_source_exists", False,
                              f"{QAPARTITION_PATH} not found"))
        return checks

    checks.append(_check("lean_partition_source_exists", True))

    psrc = QAPARTITION_PATH.read_text(encoding="utf-8")

    checks.append(_check("lean_partition_imports_first",
                          psrc.startswith("import"),
                          "imports must be the first content in QAOrbitPartition.lean"))

    partition_names = {
        "qa_cosmos_card", "qa_orbit_partition", "qa_cosmos_period_exact",
        "qa_satellite_period_exact", "qa_singularity_unique", "qa_pisano_9_exact",
    }
    for name in partition_names:
        checks.append(_check(f"lean_partition_defines_{name}",
                              f"theorem {name}" in psrc,
                              f"theorem {name} not found in QAOrbitPartition.lean"))

    # ── QAOrbitInvariance.lean ────────────────────────────────────────────────
    if not QAINVARIANCE_PATH.exists():
        checks.append(_check("lean_invariance_source_exists", False,
                              f"{QAINVARIANCE_PATH} not found"))
        return checks

    checks.append(_check("lean_invariance_source_exists", True))

    isrc = QAINVARIANCE_PATH.read_text(encoding="utf-8")

    checks.append(_check("lean_invariance_imports_first",
                          isrc.startswith("import"),
                          "imports must be the first content in QAOrbitInvariance.lean"))

    invariance_names = {
        "qa_cosmos_invariant", "qa_satellite_invariant",
        "qa_cosmos_orbit1_card", "qa_cosmos_orbit12_disjoint",
        "qa_cosmos_suborbit_union", "qa_cosmos_step_injective",
        "qa_cosmos_reps_distinct",
    }
    for name in invariance_names:
        checks.append(_check(f"lean_invariance_defines_{name}",
                              f"theorem {name}" in isrc,
                              f"theorem {name} not found in QAOrbitInvariance.lean"))

    # ── QAFibMatrix.lean ───────────────────────────────────────────────────────
    if not QAFIBMATRIX_PATH.exists():
        checks.append(_check("lean_fibmatrix_source_exists", False,
                              f"{QAFIBMATRIX_PATH} not found"))
        return checks

    checks.append(_check("lean_fibmatrix_source_exists", True))

    fsrc = QAFIBMATRIX_PATH.read_text(encoding="utf-8")

    checks.append(_check("lean_fibmatrix_imports_first",
                          fsrc.startswith("import"),
                          "imports must be the first content in QAFibMatrix.lean"))

    fibmatrix_names = {
        "fib_mat_pow_24", "fib_mat_order_exact", "fib_mat_det",
        "fib_mat_det_ne_zero", "fib_mat_action", "fib_mat_iter",
        "fib_mat_pisano_9",
    }
    for name in fibmatrix_names:
        checks.append(_check(f"lean_fibmatrix_defines_{name}",
                              f"theorem {name}" in fsrc,
                              f"theorem {name} not found in QAFibMatrix.lean"))

    # ── QAFibMatrixGroup.lean ─────────────────────────────────────────────────
    if not QAFIBMATRIXGROUP_PATH.exists():
        checks.append(_check("lean_fibmatrixgroup_source_exists", False,
                              f"{QAFIBMATRIXGROUP_PATH} not found"))
        return checks

    checks.append(_check("lean_fibmatrixgroup_source_exists", True))

    gsrc = QAFIBMATRIXGROUP_PATH.read_text(encoding="utf-8")

    checks.append(_check("lean_fibmatrixgroup_imports_first",
                          gsrc.startswith("import"),
                          "imports must be the first content in QAFibMatrixGroup.lean"))

    fibmatrixgroup_names = {
        "fib_mat_unit_pow_24", "fib_mat_unit_order_exact",
        "fib_mat_unit_pow_12_ne_one", "fib_mat_unit_pow_8_ne_one",
        "fib_mat_unit_orderOf", "fib_mat_zpowers_card",
        "fib_mat_zpowers_isCyclic",
    }
    for name in fibmatrixgroup_names:
        checks.append(_check(f"lean_fibmatrixgroup_defines_{name}",
                              f"theorem {name}" in gsrc,
                              f"theorem {name} not found in QAFibMatrixGroup.lean"))

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
