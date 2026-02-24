#!/usr/bin/env python3
"""
validator.py

QA_LOCALITY_REGIME_SEP_CERT.v1 validator (Machine Tract).

Certifies the Locality Regime Separator — the theorem-level bridge between
the empirical cert families [77]/[78] and the variance-bias decomposition:

    err(r) ≈ V(r) + B(r)
    V(r) ∝ σ²/(2r+1)²     (decreasing — variance reduction)
    B(r) ∝ adj_4·r·‖Δμ‖²  (increasing — boundary contamination)

The cert classifies a scene as:
    DOMINANT:  ∃ r s.t. ΔOA(r) > 0  (variance wins for at least one radius)
    BOUNDARY:  ∀ r, ΔOA(r) ≤ 0      (boundary bias dominates all tested radii)

Gates:
- Gate 1: JSON schema validity
- Gate 2: Canonical SHA-256 digest integrity
- Gate 3: Recompute delta_oa consistency — verify max/min deltas in evidence block
- Gate 4: Regime declaration must be consistent with computed delta sign structure
           (DOMINANT requires any_delta_positive=True; BOUNDARY requires all_deltas_nonpositive=True)
- Gate 5: regime_evidence.regime_consistent must be True

Failure modes: REGIME_INCONSISTENT, EVIDENCE_MISMATCH, SCHEMA_INVALID, HASH_MISMATCH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"gate": self.gate, "status": self.status.value,
                "message": self.message, "details": self.details}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema
    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    copy = json.loads(_canonical_json_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_json_compact(copy))


def validate_cert(obj: Dict[str, Any], cert_dir: Optional[str] = None) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Schema valid"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema invalid: {e}"))
        return results

    # Gate 2 — Canonical hash integrity
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got = _compute_canonical_sha256(obj)
    if want == "0" * 64:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL,
                                  "canonical_sha256 is placeholder", {"got": got}))
        return results
    if want != got:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL,
                                  "canonical_sha256 mismatch", {"want": want, "got": got}))
        return results
    results.append(GateResult("gate_2_canonical_hash", GateStatus.PASS, "canonical_sha256 matches"))

    # Gate 3 — Recompute delta evidence from delta_oa_by_radius
    delta_oa = obj["delta_oa_by_radius"]
    deltas = list(delta_oa.values())
    if not deltas:
        results.append(GateResult("gate_3_delta_evidence", GateStatus.FAIL,
                                  "delta_oa_by_radius is empty"))
        return results

    computed_all_nonpos = all(d <= 0 for d in deltas)
    computed_any_pos = any(d > 0 for d in deltas)
    computed_max_pp = round(max(deltas) * 100, 4)
    computed_min_pp = round(min(deltas) * 100, 4)

    ev = obj["regime_evidence"]
    declared_all_nonpos = ev["all_deltas_nonpositive"]
    declared_any_pos = ev["any_delta_positive"]

    mismatches = []
    if declared_all_nonpos != computed_all_nonpos:
        mismatches.append(f"all_deltas_nonpositive: declared={declared_all_nonpos} computed={computed_all_nonpos}")
    if declared_any_pos != computed_any_pos:
        mismatches.append(f"any_delta_positive: declared={declared_any_pos} computed={computed_any_pos}")
    # Check max/min if present
    declared_max = ev.get("max_delta_pp")
    declared_min = ev.get("min_delta_pp")
    tol = 0.0001  # 0.0001 pp tolerance for rounding
    if declared_max is not None and abs(declared_max - computed_max_pp) > tol:
        mismatches.append(f"max_delta_pp: declared={declared_max} computed={computed_max_pp}")
    if declared_min is not None and abs(declared_min - computed_min_pp) > tol:
        mismatches.append(f"min_delta_pp: declared={declared_min} computed={computed_min_pp}")

    if mismatches:
        results.append(GateResult("gate_3_delta_evidence", GateStatus.FAIL,
                                  f"EVIDENCE_MISMATCH: {'; '.join(mismatches)}",
                                  {"computed_all_nonpos": computed_all_nonpos,
                                   "computed_any_pos": computed_any_pos,
                                   "computed_max_pp": computed_max_pp,
                                   "computed_min_pp": computed_min_pp}))
        return results
    results.append(GateResult("gate_3_delta_evidence", GateStatus.PASS,
                               f"Evidence consistent: all_nonpos={computed_all_nonpos}, "
                               f"any_pos={computed_any_pos}, "
                               f"max={computed_max_pp:+.2f}pp, min={computed_min_pp:+.2f}pp"))

    # Gate 4 — Regime declaration consistent with computed delta signs
    regime = obj["regime"]
    if regime == "DOMINANT":
        if not computed_any_pos:
            results.append(GateResult("gate_4_regime_declaration", GateStatus.FAIL,
                                      "REGIME_INCONSISTENT: declared DOMINANT but no ΔOA(r) > 0 at any tested radius",
                                      {"regime": regime, "computed_any_pos": computed_any_pos,
                                       "max_delta_pp": computed_max_pp}))
            return results
    elif regime == "BOUNDARY":
        if not computed_all_nonpos:
            results.append(GateResult("gate_4_regime_declaration", GateStatus.FAIL,
                                      "REGIME_INCONSISTENT: declared BOUNDARY but ΔOA(r) > 0 at some radius "
                                      "(this is a DOMINANT scene)",
                                      {"regime": regime, "computed_all_nonpos": computed_all_nonpos,
                                       "max_delta_pp": computed_max_pp}))
            return results

    n_radii = len(deltas)
    results.append(GateResult("gate_4_regime_declaration", GateStatus.PASS,
                               f"Regime={regime} consistent with delta signs across {n_radii} radius/radii",
                               {"regime": regime, "n_radii": n_radii,
                                "max_delta_pp": computed_max_pp, "adj_rate_4": obj["adj_rate_4"]}))

    # Gate 5 — regime_consistent flag must be True
    if not ev["regime_consistent"]:
        results.append(GateResult("gate_5_regime_consistent_flag", GateStatus.FAIL,
                                  "REGIME_INCONSISTENT: regime_evidence.regime_consistent=False",
                                  {"regime_consistent": False}))
        return results
    results.append(GateResult("gate_5_regime_consistent_flag", GateStatus.PASS,
                               "regime_consistent=True confirmed"))

    # Gate 6 — Adjacency witness (optional, v1.1+)
    aw = obj.get("adjacency_witness")
    if aw is not None:
        declared_adj = aw["adj_rate_4"]
        has_grid = aw.get("gt_label_grid") is not None
        has_path = aw.get("gt_mask_path") is not None

        if has_grid and has_path:
            results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                      "ADJ_WITNESS_MODE_AMBIGUOUS: both gt_label_grid and gt_mask_path present"))
            return results

        if not has_grid and not has_path:
            results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                      "ADJ_GRID_INVALID: adjacency_witness present but no grid or path"))
            return results

        if has_grid:
            # Mode A — inline grid
            grid = aw["gt_label_grid"]
            grid_canonical = _canonical_json_compact(grid)
            computed_sha = _sha256_hex(grid_canonical)
            declared_sha = aw.get("gt_label_sha256", "")
            if declared_sha != computed_sha:
                results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                          "ADJ_WITNESS_HASH_MISMATCH",
                                          {"declared_sha": declared_sha, "computed_sha": computed_sha}))
                return results
            # Recompute adj_rate_4
            cross, total = 0, 0
            for i, row in enumerate(grid):
                for j in range(len(row) - 1):
                    total += 1
                    if row[j] != row[j + 1]:
                        cross += 1
            for i in range(len(grid) - 1):
                for j in range(len(grid[i])):
                    total += 1
                    if grid[i][j] != grid[i + 1][j]:
                        cross += 1
            if total == 0:
                results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                          "ADJ_GRID_INVALID: empty grid"))
                return results
            computed_adj = cross / total
            mode_note = f"Mode A, grid {len(grid)}×{len(grid[0])}"

        else:
            # Mode B — .npy path
            import numpy as np
            rel_path = aw["gt_mask_path"]
            abs_path = os.path.join(cert_dir, rel_path) if cert_dir else rel_path
            if not os.path.exists(abs_path):
                results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                          f"ADJ_MASK_NOT_FOUND: {rel_path}",
                                          {"resolved": abs_path}))
                return results
            with open(abs_path, "rb") as f:
                raw = f.read()
            computed_sha = hashlib.sha256(raw).hexdigest()
            declared_sha = aw.get("gt_mask_sha256", "")
            if declared_sha != computed_sha:
                results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                          "ADJ_WITNESS_HASH_MISMATCH",
                                          {"declared_sha": declared_sha, "computed_sha": computed_sha}))
                return results
            arr = np.load(abs_path)
            H, W = arr.shape
            cross, total = 0, 0
            for i in range(H):
                for j in range(W - 1):
                    total += 1
                    if arr[i, j] != arr[i, j + 1]:
                        cross += 1
            for i in range(H - 1):
                for j in range(W):
                    total += 1
                    if arr[i, j] != arr[i + 1, j]:
                        cross += 1
            computed_adj = cross / total
            mode_note = f"Mode B, {H}×{W} mask"

        if abs(computed_adj - declared_adj) > 1e-6:
            results.append(GateResult("gate_6_adjacency_witness", GateStatus.FAIL,
                                      "ADJ_RATE_MISMATCH",
                                      {"declared_adj": declared_adj,
                                       "computed_adj": computed_adj,
                                       "diff": abs(computed_adj - declared_adj)}))
            return results
        results.append(GateResult("gate_6_adjacency_witness", GateStatus.PASS,
                                   f"adj_rate_4 verified: {computed_adj:.6f} ({mode_note})"))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")
    fixtures = [
        ("valid_salinas_dominant.json",          True,  None),
        ("valid_ksc_boundary.json",               True,  None),
        ("invalid_regime_inconsistent.json",      False, "gate_4_regime_declaration"),
        ("invalid_digest_mismatch.json",          False, "gate_2_canonical_hash"),
        # v1.1 adjacency witness fixtures
        ("valid_salinas_dominant_v1_1.json",     True,  None),
        ("valid_ksc_boundary_v1_1.json",          True,  None),
        ("invalid_adj_rate_mismatch.json",        False, "gate_6_adjacency_witness"),
        ("invalid_adj_hash_mismatch.json",        False, "gate_6_adjacency_witness"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({"fixture": name, "ok": None, "expected_ok": should_pass,
                             "failed_gates": [], "note": "MISSING"})
            ok = False
            continue
        obj = _load_json(path)
        res = validate_cert(obj, cert_dir=fx)
        passed = _report_ok(res)
        if should_pass != passed:
            ok = False
        fail_gates = [r.gate for r in res if r.status == GateStatus.FAIL]
        if (not should_pass) and expected_fail_gate and expected_fail_gate not in fail_gates:
            ok = False
        details.append({"fixture": name, "ok": passed, "expected_ok": should_pass,
                         "failed_gates": fail_gates})
    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_LOCALITY_REGIME_SEP_CERT.v1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_LOCALITY_REGIME_SEP_CERT.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(as_json=args.json)
    if not args.file:
        ap.print_help()
        return 2
    obj = _load_json(args.file)
    cert_dir = os.path.dirname(os.path.abspath(args.file))
    results = validate_cert(obj, cert_dir=cert_dir)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
