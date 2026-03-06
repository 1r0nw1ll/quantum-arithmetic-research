#!/usr/bin/env python3
"""
validator.py

QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1 validator (Machine tract).

This cert family formalizes the minimal algebraic presentation of QA failure
structure as a finite join-semilattice with monotone associative composition.

Gates:
  1) Schema shape / type checks
  2) Poset laws for <= (reflexive, antisymmetric, transitive)
  3) Join-semilattice laws (comm, assoc, idem) + join is LUB w.r.t <=
  4) Composition laws: associativity + monotonicity + failure propagation law
  5) invariant_diff_map claim binding + rollup hash

CLI:
  python validator.py <cert.json>
  python validator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class Diff:
    gate: int
    fail_type: str
    path: str
    reason: str


@dataclass
class GateResult:
    gate_id: int
    status: GateStatus
    message: str
    diffs: List[Diff] = field(default_factory=list)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _pass(gate_id: int, message: str) -> GateResult:
    return GateResult(gate_id, GateStatus.PASS, message, [])


def _fail(gate_id: int, fail_type: str, path: str, reason: str) -> GateResult:
    d = Diff(gate_id, fail_type, path, reason)
    return GateResult(gate_id, GateStatus.FAIL, f"{fail_type} @ {path} -- {reason}", [d])


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_from_pairs(pairs: List[Dict[str, str]]) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    for row in pairs:
        out.add((row["a"], row["b"]))
    return out


def _table_to_map_2(table: List[Dict[str, str]], key_out: str) -> Dict[Tuple[str, str], str]:
    out: Dict[Tuple[str, str], str] = {}
    for row in table:
        out[(row["a"], row["b"])] = row[key_out]
    return out


def _gate1_schema(cert: Any) -> GateResult:
    if not isinstance(cert, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", ".", "certificate must be a JSON object")

    required = [
        "schema_version",
        "cert_id",
        "created_utc",
        "carrier",
        "leq",
        "join_table",
        "compose_table",
        "unit",
        "result",
    ]
    for k in required:
        if k not in cert:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", k, f"required field '{k}' missing")

    if cert.get("schema_version") != "QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1":
        return _fail(
            1,
            "SCHEMA_VERSION_MISMATCH",
            "schema_version",
            f"expected 'QA_FAILURE_ALGEBRA_STRUCTURE_CERT.v1', got {cert.get('schema_version')!r}",
        )

    carrier = cert.get("carrier")
    if not (isinstance(carrier, list) and len(carrier) >= 2 and all(isinstance(x, str) and x for x in carrier)):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "carrier", "must be array[str] length>=2")
    if len(set(carrier)) != len(carrier):
        return _fail(1, "SCHEMA_VALUE_INVALID", "carrier", "must be unique items")

    if not isinstance(cert.get("unit"), str) or not cert["unit"]:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "unit", "must be non-empty string")
    if cert["unit"] not in set(carrier):
        return _fail(1, "SCHEMA_VALUE_INVALID", "unit", "unit must be in carrier")

    for fname in ("leq", "join_table", "compose_table"):
        if not isinstance(cert.get(fname), list) or len(cert[fname]) == 0:
            return _fail(1, "SCHEMA_TYPE_MISMATCH", fname, "must be non-empty array")

    carrier_set = set(carrier)

    for i, row in enumerate(cert["leq"]):
        if not (isinstance(row, dict) and set(row.keys()) == {"a", "b"}):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"leq[{i}]", "must be object with keys {a,b}")
        if row["a"] not in carrier_set or row["b"] not in carrier_set:
            return _fail(1, "SCHEMA_VALUE_INVALID", f"leq[{i}]", "a,b must be elements of carrier")

    for i, row in enumerate(cert["join_table"]):
        if not (isinstance(row, dict) and set(row.keys()) == {"a", "b", "join"}):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"join_table[{i}]", "must be object with keys {a,b,join}")
        if row["a"] not in carrier_set or row["b"] not in carrier_set or row["join"] not in carrier_set:
            return _fail(1, "SCHEMA_VALUE_INVALID", f"join_table[{i}]", "a,b,join must be elements of carrier")

    for i, row in enumerate(cert["compose_table"]):
        if not (isinstance(row, dict) and set(row.keys()) == {"a", "b", "comp"}):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"compose_table[{i}]", "must be object with keys {a,b,comp}")
        if row["a"] not in carrier_set or row["b"] not in carrier_set or row["comp"] not in carrier_set:
            return _fail(1, "SCHEMA_VALUE_INVALID", f"compose_table[{i}]", "a,b,comp must be elements of carrier")

    res = cert.get("result")
    if not isinstance(res, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result", "must be object")
    for k in ("ok", "violations", "invariant_diff_map"):
        if k not in res:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"result.{k}", "missing")
    if not isinstance(res["ok"], bool):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.ok", "must be boolean")
    if not isinstance(res["violations"], list):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.violations", "must be array")

    idm = res["invariant_diff_map"]
    if not isinstance(idm, dict) or "entries" not in idm or "rollup_sha256" not in idm:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map", "must have entries + rollup_sha256")
    if not isinstance(idm["entries"], list):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map.entries", "must be array")
    if not (isinstance(idm["rollup_sha256"], str) and len(idm["rollup_sha256"]) == 64):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map.rollup_sha256", "must be 64-hex string")

    return _pass(1, "schema shape valid")


def _closure_leq(carrier: List[str], leq_pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Compute reflexive-transitive closure over finite carrier."""
    idx = {x: i for i, x in enumerate(carrier)}
    n = len(carrier)
    reach = [[False] * n for _ in range(n)]

    for a in carrier:
        reach[idx[a]][idx[a]] = True
    for (a, b) in leq_pairs:
        reach[idx[a]][idx[b]] = True

    for k in range(n):
        for i in range(n):
            if not reach[i][k]:
                continue
            for j in range(n):
                if reach[k][j]:
                    reach[i][j] = True

    out: Set[Tuple[str, str]] = set()
    for i, a in enumerate(carrier):
        for j, b in enumerate(carrier):
            if reach[i][j]:
                out.add((a, b))
    return out


def _gate2_poset(cert: Dict[str, Any]) -> GateResult:
    carrier: List[str] = cert["carrier"]
    leq_pairs = _set_from_pairs(cert["leq"])
    clo = _closure_leq(carrier, leq_pairs)

    for a in carrier:
        for b in carrier:
            if a == b:
                continue
            if (a, b) in clo and (b, a) in clo:
                return _fail(
                    2,
                    "POSET_ANTISYMMETRY_VIOLATION",
                    "leq",
                    f"found {a}<= {b} and {b}<= {a} with a!=b",
                )

    return _pass(2, "poset laws verified on reflexive-transitive closure")


def _join(a: str, b: str, join_map: Dict[Tuple[str, str], str]) -> Optional[str]:
    if (a, b) in join_map:
        return join_map[(a, b)]
    if (b, a) in join_map:
        return join_map[(b, a)]
    return None


def _gate3_join_semilattice(cert: Dict[str, Any]) -> GateResult:
    carrier: List[str] = cert["carrier"]
    leq_pairs = _set_from_pairs(cert["leq"])
    leq_clo = _closure_leq(carrier, leq_pairs)
    join_map = _table_to_map_2(cert["join_table"], "join")

    def leq(x: str, y: str) -> bool:
        return (x, y) in leq_clo

    for a in carrier:
        for b in carrier:
            if _join(a, b, join_map) is None:
                return _fail(3, "JOIN_TABLE_INCOMPLETE", "join_table", f"missing join for pair ({a},{b})")

    for a in carrier:
        j = _join(a, a, join_map)
        if j != a:
            return _fail(3, "JOIN_IDEMPOTENCE_VIOLATION", "join_table", f"expected join({a},{a})={a}, got {j}")

    for a in carrier:
        for b in carrier:
            jab = _join(a, b, join_map)
            jba = _join(b, a, join_map)
            if jab != jba:
                return _fail(3, "JOIN_COMMUTATIVITY_VIOLATION", "join_table", f"join({a},{b})={jab} != join({b},{a})={jba}")

    for a in carrier:
        for b in carrier:
            for c in carrier:
                left = _join(_join(a, b, join_map), c, join_map)
                right = _join(a, _join(b, c, join_map), join_map)
                if left != right:
                    return _fail(
                        3,
                        "JOIN_ASSOCIATIVITY_VIOLATION",
                        "join_table",
                        f"join(join({a},{b}),{c})={left} != join({a},join({b},{c}))={right}",
                    )

    for a in carrier:
        for b in carrier:
            j = _join(a, b, join_map)
            if not leq(a, j) or not leq(b, j):
                return _fail(3, "JOIN_UPPER_BOUND_VIOLATION", "join_table", f"not an upper bound: a={a}, b={b}, join={j}")
            for u in carrier:
                if leq(a, u) and leq(b, u) and not leq(j, u):
                    return _fail(
                        3,
                        "JOIN_LEAST_UPPER_BOUND_VIOLATION",
                        "join_table",
                        f"join({a},{b})={j} not <= upper bound u={u}",
                    )

    return _pass(3, "join-semilattice laws + LUB verified")


def _gate4_composition(cert: Dict[str, Any]) -> GateResult:
    carrier: List[str] = cert["carrier"]
    leq_pairs = _set_from_pairs(cert["leq"])
    leq_clo = _closure_leq(carrier, leq_pairs)
    join_map = _table_to_map_2(cert["join_table"], "join")
    comp_map = _table_to_map_2(cert["compose_table"], "comp")

    def leq(x: str, y: str) -> bool:
        return (x, y) in leq_clo

    def join(a: str, b: str) -> str:
        j = _join(a, b, join_map)
        assert j is not None
        return j

    def comp(a: str, b: str) -> Optional[str]:
        return comp_map.get((a, b))

    for a in carrier:
        for b in carrier:
            if comp(a, b) is None:
                return _fail(4, "COMPOSE_TABLE_INCOMPLETE", "compose_table", f"missing comp for pair ({a},{b})")

    for a in carrier:
        for b in carrier:
            for c in carrier:
                left = comp(comp(a, b), c)
                right = comp(a, comp(b, c))
                if left != right:
                    return _fail(
                        4,
                        "COMPOSE_ASSOCIATIVITY_VIOLATION",
                        "compose_table",
                        f"({a}∘{b})∘{c}={left} != {a}∘({b}∘{c})={right}",
                    )

    for a1 in carrier:
        for a2 in carrier:
            if not leq(a1, a2):
                continue
            for b in carrier:
                if not leq(comp(a1, b), comp(a2, b)):
                    return _fail(
                        4,
                        "COMPOSE_MONOTONE_LEFT_VIOLATION",
                        "compose_table",
                        f"{a1}<= {a2} but {a1}∘{b}={comp(a1, b)} not <= {a2}∘{b}={comp(a2, b)}",
                    )

    for b1 in carrier:
        for b2 in carrier:
            if not leq(b1, b2):
                continue
            for a in carrier:
                if not leq(comp(a, b1), comp(a, b2)):
                    return _fail(
                        4,
                        "COMPOSE_MONOTONE_RIGHT_VIOLATION",
                        "compose_table",
                        f"{b1}<= {b2} but {a}∘{b1}={comp(a, b1)} not <= {a}∘{b2}={comp(a, b2)}",
                    )

    for a in carrier:
        for b in carrier:
            if comp(a, b) != join(a, b):
                return _fail(
                    4,
                    "FAILURE_PROPAGATION_LAW_VIOLATION",
                    "compose_table",
                    f"expected comp({a},{b})==join({a},{b})=={join(a, b)}, got {comp(a, b)}",
                )

    u = cert["unit"]
    for a in carrier:
        if comp(u, a) != a:
            return _fail(4, "COMPOSE_UNIT_LEFT_VIOLATION", "unit", f"expected unit∘{a}={a}, got {comp(u, a)}")
        if comp(a, u) != a:
            return _fail(4, "COMPOSE_UNIT_RIGHT_VIOLATION", "unit", f"expected {a}∘unit={a}, got {comp(a, u)}")

    return _pass(4, "composition laws verified (assoc + monotone + propagation + unit)")


def _gate5_claim_binding(cert: Dict[str, Any], computed_diffs: List[Diff]) -> GateResult:
    computed_entries = [
        {"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason}
        for d in computed_diffs
    ]
    claimed_entries = cert["result"]["invariant_diff_map"]["entries"]

    computed_entries_sorted = sorted(computed_entries, key=lambda x: (x["gate"], x["fail_type"], x["path"], x["reason"]))
    claimed_entries_sorted = sorted(claimed_entries, key=lambda x: (x["gate"], x["fail_type"], x["path"], x["reason"]))

    if computed_entries_sorted != claimed_entries_sorted:
        return _fail(
            5,
            "INVARIANT_DIFF_MAP_CLAIM_MISMATCH",
            "result.invariant_diff_map.entries",
            "claimed entries do not match recomputed violations",
        )

    rollup = _sha256_hex(_canonical_json(computed_entries_sorted))
    claimed_rollup = cert["result"]["invariant_diff_map"]["rollup_sha256"]
    if rollup != claimed_rollup:
        return _fail(
            5,
            "INVARIANT_DIFF_MAP_ROLLUP_MISMATCH",
            "result.invariant_diff_map.rollup_sha256",
            f"expected {rollup}, got {claimed_rollup}",
        )

    return _pass(5, f"invariant_diff_map claim verified (entries={len(computed_entries_sorted)})")


def _finalize(gates: List[GateResult], diffs: List[Diff]) -> Dict[str, Any]:
    failures_json = [{"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in diffs]
    failures_sorted = sorted(failures_json, key=lambda x: (x["gate"], x["fail_type"], x["path"], x["reason"]))

    ok = all(g.status == GateStatus.PASS for g in gates)
    return {
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "failures": failures_json,
        "gates": [{"gate_id": g.gate_id, "status": g.status.value, "message": g.message} for g in gates],
        "invariant_diff_map": {
            "entries": failures_sorted,
            "rollup_sha256": _sha256_hex(_canonical_json(failures_sorted)),
        },
    }


def validate_cert(cert: Any) -> Dict[str, Any]:
    gates: List[GateResult] = []
    failures: List[Diff] = []

    g1 = _gate1_schema(cert)
    gates.append(g1)
    failures.extend(g1.diffs)
    if g1.status == GateStatus.FAIL:
        return _finalize(gates, failures)

    g2 = _gate2_poset(cert)
    gates.append(g2)
    failures.extend(g2.diffs)

    g3 = _gate3_join_semilattice(cert)
    gates.append(g3)
    failures.extend(g3.diffs)

    g4 = _gate4_composition(cert)
    gates.append(g4)
    failures.extend(g4.diffs)

    g5 = _gate5_claim_binding(cert, failures)
    gates.append(g5)
    failures.extend(g5.diffs)

    return _finalize(gates, failures)


def _self_test() -> bool:
    base = os.path.dirname(os.path.abspath(__file__))
    checks = [
        ("PASS_tiny.json", True),
        ("invalid_compose_associativity_violation.json", False),
    ]
    all_ok = True
    for name, expected_ok in checks:
        fixture = os.path.join(base, "fixtures", name)
        cert = _load_json(fixture)
        result = validate_cert(cert)
        got = bool(result["ok"])
        if got != expected_ok:
            all_ok = False
            print(f"[FAIL] self-test {name}: got {got} (expected {expected_ok})")
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"[PASS] self-test {name}: got {got} (expected {expected_ok})")
    if not all_ok:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cert_path", nargs="?", help="path to cert JSON")
    ap.add_argument("--self-test", action="store_true", help="run built-in self-test")
    args = ap.parse_args()

    if args.self_test:
        return 0 if _self_test() else 1

    if not args.cert_path:
        ap.print_help()
        return 2

    cert = _load_json(args.cert_path)
    out = validate_cert(cert)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
