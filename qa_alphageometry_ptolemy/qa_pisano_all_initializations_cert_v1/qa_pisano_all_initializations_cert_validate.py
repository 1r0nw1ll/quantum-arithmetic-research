#!/usr/bin/env python3
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pisano orbit partition: "
    "sigma(b,e)=(e,((b+e-1)%m)+1); orbit periods are integer path counts; "
    "counts: Cosmos=72 period-24, Satellite=8 period-8, Singularity=1 period-1, total=81=9^2; "
    "swap symmetry: orbit family on (b,e) = orbit family on (e,b); negation parity: orbit family on (b,e) = orbit family on (neg(b),neg(e))); "
    "Theorem NT: 'pi(9)=24' is an integer circuit count; no float state, no continuous observer in QA layer"
)
"""QA Pisano All-Initializations Cert [499] validator.

Cert claim: QA three-orbit partition of {1,...,9}^2 (Cosmos 72 + Satellite 8 +
Singularity 1 = 81 total) is structurally equivalent to Pudelko's complete
Pisano period classification of all m^2 = 81 initial pairs in (Z/9Z)^2.

The QA step function on {1,...,m}^2 (A1 no-zero compliant):
    sigma(b, e) = (e, ((b + e - 1) % m) + 1)

This is the Fibonacci shift T(x,y)=(y,x+y) mod m conjugated by the QA offset
bijection phi(b) = b mod m  (phi^{-1}(0) = m, phi^{-1}(x) = x for x>0).
Period of (b,e) under sigma equals period of (phi(b), phi(e)) under T mod m.

Orbit families (m=9, prime-power p^k = 3^2):
  Cosmos     72 pairs  period 24  content-ideal class J=0: min(v_3(b),v_3(e))=0
  Satellite   8 pairs  period  8  content-ideal class J=1: min(v_3(b),v_3(e))=1
  Singularity 1 pair   period  1  content-ideal class J=2: min(v_3(b),v_3(e))=2

These are the three distinct Pisano periods achievable from (Z/9Z)^2. Pudelko
(arXiv:2510.24882) independently classifies ALL m^2=81 initial pairs and finds
the same three periods with the same counts (Results 1.1-1.5).

Checks:
  QAP_1  Exhaustive partition — cosmos_count + satellite_count + 1 = m^2
  QAP_2  Period witnesses — declared (b,e) pairs have stated period under sigma
  QAP_3  Swap symmetry — orbit family is preserved under (b,e)->(e,b) for all m^2 pairs
  QAP_4  Period completeness — exactly 3 distinct Pisano periods achievable
  QAP_5  Negation parity — orbit family preserved under (b,e)->(neg(b),neg(e)) for all
         m^2 pairs, where neg(b) = m-(b%m) if b%m!=0 else m; mirrors Pudelko's
         observation that additive inversion preserves the period distribution
  SRC    schema_version declared
  F      fail_ledger well-formed

Primary: Pudelko M.T. (2025). Modular Periodicity of Random Initialized
Recurrences. arXiv:2510.24882 v5 (2026-04-09). Results 1.1-1.5: complete
period classification; distinct periods = {1, 8, 24} for m=9; additive-inverse
parity. Wildberger N.J. (2005). Divine Proportions. Wild Egg Books.
ISBN 978-0-9757492-0-8 (QA step operator, no-zero convention).
Wall D.D. (1960). Fibonacci series modulo m. Amer. Math. Monthly
67(6):525-532. doi:10.2307/2309169 (Pisano periods, prime-power structure).
Companion cert: [261] qa_orbit_stratification_cert_v1 (content-ideal classification).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CERT_FAMILY = "QA_PISANO_ALL_INIT_CERT.v1"


# ---------------------------------------------------------------------------
# Pure-integer QA arithmetic  (A1, A2, S1, S2, T1 compliant)
# ---------------------------------------------------------------------------

def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    """A1-compliant QA step: sigma(b,e) = (e, ((b+e-1)%m)+1)."""
    return e, ((b + e - 1) % m) + 1


def qa_period(b: int, e: int, m: int) -> int:
    """Period of (b,e) under qa_step. Returns integer >= 1."""
    b0, e0 = b, e
    for k in range(1, m * m * m + 2):
        b, e = qa_step(b, e, m)
        if b == b0 and e == e0:
            return k
    raise RuntimeError(f"period not found for ({b0},{e0}) mod {m}")  # unreachable


def qa_neg(b: int, m: int) -> int:
    """Additive inverse in QA {1,...,m}: neg(b) = m-(b%m) if b%m!=0 else m."""
    r = b % m
    return m - r if r != 0 else m


def classify_all(m: int) -> dict[int, list[tuple[int, int]]]:
    """Partition {1,...,m}^2 by Pisano period under qa_step."""
    result: dict[int, list[tuple[int, int]]] = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            p = qa_period(b, e, m)
            result.setdefault(p, []).append((b, e))
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _err(errors: list[str], code: str, msg: str) -> None:
    errors.append(f"{code}: {msg}")


def validate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    # SRC — schema_version
    sv = fixture.get("schema_version")
    checks["SRC"] = sv == CERT_FAMILY
    if not checks["SRC"]:
        _err(errors, "SRC", f"schema_version must be {CERT_FAMILY!r}, got {sv!r}")

    m: int = fixture.get("modulus", 9)
    if not isinstance(m, int) or m < 2:
        _err(errors, "QAP_1", "modulus must be integer >= 2")
        m = 9

    cosmos_count_decl: int | None = fixture.get("cosmos_count")
    satellite_count_decl: int | None = fixture.get("satellite_count")

    # Compute exhaustive partition (pure integer)
    period_map = classify_all(m)
    period_keys = sorted(period_map.keys())

    p_sing = 1
    p_cosmos = max(period_keys)
    p_satellite_list = [p for p in period_keys if p != p_sing and p != p_cosmos]

    sing_pairs = period_map.get(p_sing, [])
    cosmos_pairs = period_map.get(p_cosmos, [])
    sat_pairs: list[tuple[int, int]] = []
    for p in p_satellite_list:
        sat_pairs.extend(period_map[p])

    cosmos_count = len(cosmos_pairs)
    satellite_count = len(sat_pairs)
    total = cosmos_count + satellite_count + len(sing_pairs)

    # QAP_1 — exhaustive partition
    qap1_ok = total == m * m and len(sing_pairs) == 1
    if total != m * m:
        _err(errors, "QAP_1", f"total={total} != m^2={m * m}")
    if len(sing_pairs) != 1:
        _err(errors, "QAP_1", f"expected 1 singularity, got {len(sing_pairs)}")
    if cosmos_count_decl is not None and cosmos_count_decl != cosmos_count:
        _err(errors, "QAP_1", f"declared cosmos_count={cosmos_count_decl} computed={cosmos_count}")
        qap1_ok = False
    if satellite_count_decl is not None and satellite_count_decl != satellite_count:
        _err(errors, "QAP_1", f"declared satellite_count={satellite_count_decl} computed={satellite_count}")
        qap1_ok = False
    checks["QAP_1"] = qap1_ok

    # QAP_2 — period witnesses
    witnesses = fixture.get("period_witnesses", {})
    qap2_ok = True
    for family_name, witness_list in witnesses.items():
        if not isinstance(witness_list, list):
            _err(errors, "QAP_2", f"period_witnesses.{family_name} must be list")
            qap2_ok = False
            continue
        for w in witness_list:
            if not isinstance(w, dict):
                _err(errors, "QAP_2", f"witness must be object")
                qap2_ok = False
                continue
            wb, we, decl_p = w.get("b"), w.get("e"), w.get("period")
            if not all(isinstance(x, int) for x in (wb, we, decl_p)):
                _err(errors, "QAP_2", f"witness b/e/period must be int, got {w}")
                qap2_ok = False
                continue
            if not (1 <= wb <= m and 1 <= we <= m):
                _err(errors, "QAP_2", f"witness ({wb},{we}) out of range {{1,...,{m}}}")
                qap2_ok = False
                continue
            actual_p = qa_period(wb, we, m)
            if actual_p != decl_p:
                _err(errors, "QAP_2", f"({wb},{we}): declared period={decl_p} computed={actual_p}")
                qap2_ok = False
    checks["QAP_2"] = qap2_ok

    # QAP_3 — swap symmetry: orbit family preserved under (b,e)->(e,b)
    period_of: dict[tuple[int, int], int] = {}
    for p, pairs in period_map.items():
        for pair in pairs:
            period_of[pair] = p

    qap3_ok = True
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if period_of[(b, e)] != period_of[(e, b)]:
                _err(errors, "QAP_3", f"swap fails: period({b},{e})={period_of[(b,e)]} != period({e},{b})={period_of[(e,b)]}")
                qap3_ok = False
                break
        if not qap3_ok:
            break
    checks["QAP_3"] = qap3_ok

    # QAP_4 — period completeness: exactly 3 distinct periods with 1 being the minimum
    declared_period_set = fixture.get("period_set")
    actual_period_set = set(period_keys)
    qap4_ok = len(actual_period_set) == 3 and p_sing in actual_period_set
    if len(actual_period_set) != 3:
        _err(errors, "QAP_4", f"expected 3 distinct periods, got {len(actual_period_set)}: {sorted(actual_period_set)}")
    if p_sing not in actual_period_set:
        _err(errors, "QAP_4", "period 1 (singularity) absent from period set")
    if declared_period_set is not None:
        if not isinstance(declared_period_set, list):
            _err(errors, "QAP_4", "period_set must be list")
            qap4_ok = False
        elif set(declared_period_set) != actual_period_set:
            _err(errors, "QAP_4", f"declared period_set={sorted(declared_period_set)} computed={sorted(actual_period_set)}")
            qap4_ok = False
    checks["QAP_4"] = qap4_ok

    # QAP_5 — negation parity: orbit family preserved under (b,e)->(neg(b),neg(e))
    qap5_ok = True
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            nb, ne = qa_neg(b, m), qa_neg(e, m)
            if period_of[(b, e)] != period_of[(nb, ne)]:
                _err(errors, "QAP_5", f"negation parity fails at ({b},{e}): period={period_of[(b,e)]} neg=({nb},{ne}) period={period_of[(nb,ne)]}")
                qap5_ok = False
                break
        if not qap5_ok:
            break
    checks["QAP_5"] = qap5_ok

    # F — fail_ledger
    fail_ledger = fixture.get("fail_ledger")
    checks["F"] = fail_ledger is None or isinstance(fail_ledger, list)
    if not checks["F"]:
        _err(errors, "F", "fail_ledger must be a list if present")

    ok = not errors
    result: dict[str, Any] = {
        "ok": ok,
        "checks": {k: {"ok": v} for k, v in checks.items()},
    }
    if errors:
        result["errors"] = errors
    else:
        result["computed"] = {
            "modulus": m,
            "cosmos_count": cosmos_count,
            "cosmos_period": p_cosmos,
            "satellite_count": satellite_count,
            "satellite_periods": sorted(p_satellite_list),
            "singularity_count": len(sing_pairs),
            "period_set": sorted(actual_period_set),
        }
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    fixtures_dir = base / "fixtures"

    results = []
    all_ok = True

    pass_fixtures = ["pass_full_partition_m9.json"]
    fail_fixtures = [
        "fail_wrong_cosmos_count.json",
        "fail_wrong_period_witness.json",
        "fail_wrong_period_set.json",
    ]

    for fname in pass_fixtures:
        r = validate_fixture(_load(fixtures_dir / fname))
        entry = {"fixture": fname, "expected": "PASS", "ok": r["ok"] is True}
        if not entry["ok"]:
            entry["errors"] = r.get("errors", [])
            all_ok = False
        results.append(entry)

    for fname in fail_fixtures:
        r = validate_fixture(_load(fixtures_dir / fname))
        entry = {"fixture": fname, "expected": "FAIL", "ok": r["ok"] is False}
        if not entry["ok"]:
            entry["errors"] = ["expected FAIL but got PASS"]
            all_ok = False
        results.append(entry)

    payload = {
        "cert_slug": "qa_pisano_all_initializations_cert_v1",
        "ok": all_ok,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate QA_PISANO_ALL_INIT_CERT.v1 fixtures"
    )
    parser.add_argument("fixture", nargs="?", help="Path to fixture JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.fixture:
        parser.print_help()
        return 2

    data = _load(Path(args.fixture))
    r = validate_fixture(data)
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
