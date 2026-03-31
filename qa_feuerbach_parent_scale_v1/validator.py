"""
QA_FEUERBACH_PARENT_SCALE_CERT.v1 — Validator

Certifies the Feuerbach parent-scale law for primitive Pythagorean triples.

Core geometric construction (exact, integer arithmetic):
  Place right triangle with legs C (horizontal), F (vertical), hypotenuse G at origin.
  Incenter:         I  = (r, r)       where r = (C + F - G) / 2
  Nine-point center: N = (C/4, F/4)
  Parent legs: leg1 = |C + 2F - 2G|,  leg2 = |2C + F - 2G|
  (Equivalently: leg1 = |4(r - C/4)| = |4r - C|,  leg2 = |4r - F|)

Interior law (all non-root primitive triples):
  Both legs are positive integers forming a primitive Pythagorean triple — the Barning parent.
  The implicit scale factor is always 4.

Root exception (3,4,5):
  leg1 = |3 + 8 - 10| = 1,  leg2 = |6 + 4 - 10| = 0  →  degenerate (one leg = 0).
  QA interpretation: boundary obstruction; closure scale = 2G = 10.

Gates:
  1 — Schema anchor  (required fields, schema_id const)
  2 — Root exception (triple=[3,4,5], raw_legs=[0,1], qa_scale=10)
  3 — Sample recompute (parent_legs match formula for every sample)
  4 — Batch recompute (primitive_count, confirmed_count, uniqueness, sex_invariant all match)

Usage:
  python validator.py --self-test          # run built-in tests, emit JSON
  python validator.py <cert.json>          # validate a certificate file
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_ID = "QA_FEUERBACH_PARENT_SCALE_CERT.v1"
REQUIRED_FIELDS = [
    "schema_id", "cert_id", "created_utc", "batch_limit",
    "primitive_count", "interior_law", "root_exception",
    "uniqueness_confirmed", "sex_invariant_confirmed", "samples", "invariant_diff",
]

# ── Core mathematics ──────────────────────────────────────────────────────────

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _is_primitive(C: int, F: int, G: int) -> bool:
    return _gcd(_gcd(abs(C), abs(F)), abs(G)) == 1


def _generate_primitives(limit: int) -> List[Tuple[int, int, int]]:
    """Return all primitive triples (C, F, G) with C < F, C²+F²=G², G ≤ limit, sorted by G."""
    seen: set = set()
    triples: List[Tuple[int, int, int]] = []
    m = 2
    while True:
        # Smallest G for this m is m²+1 (n=1).  If m²+1 > limit we are done.
        if m * m + 1 > limit:
            break
        for n in range(1, m):
            if (m - n) % 2 == 0:
                continue                        # must differ in parity
            if _gcd(m, n) != 1:
                continue                        # must be coprime
            G = m * m + n * n
            if G > limit:
                break
            C0 = m * m - n * n
            F0 = 2 * m * n
            C, F = min(C0, F0), max(C0, F0)
            key = (C, F, G)
            if key not in seen:
                seen.add(key)
                triples.append(key)
        m += 1
    return sorted(triples, key=lambda t: (t[2], t[0]))


def _feuerbach_parent_legs(C: int, F: int, G: int) -> Tuple[int, int]:
    """
    Compute sorted (min, max) parent legs via the exact Feuerbach construction.

    Derivation (all integer):
      r   = (C + F - G) // 2   [inradius; integer for any Pythagorean triple]
      4r  = 2C + 2F - 2G
      leg1 = |4r - C| = |C + 2F - 2G|
      leg2 = |4r - F| = |2C + F - 2G|
    """
    leg1 = abs(C + 2 * F - 2 * G)
    leg2 = abs(2 * C + F - 2 * G)
    return (min(leg1, leg2), max(leg1, leg2))


def _classify_sex(C: int, F: int) -> str:
    """
    male  → C is divisible by 4 (C is the even leg)
    female → F is divisible by 4 (F is the even leg)
    In a primitive triple exactly one of C, F is even, and that leg ≡ 0 (mod 4).
    """
    return "male" if C % 2 == 0 else "female"


def _sex_invariant_holds(C: int, F: int) -> bool:
    """Even leg must be divisible by 4."""
    even = C if C % 2 == 0 else F
    return even % 4 == 0


# ── Validation gates ──────────────────────────────────────────────────────────

GateResult = Dict[str, Any]


def _gate1_schema(cert: Dict[str, Any]) -> GateResult:
    """Gate 1: required fields present, schema_id anchor correct."""
    missing = [f for f in REQUIRED_FIELDS if f not in cert]
    if missing:
        return {"gate": 1, "ok": False, "fail_type": "MISSING_FIELDS",
                "detail": f"Missing: {missing}"}
    if cert["schema_id"] != SCHEMA_ID:
        return {"gate": 1, "ok": False, "fail_type": "SCHEMA_ID_MISMATCH",
                "detail": f"Expected {SCHEMA_ID!r}, got {cert['schema_id']!r}"}
    il = cert["interior_law"]
    if not isinstance(il, dict) or "scale_value" not in il or "confirmed_count" not in il:
        return {"gate": 1, "ok": False, "fail_type": "BAD_INTERIOR_LAW",
                "detail": "interior_law must have scale_value and confirmed_count"}
    return {"gate": 1, "ok": True}


def _gate2_root_exception(cert: Dict[str, Any]) -> GateResult:
    """Gate 2: root exception triple=[3,4,5], raw_legs=[0,1], qa_scale=10."""
    rex = cert["root_exception"]
    expected_triple = [3, 4, 5]
    if rex.get("triple") != expected_triple:
        return {"gate": 2, "ok": False, "fail_type": "WRONG_ROOT_TRIPLE",
                "detail": f"Expected {expected_triple}, got {rex.get('triple')}"}

    # Recompute raw_legs for (3,4,5)
    leg_min, leg_max = _feuerbach_parent_legs(3, 4, 5)
    expected_raw = [leg_min, leg_max]          # should be [0, 1]
    if rex.get("raw_legs") != expected_raw:
        return {"gate": 2, "ok": False, "fail_type": "WRONG_RAW_LEGS",
                "detail": f"Expected raw_legs={expected_raw}, got {rex.get('raw_legs')}"}

    expected_qa_scale = 2 * 5                  # 2 * G_root = 10
    if rex.get("qa_scale") != expected_qa_scale:
        return {"gate": 2, "ok": False, "fail_type": "WRONG_QA_SCALE",
                "detail": f"Expected qa_scale={expected_qa_scale}, got {rex.get('qa_scale')}"}

    return {"gate": 2, "ok": True}


def _gate3_sample_recompute(cert: Dict[str, Any]) -> GateResult:
    """Gate 3: recompute parent_legs for every sample and verify claimed values."""
    errors = []
    for i, s in enumerate(cert.get("samples", [])):
        try:
            C, F, G = s["triple"]
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"sample[{i}] bad triple: {exc}")
            continue

        computed = list(_feuerbach_parent_legs(C, F, G))
        claimed  = s.get("parent_legs")
        if claimed != computed:
            errors.append(
                f"sample[{i}] triple=({C},{F},{G}): "
                f"claimed parent_legs={claimed}, computed={computed}"
            )

        computed_sex = _classify_sex(C, F)
        claimed_sex  = s.get("sex")
        if claimed_sex != computed_sex:
            errors.append(
                f"sample[{i}] triple=({C},{F},{G}): "
                f"claimed sex={claimed_sex!r}, computed={computed_sex!r}"
            )

    if errors:
        return {"gate": 3, "ok": False, "fail_type": "SAMPLE_MISMATCH",
                "detail": errors}
    return {"gate": 3, "ok": True}


def _gate4_batch_recompute(cert: Dict[str, Any]) -> GateResult:
    """Gate 4: full batch sweep — verify primitive_count, confirmed_count, uniqueness, sex_invariant."""
    limit = cert["batch_limit"]
    triples = _generate_primitives(limit)

    # primitive_count
    if len(triples) != cert["primitive_count"]:
        return {"gate": 4, "ok": False, "fail_type": "PRIMITIVE_COUNT_MISMATCH",
                "detail": f"Expected {cert['primitive_count']}, computed {len(triples)}"}

    # interior_law.scale_value must be 4
    if cert["interior_law"]["scale_value"] != 4:
        return {"gate": 4, "ok": False, "fail_type": "WRONG_SCALE_VALUE",
                "detail": f"interior_law.scale_value must be 4, got {cert['interior_law']['scale_value']}"}

    # Walk every triple
    degenerate = []
    confirmed = 0
    sex_violations = []

    for C, F, G in triples:
        legs = _feuerbach_parent_legs(C, F, G)
        is_degenerate = (legs[0] == 0)

        if is_degenerate:
            degenerate.append((C, F, G))
        else:
            # Both legs positive — verify they form a valid Pythagorean triple
            a, b = legs
            hyp_sq = a * a + b * b
            hyp = int(math.isqrt(hyp_sq))
            if hyp * hyp != hyp_sq:
                return {"gate": 4, "ok": False, "fail_type": "PARENT_NOT_PYTHAGOREAN",
                        "detail": f"triple=({C},{F},{G}) gives non-Pythagorean parent legs {legs}"}
            confirmed += 1

        if not _sex_invariant_holds(C, F):
            sex_violations.append((C, F, G))

    # confirmed_count
    if confirmed != cert["interior_law"]["confirmed_count"]:
        return {"gate": 4, "ok": False, "fail_type": "CONFIRMED_COUNT_MISMATCH",
                "detail": f"Expected confirmed_count={cert['interior_law']['confirmed_count']}, got {confirmed}"}

    # uniqueness: exactly one degenerate triple, and it must be (3,4,5)
    if len(degenerate) != 1 or degenerate[0] != (3, 4, 5):
        return {"gate": 4, "ok": False, "fail_type": "UNIQUENESS_VIOLATION",
                "detail": f"Degenerate triples (expected exactly [(3,4,5)]): {degenerate}"}

    if not cert["uniqueness_confirmed"]:
        return {"gate": 4, "ok": False, "fail_type": "UNIQUENESS_FLAG_FALSE",
                "detail": "uniqueness_confirmed must be true"}

    # sex invariant
    if sex_violations:
        return {"gate": 4, "ok": False, "fail_type": "SEX_INVARIANT_VIOLATION",
                "detail": f"Even leg not div by 4: {sex_violations[:5]}"}

    if not cert["sex_invariant_confirmed"]:
        return {"gate": 4, "ok": False, "fail_type": "SEX_INVARIANT_FLAG_FALSE",
                "detail": "sex_invariant_confirmed must be true"}

    return {"gate": 4, "ok": True, "primitive_count": len(triples), "confirmed": confirmed}


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_cert(cert: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a certificate dict.  Returns {"ok": bool, "gates": [...], "invariant_diff": [...]}."""
    gates: List[GateResult] = []
    inv_diff: List[Dict[str, Any]] = []

    for fn in (_gate1_schema, _gate2_root_exception, _gate3_sample_recompute, _gate4_batch_recompute):
        result = fn(cert)
        gates.append(result)
        if not result["ok"]:
            inv_diff.append({
                "gate":      result["gate"],
                "fail_type": result.get("fail_type", "UNKNOWN"),
                "path":      f"gate{result['gate']}",
                "reason":    str(result.get("detail", "")),
            })
            return {"ok": False, "gates": gates, "invariant_diff": inv_diff}

    return {"ok": True, "gates": gates, "invariant_diff": []}


# ── Certificate generation (for self-test) ─────────────────────────────────────

def _generate_cert(limit: int = 100) -> Dict[str, Any]:
    """Generate a valid certificate for all primitives up to limit."""
    triples = _generate_primitives(limit)

    # Build samples (first 5 non-root triples)
    samples = []
    for C, F, G in triples:
        if (C, F, G) == (3, 4, 5):
            continue
        legs = _feuerbach_parent_legs(C, F, G)
        samples.append({
            "triple":      [C, F, G],
            "parent_legs": list(legs),
            "sex":         _classify_sex(C, F),
        })
        if len(samples) >= 5:
            break

    root_legs = _feuerbach_parent_legs(3, 4, 5)
    confirmed = sum(
        1 for C, F, G in triples
        if (C, F, G) != (3, 4, 5) and _feuerbach_parent_legs(C, F, G)[0] != 0
    )

    return {
        "schema_id":             SCHEMA_ID,
        "cert_id":               f"feuerbach-autogen-L{limit}",
        "created_utc":           "2026-03-21T00:00:00Z",
        "batch_limit":           limit,
        "primitive_count":       len(triples),
        "interior_law":          {"scale_value": 4, "confirmed_count": confirmed},
        "root_exception":        {
            "triple":   [3, 4, 5],
            "raw_legs": list(root_legs),
            "qa_scale": 10,
        },
        "uniqueness_confirmed":  True,
        "sex_invariant_confirmed": True,
        "samples":               samples,
        "invariant_diff":        [],
    }


# ── Entry points ───────────────────────────────────────────────────────────────

def _self_test() -> Dict[str, Any]:
    """Run built-in tests.  Returns {"ok": bool, ...}."""
    errors: List[str] = []

    # --- Test 1: generate cert for limit=100 and validate it ---
    cert_100 = _generate_cert(100)
    r100 = validate_cert(cert_100)
    if not r100["ok"]:
        errors.append(f"generated cert (limit=100) failed validation: {r100}")

    # --- Test 2: known bad cert — wrong scale_value ---
    bad_scale = dict(cert_100)
    bad_scale["interior_law"] = {"scale_value": 5, "confirmed_count": cert_100["interior_law"]["confirmed_count"]}
    rb = validate_cert(bad_scale)
    if rb["ok"]:
        errors.append("bad_scale cert should have failed but passed")
    if not any(d["fail_type"] == "WRONG_SCALE_VALUE" for d in rb["invariant_diff"]):
        errors.append(f"bad_scale cert: expected WRONG_SCALE_VALUE in inv_diff, got {rb['invariant_diff']}")

    # --- Test 3: bad root exception — wrong qa_scale ---
    bad_root = dict(cert_100)
    bad_root["root_exception"] = {"triple": [3, 4, 5], "raw_legs": [0, 1], "qa_scale": 4}
    rr = validate_cert(bad_root)
    if rr["ok"]:
        errors.append("bad_root_qa_scale cert should have failed but passed")
    if not any(d["fail_type"] == "WRONG_QA_SCALE" for d in rr["invariant_diff"]):
        errors.append(f"bad_root_qa_scale cert: expected WRONG_QA_SCALE, got {rr['invariant_diff']}")

    # --- Test 4: validate fixture files ---
    here = Path(__file__).parent
    for fpath in sorted(here.glob("fixtures/pass/*.json")):
        with fpath.open() as fh:
            fc = json.load(fh)
        fr = validate_cert(fc)
        if not fr["ok"]:
            errors.append(f"PASS fixture {fpath.name} failed: {fr['invariant_diff']}")

    for fpath in sorted(here.glob("fixtures/fail/*.json")):
        with fpath.open() as fh:
            fc = json.load(fh)
        fr = validate_cert(fc)
        if fr["ok"]:
            errors.append(f"FAIL fixture {fpath.name} unexpectedly passed")

    if errors:
        return {"ok": False, "errors": errors}
    return {
        "ok": True,
        "gates_tested": 4,
        "cert_limit_100_primitive_count": cert_100["primitive_count"],
        "cert_limit_100_confirmed": cert_100["interior_law"]["confirmed_count"],
    }


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) == 2:
        path = Path(sys.argv[1])
        if not path.exists():
            print(json.dumps({"ok": False, "error": f"File not found: {path}"}))
            sys.exit(1)
        with path.open() as fh:
            cert = json.load(fh)
        result = validate_cert(cert)
        print(json.dumps(result, sort_keys=True, indent=2))
        sys.exit(0 if result["ok"] else 1)

    print(f"Usage: python {sys.argv[0]} --self-test | <cert.json>", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
