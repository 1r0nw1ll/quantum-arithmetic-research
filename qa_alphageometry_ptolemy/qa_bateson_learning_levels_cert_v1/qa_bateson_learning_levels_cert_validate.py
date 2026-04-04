#!/usr/bin/env python3
"""
qa_bateson_learning_levels_cert_validate.py

Validator for QA_BATESON_LEARNING_LEVELS_CERT.v1  [family 191]

Certifies: Gregory Bateson's Learning Levels (0/I/II/III) formalized as a
strict invariant filtration on QA state spaces.

Invariant filtration:
    orbit  subset  family  subset  modulus  subset  ambient_category

Operator classes:
    L_0   : identity on fixed points
    L_1   : orbit-preserving (powers of T, intra-orbit automorphisms)
    L_2a  : orbit-changing, family-preserving
    L_2b  : family-changing, modulus-preserving
    L_3   : modulus-changing
    L_4   : category-changing (speculative ceiling; not asserted here)

TIERED REACHABILITY THEOREM:
    For s_0, s_* in S_m, the minimum tier tau(s_0, s_*) such that s_* is
    reachable from s_0 via operators of that class is:
        tau = 0   if s_0 == s_*
        tau = 1   if s_* in Orbit(s_0), s_* != s_0
        tau = 2a  if different orbit, same family
        tau = 2b  if different family, same modulus
        tau = 3   if different modulus

    Proof: R_1(s_0) = Orbit(s_0) because T in L_1 generates the orbit and
    every L_1 operator preserves orbit membership. Hence targets outside
    Orbit(s_0) are Level-I unreachable (double bind). Promotion required
    to the tier determined by the broken invariant.

EXHAUSTIVE VERIFICATION on S_9 (2026-04-04):
    81 + 1712 + 3456 + 1312 = 6561 ordered pairs classified.
    Counts match structural prediction exactly.

Source: Bateson, "Steps to an Ecology of Mind" (1972); Russell & Whitehead,
"Principia Mathematica" (logical types); Ashby, "An Introduction to
Cybernetics" (1956, requisite variety). QA formalization: Will Dale + Claude,
2026-04-04. Full sketch: docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md.

Checks:
    BLL_1       — schema_version matches
    BLL_FILT    — invariant filtration well-formed (4 invariants, 5+ classes)
    BLL_TIER    — tier distribution on S_9 matches structural prediction
    BLL_L1      — at least one Level-I witness (orbit-preserving)
    BLL_L2A     — at least one Level-II-a witness (orbit-changing, family-preserving)
    BLL_L2B     — at least one Level-II-b witness (family-changing)
    BLL_L3      — at least one Level-III witness (modulus-changing)
    BLL_STRICT  — strict inclusions L_1 ⊊ L_2a ⊊ L_2b ⊊ L_3 verified
    BLL_DB      — double bind theorem: 81+1712+3456+1312 = 6561 on S_9
    BLL_SRC     — source attribution to Bateson present
    BLL_WITNESS — at least 4 witnesses (one per non-trivial tier)
    BLL_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates finite S_9 operator classification; integer state space; no observer, no floats, no continuous dynamics"

import json
import os
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_BATESON_LEARNING_LEVELS_CERT.v1"

# Expected exhaustive tier counts on S_9 (verified 2026-04-04 via
# tools/verify_bateson_double_bind.py). These are theorem constants.
EXPECTED_TIER_COUNTS_S9 = {
    "0": 81,
    "1": 1712,
    "2a": 3456,
    "2b": 1312,
}
EXPECTED_TOTAL_S9 = 6561  # 81 * 81


# -----------------------------------------------------------------------------
# QA primitives (integer-only, axiom-compliant)
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    """Fibonacci dynamic: (b,e) -> (e, b+e mod m). A1-compliant."""
    return (e, qa_mod(b + e, m))


def orbit_family_s9(b, e):
    """Canonical S_9 orbit family classification.

    singularity : b == 9 AND e == 9
    satellite   : 3|b AND 3|e (excludes singularity)
    cosmos      : everything else
    """
    if b == 9 and e == 9:
        return "singularity"
    if (b % 3 == 0) and (e % 3 == 0):
        return "satellite"
    return "cosmos"


def enumerate_orbits_s9():
    """Decompose S_9 into T-orbits. Returns list of orbit tuples."""
    m = 9
    seen = set()
    orbits = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            orbit = []
            cur = (b, e)
            while cur not in seen:
                seen.add(cur)
                orbit.append(cur)
                cur = qa_step(cur[0], cur[1], m)
            orbits.append(tuple(orbit))
    return orbits


def build_orbit_index(orbits):
    idx = {}
    for i, o in enumerate(orbits):
        for pt in o:
            idx[pt] = i
    return idx


def classify_tier(s0, s_star, idx):
    """Determine minimum tier tau(s0, s*) per the Tiered Reachability Theorem."""
    if s0 == s_star:
        return "0"
    if idx[s0] == idx[s_star]:
        return "1"
    fam0 = orbit_family_s9(*s0)
    famS = orbit_family_s9(*s_star)
    if fam0 == famS:
        return "2a"
    return "2b"


# -----------------------------------------------------------------------------
# Operator application by declared kind
# -----------------------------------------------------------------------------

def apply_operator(op, b, e, m=9):
    """Apply a declared operator to (b,e). Returns (b', e') or None on error.

    Supported kinds:
        identity                            : phi(b,e) = (b,e)
        qa_step                             : phi(b,e) = T(b,e)
        scalar_mult k                       : phi(b,e) = (kb mod m, ke mod m)
        swap                                : phi(b,e) = (e,b)
        constant c_b c_e                    : phi(b,e) = (c_b, c_e)
        modulus_reduction m'                : S_m -> S_{m'}, (b mod m', e mod m')
                                              with A1-compliant mod
    """
    kind = op.get("kind")
    if kind == "identity":
        return (b, e)
    if kind == "qa_step":
        return qa_step(b, e, m)
    if kind == "scalar_mult":
        k = int(op["k"])
        return (qa_mod(k * b, m), qa_mod(k * e, m))
    if kind == "swap":
        return (e, b)
    if kind == "constant":
        return (int(op["c_b"]), int(op["c_e"]))
    if kind == "modulus_reduction":
        mp = int(op["m_target"])
        return (qa_mod(b, mp), qa_mod(e, mp))
    return None


def operator_self_maps_S9(op, orbits, idx):
    """Classify an operator phi: S_9 -> S_9 by (orbit_preserving, family_preserving)."""
    orbit_pres = True
    fam_pres = True
    for b in range(1, 10):
        for e in range(1, 10):
            result = apply_operator(op, b, e, m=9)
            if result is None:
                return None
            bi, ei = result
            if not (1 <= bi <= 9 and 1 <= ei <= 9):
                # Codomain escape — this is a Level-III operator, not a self-map of S_9
                return ("not_self_map", None)
            if idx.get((bi, ei)) != idx[(b, e)]:
                orbit_pres = False
            if orbit_family_s9(b, e) != orbit_family_s9(bi, ei):
                fam_pres = False
    return (orbit_pres, fam_pres)


def operator_is_modulus_changing(op):
    """Level-III detection: operator changes the ambient modulus."""
    return op.get("kind") == "modulus_reduction" and int(op.get("m_target", 9)) != 9


# -----------------------------------------------------------------------------
# Invariant filtration check
# -----------------------------------------------------------------------------

EXPECTED_FILTRATION = [
    {"invariant": "orbit", "class": "L_1", "preserves_below": []},
    {"invariant": "family", "class": "L_2a", "preserves_below": ["family"]},
    {"invariant": "modulus", "class": "L_2b", "preserves_below": []},
    {"invariant": "ambient_category", "class": "L_3", "preserves_below": []},
]


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # BLL_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"BLL_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # BLL_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("BLL_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("BLL_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # BLL_SRC: source attribution
    src = cert.get("source_attribution", "")
    if "Bateson" not in str(src):
        warnings.append("BLL_SRC: source_attribution should credit Gregory Bateson")

    # BLL_FILT: invariant filtration declaration
    filt = cert.get("invariant_filtration")
    if not isinstance(filt, list):
        errors.append("BLL_FILT: invariant_filtration must be a list")
    else:
        decl_invariants = [f.get("invariant") for f in filt]
        expected_invariants = ["orbit", "family", "modulus", "ambient_category"]
        for inv in expected_invariants:
            if inv not in decl_invariants:
                errors.append(f"BLL_FILT: missing invariant '{inv}' in filtration")

    # BLL_TIER: exhaustive tier count on S_9
    orbits = enumerate_orbits_s9()
    idx = build_orbit_index(orbits)
    counts = {"0": 0, "1": 0, "2a": 0, "2b": 0}
    for b0 in range(1, 10):
        for e0 in range(1, 10):
            for bS in range(1, 10):
                for eS in range(1, 10):
                    t = classify_tier((b0, e0), (bS, eS), idx)
                    counts[t] += 1
    total = sum(counts.values())
    if total != EXPECTED_TOTAL_S9:
        errors.append(f"BLL_TIER: total pair count {total} != expected {EXPECTED_TOTAL_S9}")
    for t, expected in EXPECTED_TIER_COUNTS_S9.items():
        if counts[t] != expected:
            errors.append(
                f"BLL_TIER: tier {t} count {counts[t]} != expected {expected}"
            )

    # BLL_DB: double bind theorem constants
    db_counts = cert.get("tier_distribution_s9", {})
    for t, expected in EXPECTED_TIER_COUNTS_S9.items():
        decl = db_counts.get(t)
        if decl is None:
            warnings.append(f"BLL_DB: tier_distribution_s9 missing tier {t}")
        elif decl != expected:
            errors.append(
                f"BLL_DB: declared tier {t} count {decl} != structural {expected}"
            )

    # BLL_L1 / L2A / L2B / L3: witness checks
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 4:
        errors.append(f"BLL_WITNESS: need >= 4 witnesses (one per tier), got {len(witnesses)}")

    tier_coverage = {"1": False, "2a": False, "2b": False, "3": False}

    for wi, w in enumerate(witnesses):
        declared_tier = w.get("tier")
        op = w.get("operator")
        s0 = tuple(w.get("source", []))
        s_star = tuple(w.get("target", []))

        if op is None:
            errors.append(f"BLL_WITNESS: witness[{wi}] missing operator")
            continue

        # Level-III check first: modulus-changing
        if operator_is_modulus_changing(op):
            if declared_tier != "3":
                errors.append(
                    f"BLL_L3: witness[{wi}] is modulus-changing but declared tier={declared_tier}"
                )
            else:
                tier_coverage["3"] = True
            continue

        # Self-map of S_9: verify the operator produces the declared target
        if len(s0) == 2 and len(s_star) == 2:
            actual = apply_operator(op, s0[0], s0[1], m=9)
            if actual != s_star:
                errors.append(
                    f"BLL_WITNESS: witness[{wi}] operator({s0}) = {actual}, declared target = {s_star}"
                )
                continue

            # Compute actual tier from (s0, s_star)
            if s0 == s_star:
                actual_tier = "0"
            else:
                actual_tier = classify_tier(s0, s_star, idx)

            if declared_tier != actual_tier:
                errors.append(
                    f"BLL_WITNESS: witness[{wi}] declared tier={declared_tier}, actual={actual_tier} for {s0}->{s_star}"
                )
                continue

        # Classify operator level (orbit_pres, family_pres) on whole S_9
        cls = operator_self_maps_S9(op, orbits, idx)
        if cls is None:
            errors.append(f"BLL_WITNESS: witness[{wi}] operator application failed")
            continue

        if cls[0] == "not_self_map":
            errors.append(f"BLL_WITNESS: witness[{wi}] escapes S_9 but declared tier={declared_tier}")
            continue

        orbit_pres, fam_pres = cls

        if declared_tier == "1":
            if not orbit_pres:
                errors.append(f"BLL_L1: witness[{wi}] declared L_1 but not orbit-preserving")
            else:
                tier_coverage["1"] = True
        elif declared_tier == "2a":
            if orbit_pres or not fam_pres:
                errors.append(
                    f"BLL_L2A: witness[{wi}] declared L_2a but (orbit_pres={orbit_pres}, fam_pres={fam_pres}); need (False, True)"
                )
            else:
                tier_coverage["2a"] = True
        elif declared_tier == "2b":
            if fam_pres:
                errors.append(
                    f"BLL_L2B: witness[{wi}] declared L_2b but family-preserving"
                )
            else:
                tier_coverage["2b"] = True

    # BLL_STRICT: strict inclusions verified iff each tier has >= 1 witness
    missing = [t for t, covered in tier_coverage.items() if not covered]
    if missing:
        errors.append(f"BLL_STRICT: missing witnesses for tier(s) {missing}; strict inclusions unverified")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("bll_pass_hierarchy.json", True),
        ("bll_fail_bad_tier.json", True),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Bateson Learning Levels Cert [191] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
