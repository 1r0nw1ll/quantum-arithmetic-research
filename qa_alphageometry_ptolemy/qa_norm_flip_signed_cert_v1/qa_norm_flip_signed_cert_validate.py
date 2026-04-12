#!/usr/bin/env python3
"""
qa_norm_flip_signed_cert_validate.py

Validator for QA_NORM_FLIP_SIGNED_CERT.v1  [family 214]

Certifies: The Eisenstein quadratic form f(b, e) = b*b + b*e - e*e satisfies
the integer identity

    f(T(b, e)) = -f(b, e)

where T(b, e) = (e, b+e) is the QA generator. Consequently T^2 preserves f
mod m, giving the T-orbit graph of S_m a natural signed-temporal structure:
at each integer path-time step, the sign of the unreduced Eisenstein norm
flips, and the mod-m norm values partition into sign-pair cohorts.

On S_9, the 5 T-orbits decompose exactly as:

    Cosmos orbit 0 (rep (1,1)): length 24, norm pair {1, 8} mod 9  [Fibonacci]
    Cosmos orbit 1 (rep (1,3)): length 24, norm pair {4, 5} mod 9  [Lucas]
    Cosmos orbit 2 (rep (1,4)): length 24, norm pair {2, 7} mod 9  [Phibonacci]
    Satellite   (rep (3,3)):   length 8,  norms {0}                [Tribonacci]
    Singularity (rep (9,9)):   length 1,  norms {0}                [Ninbonacci]

The three cosmos orbits are bipartite signed (12 states with norm +k, 12
with norm -k = m-k). The satellite and singularity together form the "null"
subgraph where the Eisenstein norm is identically zero mod 9.

Proof of the flip identity (integer):
    f(e, b+e) = e*e + e*(b+e) - (b+e)*(b+e)
              = e*e + e*b + e*e - b*b - 2*b*e - e*e
              = e*e - b*e - b*b
              = -(b*b + b*e - e*e)
              = -f(b, e)  ∎

Temporal sign formula on the Z^2 integer lift:
    sign(f(T^t(s_0))) = (-1)^t * sign(f(s_0))
for as long as mod reduction does not intervene. On the mod-m orbit, the
norm-mod-m values exhibit a bipartite coloring with period 2.

Source grounding:
    - Gotthold Eisenstein, "Untersuchungen ueber die cubischen Formen mit zwei
      Variabeln" (Journal fuer die reine und angewandte Mathematik 27, 1844)
      — binary/ternary quadratic forms with determinant-1 linear action
    - Pythagorean Families paper (Will Dale + Claude, 2026-03): 5-orbit
      classification with classical names
    - Prerequisite [133] qa_eisenstein_cert — the Eisenstein norm identities
      F^2 - FW + W^2 = Z^2 that use this quadratic form
    - Related [155] qa_bearden_phase_conjugate_cert — phase conjugation
      mechanism structurally parallel to norm-sign flip
    - Related [191] qa_bateson_learning_levels_cert — cosmos/satellite/
      singularity stratification IS the signed-vs-null orbit partition

Checks:
    NFS_1        — schema_version matches
    NFS_FLIP     — integer identity f(e, b+e) = -f(b, e) verified 81/81
    NFS_T2       — T^2 preserves f mod 9, 81/81
    NFS_PAIRS    — 5 orbit families declared with correct norm pairs
                   (or null {0}) and classical names
    NFS_TEMPORAL — temporal sign formula declared: sign = (-1)^t * sign(s_0)
    NFS_155      — cross-reference to family 155 present
    NFS_133      — cross-reference to family 133 present
    NFS_SRC      — source attribution to Eisenstein present
    NFS_WIT      — >= 5 witnesses (one per orbit family)
    NFS_F        — fail_ledger well-formed

QA axiom compliance: integer state alphabet {1..m}, A1-adjusted sector labels,
S1-compliant (b*b not b**2), no floats, no observer inputs crossing into QA state.
"""

QA_COMPLIANCE = "cert_validator — validates signed-temporal Eisenstein norm flip theorem on S_9; integer state space; no observer, no floats, no continuous dynamics"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_NORM_FLIP_SIGNED_CERT.v1"

EXPECTED_S9_ORBITS = [
    {"length": 24, "norms": [1, 8], "pair_type": "signed"},
    {"length": 24, "norms": [4, 5], "pair_type": "signed"},
    {"length": 24, "norms": [2, 7], "pair_type": "signed"},
    {"length": 8,  "norms": [0],    "pair_type": "null"},
    {"length": 1,  "norms": [0],    "pair_type": "null"},
]


# -----------------------------------------------------------------------------
# QA primitives (integer-only, S1-compliant, stdlib only)
# -----------------------------------------------------------------------------

def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    return (e, qa_mod(b + e, m))


def eisenstein_norm(b, e):
    """f(b, e) = b*b + b*e - e*e. S1: b*b not b**2."""
    return b * b + b * e - e * e


def enumerate_state_space(m):
    return [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)]


# -----------------------------------------------------------------------------
# Independent theorem recomputation on S_m
# -----------------------------------------------------------------------------

def compute_flip_identity(m):
    checked = 0
    matched = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            checked += 1
            if eisenstein_norm(e, b + e) == -eisenstein_norm(b, e):
                matched += 1
    return (checked, matched)


def compute_t2_preservation(m):
    checked = 0
    matched = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            checked += 1
            n0 = eisenstein_norm(b, e) % m
            b2, e2 = qa_step(b, e, m)
            b4, e4 = qa_step(b2, e2, m)
            n2 = eisenstein_norm(b4, e4) % m
            if n0 == n2:
                matched += 1
    return (checked, matched)


def compute_signed_orbits(m):
    """Return list of (length, sorted_norms, pair_type) for each T-orbit."""
    seen = set()
    results = []
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
            norms = sorted(set(eisenstein_norm(bi, ei) % m for bi, ei in orbit))
            if norms == [0]:
                pair_type = "null"
            elif len(norms) == 2 and (norms[0] + norms[1]) % m == 0:
                pair_type = "signed"
            else:
                pair_type = "other"
            results.append((len(orbit), norms, pair_type))
    return results


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # NFS_1
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"NFS_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # NFS_F
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("NFS_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("NFS_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # NFS_SRC
    src = str(cert.get("source_attribution", ""))
    if "Eisenstein" not in src:
        warnings.append("NFS_SRC: source_attribution should credit Eisenstein")

    # NFS_155
    xrefs = cert.get("cross_references", [])
    has_155 = isinstance(xrefs, list) and any(
        isinstance(x, dict) and x.get("family") == 155 for x in xrefs
    )
    if not has_155:
        errors.append("NFS_155: missing cross_reference to family 155 (Bearden phase conjugate)")

    # NFS_133
    has_133 = isinstance(xrefs, list) and any(
        isinstance(x, dict) and x.get("family") == 133 for x in xrefs
    )
    if not has_133:
        errors.append("NFS_133: missing cross_reference to family 133 (Eisenstein norm)")

    # NFS_FLIP: independently recompute on {1..9}^2
    checked, matched = compute_flip_identity(9)
    if checked != 81 or matched != 81:
        errors.append(f"NFS_FLIP: integer identity f(e,b+e)=-f(b,e) recomputation gave {matched}/{checked} (expected 81/81)")
    decl_flip = cert.get("flip_identity_s9", {})
    if decl_flip:
        if decl_flip.get("checked") != 81 or decl_flip.get("matched") != 81:
            errors.append(f"NFS_FLIP: declared flip_identity_s9 {decl_flip} != expected 81/81")

    # NFS_T2
    checked2, matched2 = compute_t2_preservation(9)
    if checked2 != 81 or matched2 != 81:
        errors.append(f"NFS_T2: T^2 preservation recomputation gave {matched2}/{checked2} (expected 81/81)")
    decl_t2 = cert.get("t2_preservation_s9", {})
    if decl_t2:
        if decl_t2.get("checked") != 81 or decl_t2.get("matched") != 81:
            errors.append(f"NFS_T2: declared t2_preservation_s9 {decl_t2} != expected 81/81")

    # NFS_PAIRS: orbit classification matches expected
    actual_orbits = compute_signed_orbits(9)
    expected_sorted = sorted(
        ((o["length"], tuple(o["norms"]), o["pair_type"]) for o in EXPECTED_S9_ORBITS),
        reverse=True,
    )
    actual_sorted = sorted(
        ((length, tuple(norms), pt) for length, norms, pt in actual_orbits),
        reverse=True,
    )
    if actual_sorted != expected_sorted:
        errors.append(
            f"NFS_PAIRS: orbit classification {actual_sorted} != expected {expected_sorted}"
        )
    decl_orbits = cert.get("signed_orbits_s9", [])
    if decl_orbits:
        decl_sorted = sorted(
            (
                (
                    int(o.get("length", -1)),
                    tuple(int(n) for n in o.get("distinct_norms_mod_m", [])),
                    str(o.get("pair_type", "")),
                )
                for o in decl_orbits
            ),
            reverse=True,
        )
        if decl_sorted != expected_sorted:
            errors.append(
                f"NFS_PAIRS: declared signed_orbits_s9 {decl_sorted} != expected {expected_sorted}"
            )

    # NFS_TEMPORAL: formula field
    temporal = cert.get("temporal_sign_formula", "")
    if not temporal or "(-1)^t" not in str(temporal) and "(-1)**t" not in str(temporal):
        warnings.append("NFS_TEMPORAL: temporal_sign_formula should reference (-1)^t")

    # NFS_WIT
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 5:
        errors.append(
            f"NFS_WIT: need >= 5 witnesses (one per orbit family), got {len(witnesses) if isinstance(witnesses, list) else 'none'}"
        )

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("nfs_pass_norm_flip.json", True),
        ("nfs_fail_wrong_orbit.json", False),
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
    parser = argparse.ArgumentParser(description="QA Norm-Flip Signed-Temporal Cert [214] validator")
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
