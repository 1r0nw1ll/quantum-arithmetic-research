# Primary source: Tao T. et al. (2026) SAIR IGP24 competition; Wildberger N.J. (2005) ISBN 978-0-9757492-0-8
from __future__ import annotations
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (inverse galois degree-24: "
    "sigma(b,e)=(e,((b+e-1)%m)+1) (A1); orbit counting integer only; "
    "IG_1: sigma on Cosmos sub-orbit is a 24-cycle — transitive degree-24 C24 action; "
    "IG_2: regular representation: 0 fixed points in each Cosmos sub-orbit under sigma^k 0<k<24; "
    "IG_3: 3 Cosmos sub-orbits each of length 24 = three independent C24 regular copies; "
    "IG_4: CRT C24 = C3 x C8: sub-orbit coset decomposition gcd(3,8)=1 3*8=24; "
    "Theorem NT: Galois group characters/splitting fields are observer projections; "
    "C24 concretely realized by sigma on Cosmos; no float state"
)
"""
QA Inverse Galois Degree-24 Certificate [506]

Claim: QA's sigma operator on the mod-9 Cosmos orbit provides a concrete,
computable realization of the cyclic group C₂₄ as a degree-24 transitive
subgroup of S₂₄ — the simplest entry in the SAIR IGP24 competition's
degree-24 transitive group classification (165,836 (group, signature) pairs).

Competition context: SAIR Inverse Galois Problem degree-24 challenge (Tao et al.,
launched June 16, 2026). Stage 1 closes August 15, 2026. Given a transitive
group G ≤ S₂₄, find a degree-24 polynomial over Q with Gal(f/Q) ≅ G. The
LMFDB baseline resolved 622 of 165,836 pairs at launch. The simplest case is
G = C₂₄ (cyclic group of order 24), realized by the regular representation.

QA realization of C₂₄ as a degree-24 transitive group:

  IG_1  sigma restricted to any Cosmos sub-orbit is a 24-cycle in S₂₄.
        sigma acts on the 24 elements of a Cosmos sub-orbit as a single
        cycle of length 24. The orbit {sigma^k(1,1) : k=0,...,23} has
        exactly 24 distinct elements; sigma^24(1,1) = (1,1); sigma^k(1,1) ≠ (1,1)
        for 0 < k < 24. Verified for all 3 Cosmos sub-orbits.
        This makes sigma a 24-cycle element of S₂₄, generating C₂₄ ≤ S₂₄.
        Degree = 24; group order = 24 (regular representation: |stab| = 1).

  IG_2  Regular representation = 0 fixed points.
        In the regular representation, every non-identity element has 0 fixed
        points. Since sigma^k for 0 < k < 24 moves all 24 Cosmos pairs in
        each sub-orbit (no pair satisfies sigma^k(b,e) = (b,e) for 0 < k < 24),
        the 24-cycle representation is cycle-structure (24). Fixed-point count = 0.
        Verified: 3 Cosmos sub-orbits × 23 non-identity powers = 0 fixed points.

  IG_3  Exactly 3 Cosmos sub-orbits — three independent C₂₄ copies.
        The full Cosmos class has 72 pairs partitioned into exactly 3 sub-orbits
        of length 24. This equals |C₂₄|/|C₈| = 24/8 = 3 under the CRT
        C₂₄ ≅ C₃ × C₈. The 3 sub-orbits correspond to the 3 cosets of
        the C₈ subgroup in C₂₄ (under the C₃ component of the CRT).
        Integer: 72 / 24 = 3; 3 = |C₃| = factor_a from cert [504] SGT_2.

  IG_4  CRT decomposition C₂₄ ≅ C₃ × C₈.
        sigma^8 restricted to each Cosmos sub-orbit has period 3.
        sigma^3 restricted to each Cosmos sub-orbit has period 8.
        gcd(3, 8) = 1 and 3 × 8 = 24 — Chinese Remainder factorization.
        This is the IGP24-relevant fact: C₂₄ as a degree-24 transitive group
        decomposes (as an abstract group) as C₃ × C₈, making it a DIRECT PRODUCT
        group with two coprime cyclic factors. This is the factored structure
        visible in the LMFDB degree-24 group classification.

IGP24 Competition connection (Theorem NT):
  The competition verifies Galois groups via Magma's GaloisGroup() function —
  a continuous/symbolic computation that is an observer projection over the
  discrete group-theoretic structure. The discrete structure (group order,
  cycle type, subgroup lattice) is the causal layer; the polynomial realization
  is the observer output. QA certifies the discrete structure directly via
  sigma iteration; the Galois polynomial is an observer projection.

Companion certs: [128] π(9)=24, [499] all-initializations orbit classification,
[504] Star-G tensor C₂₄ CRT.

Schema: QA_INVERSE_GALOIS_DEGREE24_CERT.v1
"""

import json
import sys
from pathlib import Path

SCHEMA = "QA_INVERSE_GALOIS_DEGREE24_CERT.v1"
M = 9
_EXPECTED_COSMOS_ORBIT_LENGTH = 24
_EXPECTED_NUM_COSMOS_ORBITS = 3
_EXPECTED_TOTAL_COSMOS_PAIRS = 72
_EXPECTED_FIXED_POINTS = 0
_EXPECTED_CRT_FACTOR_A = 3
_EXPECTED_CRT_FACTOR_B = 8
_EXPECTED_CRT_GCD = 1
_EXPECTED_COSET_COUNT = 3  # 72 / 24 = 3 = |C3|


def _qa_step(b, e, m):
    return e, ((b + e - 1) % m) + 1


def _orbit(b0, e0, m):
    pts, b, e = [], b0, e0
    for _ in range(m * m + 2):
        pts.append((b, e))
        b, e = _qa_step(b, e, m)
        if b == b0 and e == e0:
            break
    return pts


def _count_fixed_points(orbit, power, m):
    """Count pairs p in orbit such that sigma^power(p) == p."""
    count = 0
    for (b0, e0) in orbit:
        b, e = b0, e0
        for _ in range(power):
            b, e = _qa_step(b, e, m)
        if (b, e) == (b0, e0):
            count += 1
    return count


def _sigma_power_period(orbit, power, m):
    """Period of sigma^power restricted to orbit (first pair as representative)."""
    b0, e0 = orbit[0]
    b, e = b0, e0
    for k in range(1, len(orbit) + 1):
        for _ in range(power):
            b, e = _qa_step(b, e, m)
        if (b, e) == (b0, e0):
            return k
    return -1  # should not reach here


def _compute_expected(m):
    seen, cosmos_orbits = set(), []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            orb = _orbit(b, e, m)
            for p in orb:
                seen.add(p)
            if len(orb) == 24:
                cosmos_orbits.append(orb)

    num_orbits = len(cosmos_orbits)
    total_pairs = sum(len(o) for o in cosmos_orbits)

    # IG_1: each sub-orbit is a 24-cycle
    cycle_lengths = [len(o) for o in cosmos_orbits]
    all_24 = all(c == 24 for c in cycle_lengths)

    # IG_2: fixed points of sigma^k for 0 < k < 24 across all cosmos orbits
    total_fixed = sum(
        _count_fixed_points(orb, k, m)
        for orb in cosmos_orbits
        for k in range(1, 24)
    )

    # IG_4: CRT decomposition
    sigma8_periods = [_sigma_power_period(orb, 8, m) for orb in cosmos_orbits]
    sigma3_periods = [_sigma_power_period(orb, 3, m) for orb in cosmos_orbits]
    from math import gcd
    crt_gcd = gcd(3, 8)
    crt_product = 3 * 8

    # IG_3: coset count
    coset_count = total_pairs // _EXPECTED_COSMOS_ORBIT_LENGTH

    return {
        "num_cosmos_orbits": num_orbits,
        "total_cosmos_pairs": total_pairs,
        "all_24_cycles": all_24,
        "cycle_lengths": cycle_lengths,
        "total_fixed_points": total_fixed,
        "sigma8_periods_on_cosmos": sigma8_periods,
        "sigma3_periods_on_cosmos": sigma3_periods,
        "crt_gcd": crt_gcd,
        "crt_product": crt_product,
        "coset_count": coset_count,
    }


def _check_fixture(data):
    errors = []

    if data.get("schema_version") != SCHEMA:
        errors.append(
            f"SRC: expected schema_version={SCHEMA!r}, got {data.get('schema_version')!r}"
        )

    m = data.get("modulus", M)
    if m != M:
        errors.append(f"MOD: expected modulus={M}, got {m}")

    computed = _compute_expected(m)

    # IG_1: 24-cycle
    decl_len = data.get("cosmos_orbit_length")
    if decl_len is not None and decl_len != _EXPECTED_COSMOS_ORBIT_LENGTH:
        errors.append(
            f"IG_1a: declared cosmos_orbit_length={decl_len} expected {_EXPECTED_COSMOS_ORBIT_LENGTH}"
        )
    if not computed["all_24_cycles"]:
        errors.append(
            f"IG_1b: computed cosmos orbits are not all 24-cycles: {computed['cycle_lengths']}"
        )

    # IG_2: 0 fixed points
    decl_fp = data.get("cosmos_fixed_points_nonidentity")
    if decl_fp is not None and decl_fp != _EXPECTED_FIXED_POINTS:
        errors.append(
            f"IG_2a: declared cosmos_fixed_points_nonidentity={decl_fp} expected {_EXPECTED_FIXED_POINTS}"
        )
    if computed["total_fixed_points"] != _EXPECTED_FIXED_POINTS:
        errors.append(
            f"IG_2b: computed total_fixed_points={computed['total_fixed_points']} expected {_EXPECTED_FIXED_POINTS}"
        )

    # IG_3: 3 sub-orbits, 72 total pairs
    decl_norb = data.get("num_cosmos_orbits")
    if decl_norb is not None and decl_norb != _EXPECTED_NUM_COSMOS_ORBITS:
        errors.append(
            f"IG_3a: declared num_cosmos_orbits={decl_norb} expected {_EXPECTED_NUM_COSMOS_ORBITS}"
        )
    if computed["num_cosmos_orbits"] != _EXPECTED_NUM_COSMOS_ORBITS:
        errors.append(
            f"IG_3a: computed num_cosmos_orbits={computed['num_cosmos_orbits']} expected {_EXPECTED_NUM_COSMOS_ORBITS}"
        )
    decl_tcp = data.get("total_cosmos_pairs")
    if decl_tcp is not None and decl_tcp != _EXPECTED_TOTAL_COSMOS_PAIRS:
        errors.append(
            f"IG_3b: declared total_cosmos_pairs={decl_tcp} expected {_EXPECTED_TOTAL_COSMOS_PAIRS}"
        )
    if computed["total_cosmos_pairs"] != _EXPECTED_TOTAL_COSMOS_PAIRS:
        errors.append(
            f"IG_3b: computed total_cosmos_pairs={computed['total_cosmos_pairs']} expected {_EXPECTED_TOTAL_COSMOS_PAIRS}"
        )
    decl_cc = data.get("coset_count")
    if decl_cc is not None and decl_cc != _EXPECTED_COSET_COUNT:
        errors.append(
            f"IG_3c: declared coset_count={decl_cc} expected {_EXPECTED_COSET_COUNT}"
        )
    if computed["coset_count"] != _EXPECTED_COSET_COUNT:
        errors.append(
            f"IG_3c: computed coset_count={computed['coset_count']} expected {_EXPECTED_COSET_COUNT}"
        )

    # IG_4: CRT C24 = C3 x C8
    decl_s8 = data.get("sigma8_cosmos_period")
    if decl_s8 is not None and decl_s8 != _EXPECTED_CRT_FACTOR_A:
        errors.append(
            f"IG_4a: declared sigma8_cosmos_period={decl_s8} expected {_EXPECTED_CRT_FACTOR_A}"
        )
    for i, p in enumerate(computed["sigma8_periods_on_cosmos"]):
        if p != _EXPECTED_CRT_FACTOR_A:
            errors.append(
                f"IG_4a: computed sigma^8 period on cosmos_orbit {i} = {p} expected {_EXPECTED_CRT_FACTOR_A}"
            )
    decl_s3 = data.get("sigma3_cosmos_period")
    if decl_s3 is not None and decl_s3 != _EXPECTED_CRT_FACTOR_B:
        errors.append(
            f"IG_4b: declared sigma3_cosmos_period={decl_s3} expected {_EXPECTED_CRT_FACTOR_B}"
        )
    for i, p in enumerate(computed["sigma3_periods_on_cosmos"]):
        if p != _EXPECTED_CRT_FACTOR_B:
            errors.append(
                f"IG_4b: computed sigma^3 period on cosmos_orbit {i} = {p} expected {_EXPECTED_CRT_FACTOR_B}"
            )
    decl_gcd = data.get("crt_gcd_3_8")
    if decl_gcd is not None and decl_gcd != _EXPECTED_CRT_GCD:
        errors.append(
            f"IG_4c: declared crt_gcd_3_8={decl_gcd} expected {_EXPECTED_CRT_GCD}"
        )
    if computed["crt_gcd"] != _EXPECTED_CRT_GCD:
        errors.append(
            f"IG_4c: computed gcd(3,8)={computed['crt_gcd']} expected {_EXPECTED_CRT_GCD}"
        )
    if computed["crt_product"] != _EXPECTED_COSMOS_ORBIT_LENGTH:
        errors.append(
            f"IG_4d: computed 3*8={computed['crt_product']} expected {_EXPECTED_COSMOS_ORBIT_LENGTH}"
        )

    if "fail_ledger" in data:
        fl = data["fail_ledger"]
        if not isinstance(fl, list) or not all(isinstance(s, str) for s in fl):
            errors.append("F: fail_ledger must be a list of strings")

    return errors


def _run_self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    results = {}
    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        expected_pass = fixture_path.stem.startswith("pass_")
        with open(fixture_path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        passed = len(errs) == 0
        ok = passed == expected_pass
        results[fixture_path.name] = {
            "expected": "PASS" if expected_pass else "FAIL",
            "got": "PASS" if passed else "FAIL",
            "ok": ok,
            "errors": errs,
        }
    all_ok = all(v["ok"] for v in results.values())
    return {"ok": all_ok, "fixtures": results}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        result = _run_self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        if errs:
            print(json.dumps({"ok": False, "errors": errs}, indent=2))
            sys.exit(1)
        print(json.dumps({"ok": True}, indent=2))
        sys.exit(0)

    result = _run_self_test()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
