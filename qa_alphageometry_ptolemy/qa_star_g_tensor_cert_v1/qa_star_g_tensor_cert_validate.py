# Primary source: Nguyen T.T. et al. (2025) arXiv:2605.20440; Wildberger N.J. (2005) ISBN 978-0-9757492-0-8
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (star-G tensor orbit module: "
    "sigma(b,e)=(e,((b+e-1)%m)+1); SGT_1: orbit type preserved under sigma for all 81 pairs "
    "(Cosmos/Satellite/Singularity = invariant G-submodules of G=Z/24Z x Z/8Z x {1}); "
    "SGT_2: CRT decomposition Z/24Z ≅ Z/3Z x Z/8Z — sigma^8 has period 3 on all Cosmos pairs; "
    "sigma^3 has period 8 on all Cosmos pairs; CRT: gcd(3,8)=1, 3*8=24 (Kronecker factorization "
    "of Cosmos sub-action per arXiv:2605.20440 Theorem 2); "
    "SGT_3: sigma^4 has period exactly 2 on all 8 Satellite pairs (Z/8Z half-period antipodal); "
    "SGT_4: sigma^24 = identity on all 81 pairs (lcm(24,8)=24 universal period); "
    "Theorem NT: continuous group characters / roots of unity are observer projections — "
    "G-module equivariance certified by integer period arithmetic only; no float state"
)
"""
QA Star-G Tensor Orbit Module Certificate [504]

Claim: QA mod-9 state space implements a star-G tensor algebra module
(Nguyen et al. arXiv:2605.20440) with G = Z/24Z x Z/8Z x {1}.
Four integer certificates:

  SGT_1  sigma preserves orbit type for all 81 pairs: Cosmos->Cosmos,
         Satellite->Satellite, Singularity->Singularity.
         (G-module direct sum; three invariant submodules.)

  SGT_2  CRT decomposition of Z/24Z: sigma^8 has period 3 on all 72
         Cosmos pairs; sigma^3 has period 8 on all 72 Cosmos pairs.
         gcd(3,8)=1 and 3*8=24 → Z/24Z ≅ Z/3Z x Z/8Z (Chinese Remainder).
         This is the Kronecker factorization (Theorem 2 of arXiv:2605.20440)
         applied to the Cosmos sub-action.

  SGT_3  Z/8Z half-period: sigma^4 has period exactly 2 on all 8
         Satellite pairs. sigma^4(3,3) = (6,6) ≠ (3,3), sigma^8(3,3) = (3,3).
         (The antipodal map on the 8-cycle.)

  SGT_4  Universal period: sigma^24 = identity on all 81 pairs.
         lcm(24,8) = 24 — the Satellite 8-cycle divides the Cosmos 24-cycle,
         so sigma^24 closes the entire state space simultaneously.

arXiv:2605.20440 context:
  The star-G algebra certifies that any G-equivariant tensor decomposes
  optimally under G-SVD (Eckart-Young Theorem 1) and that product symmetries
  G1 x G2 factor as a Kronecker product (Theorem 2). The 600-line Lean 4
  formalization (§4 of arXiv:2605.20440) extends cert [128]'s pi(9)=24 Lean
  infrastructure into the full orbit-module algebra.

  For QA: G = Z/24Z (Cosmos) x Z/8Z (Satellite) x {1} (Singularity).
  SGT_1 certifies the direct-sum submodule structure.
  SGT_2 certifies the Kronecker CRT factorization of the Cosmos factor.
  SGT_3 certifies the Z/8Z half-period sub-action on the Satellite factor.
  SGT_4 certifies that sigma^|G_max| = identity (|G_max| = lcm(24,8) = 24).

  Theorem NT: continuous group characters (24th/8th roots of unity) are
  observer projections. The equivariance claim is certified entirely via
  integer period arithmetic.

Schema: QA_STAR_G_TENSOR_CERT.v1
"""

import json
import math
import sys
from pathlib import Path

SCHEMA = "QA_STAR_G_TENSOR_CERT.v1"
M = 9
_EXPECTED_COSMOS_PERIOD = 24
_EXPECTED_SATELLITE_PERIOD = 8
_EXPECTED_SIGMA8_COSMOS_PERIOD = 3    # CRT: sigma^8 has order 3 in Z/24Z
_EXPECTED_SIGMA3_COSMOS_PERIOD = 8    # CRT: sigma^3 has order 8 in Z/24Z
_EXPECTED_CRT_FACTOR_A = 3
_EXPECTED_CRT_FACTOR_B = 8
_EXPECTED_CRT_GCD = 1                 # gcd(3,8) = 1 → coprime → CRT applies
_EXPECTED_SIGMA4_SAT_PERIOD = 2       # half-period antipodal on 8-cycle
_EXPECTED_SGT1_VIOLATIONS = 0
_EXPECTED_SGT4_VIOLATIONS = 0
_EXPECTED_LCM = 24                    # lcm(24,8)
_WITNESS_PAIR = [1, 1]
_SIGMA8_IMAGE = [7, 1]
_SIGMA16_IMAGE = [4, 1]
_SIGMA24_IMAGE = [1, 1]
_SAT_WITNESS_PAIR = [3, 3]
_SIGMA4_SAT_IMAGE = [6, 6]
_SIGMA8_SAT_IMAGE = [3, 3]


def _qa_step(b, e, m):
    return e, ((b + e - 1) % m) + 1


def _sigma_k(b, e, m, k):
    for _ in range(k):
        b, e = _qa_step(b, e, m)
    return b, e


def _qa_period(b0, e0, m):
    b, e = b0, e0
    for k in range(1, m * m + 2):
        b, e = _qa_step(b, e, m)
        if b == b0 and e == e0:
            return k
    raise RuntimeError(f"No period for ({b0},{e0}) mod {m}")


def _period_of_sigma_k(b0, e0, m, k):
    """Period of the iterated map x -> sigma^k(x) starting at (b0,e0)."""
    b, e = b0, e0
    for p in range(1, 200):
        b, e = _sigma_k(b, e, m, k)
        if b == b0 and e == e0:
            return p
    raise RuntimeError(f"Period of sigma^{k} not found at ({b0},{e0})")


def _orbit_type(b, e, m):
    p = _qa_period(b, e, m)
    if p == 1:
        return "S"
    if p == 8:
        return "SAT"
    if p == 24:
        return "COS"
    return f"?{p}"


def _check_fixture(data):
    errors = []

    if data.get("schema_version") != SCHEMA:
        errors.append(
            f"SRC: expected schema_version={SCHEMA!r}, got {data.get('schema_version')!r}"
        )

    m = data.get("modulus", M)
    if m != M:
        errors.append(f"MOD: expected modulus={M}, got {m}")

    # SGT_1: orbit type preserved — declared violation count == 0
    declared_sgt1 = data.get("sgt1_orbit_type_violations")
    computed_sgt1 = sum(
        1
        for b in range(1, m + 1)
        for e in range(1, m + 1)
        if _orbit_type(b, e, m) != _orbit_type(*_qa_step(b, e, m), m)
    )
    if declared_sgt1 != computed_sgt1:
        errors.append(
            f"SGT_1: declared sgt1_orbit_type_violations={declared_sgt1} "
            f"but computed={computed_sgt1}"
        )
    if computed_sgt1 != _EXPECTED_SGT1_VIOLATIONS:
        errors.append(
            f"SGT_1: expected 0 orbit-type violations, computed={computed_sgt1}"
        )

    # SGT_2: CRT decomposition — sigma^8 period 3, sigma^3 period 8 on Cosmos pairs
    declared_s8p = data.get("sgt2_sigma8_cosmos_period")
    declared_s3p = data.get("sgt2_sigma3_cosmos_period")
    crt = data.get("sgt2_crt", {})
    declared_fa = crt.get("factor_a")
    declared_fb = crt.get("factor_b")
    declared_gcd = crt.get("gcd_a_b")

    s8_periods = set()
    s3_periods = set()
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if _orbit_type(b, e, m) == "COS":
                s8_periods.add(_period_of_sigma_k(b, e, m, 8))
                s3_periods.add(_period_of_sigma_k(b, e, m, 3))

    computed_s8p = next(iter(s8_periods)) if len(s8_periods) == 1 else None
    computed_s3p = next(iter(s3_periods)) if len(s3_periods) == 1 else None

    if declared_s8p != _EXPECTED_SIGMA8_COSMOS_PERIOD:
        errors.append(
            f"SGT_2a: declared sigma8_cosmos_period={declared_s8p} "
            f"expected {_EXPECTED_SIGMA8_COSMOS_PERIOD}"
        )
    if computed_s8p != _EXPECTED_SIGMA8_COSMOS_PERIOD:
        errors.append(
            f"SGT_2a: computed sigma^8 periods on Cosmos={s8_periods} "
            f"expected {{{_EXPECTED_SIGMA8_COSMOS_PERIOD}}}"
        )
    if declared_s3p != _EXPECTED_SIGMA3_COSMOS_PERIOD:
        errors.append(
            f"SGT_2b: declared sigma3_cosmos_period={declared_s3p} "
            f"expected {_EXPECTED_SIGMA3_COSMOS_PERIOD}"
        )
    if computed_s3p != _EXPECTED_SIGMA3_COSMOS_PERIOD:
        errors.append(
            f"SGT_2b: computed sigma^3 periods on Cosmos={s3_periods} "
            f"expected {{{_EXPECTED_SIGMA3_COSMOS_PERIOD}}}"
        )

    # CRT: gcd(factor_a, factor_b) == 1 and factor_a * factor_b == cosmos_period
    if declared_fa is not None and declared_fb is not None:
        if declared_fa * declared_fb != _EXPECTED_COSMOS_PERIOD:
            errors.append(
                f"SGT_2c: factor_a*factor_b={declared_fa}*{declared_fb}="
                f"{declared_fa*declared_fb} != {_EXPECTED_COSMOS_PERIOD}"
            )
        if declared_gcd != math.gcd(declared_fa, declared_fb):
            errors.append(
                f"SGT_2c: declared gcd_a_b={declared_gcd} != "
                f"math.gcd({declared_fa},{declared_fb})={math.gcd(declared_fa, declared_fb)}"
            )
        if math.gcd(declared_fa, declared_fb) != _EXPECTED_CRT_GCD:
            errors.append(
                f"SGT_2c: gcd({declared_fa},{declared_fb})={math.gcd(declared_fa,declared_fb)} "
                f"!= 1; CRT requires coprime factors"
            )

    # Witness: sigma^8(1,1), sigma^16(1,1), sigma^24(1,1)
    declared_w = data.get("sgt2_witness_pair", [])
    declared_s8i = data.get("sgt2_sigma8_image", [])
    declared_s16i = data.get("sgt2_sigma16_image", [])
    declared_s24i = data.get("sgt2_sigma24_image", [])
    if declared_w == _WITNESS_PAIR:
        wb, we = _WITNESS_PAIR
        comp_s8 = list(_sigma_k(wb, we, m, 8))
        comp_s16 = list(_sigma_k(wb, we, m, 16))
        comp_s24 = list(_sigma_k(wb, we, m, 24))
        if declared_s8i and declared_s8i != comp_s8:
            errors.append(f"SGT_2w: sgt2_sigma8_image={declared_s8i} but computed={comp_s8}")
        if declared_s16i and declared_s16i != comp_s16:
            errors.append(f"SGT_2w: sgt2_sigma16_image={declared_s16i} but computed={comp_s16}")
        if declared_s24i and declared_s24i != comp_s24:
            errors.append(f"SGT_2w: sgt2_sigma24_image={declared_s24i} but computed={comp_s24}")
        if comp_s8 == [wb, we]:
            errors.append(
                f"SGT_2w: sigma^8({wb},{we})=({wb},{we}); witness must show sigma^8 != id"
            )
        if comp_s24 != [wb, we]:
            errors.append(
                f"SGT_2w: sigma^24({wb},{we})={comp_s24} != ({wb},{we}); "
                "witness must show sigma^24 = id"
            )

    # SGT_3: sigma^4 has period exactly 2 on all Satellite pairs
    declared_s4p = data.get("sgt3_sigma4_satellite_period")
    s4_periods = set()
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if _orbit_type(b, e, m) == "SAT":
                s4_periods.add(_period_of_sigma_k(b, e, m, 4))

    computed_s4p = next(iter(s4_periods)) if len(s4_periods) == 1 else None
    if declared_s4p != _EXPECTED_SIGMA4_SAT_PERIOD:
        errors.append(
            f"SGT_3a: declared sgt3_sigma4_satellite_period={declared_s4p} "
            f"expected {_EXPECTED_SIGMA4_SAT_PERIOD}"
        )
    if computed_s4p != _EXPECTED_SIGMA4_SAT_PERIOD:
        errors.append(
            f"SGT_3a: computed sigma^4 periods on Satellite={s4_periods} "
            f"expected {{{_EXPECTED_SIGMA4_SAT_PERIOD}}}"
        )

    # Witness: sigma^4(3,3), sigma^8(3,3)
    declared_sw = data.get("sgt3_witness_pair", [])
    declared_s4si = data.get("sgt3_sigma4_image", [])
    declared_s8si = data.get("sgt3_sigma8_image", [])
    if declared_sw == _SAT_WITNESS_PAIR:
        swb, swe = _SAT_WITNESS_PAIR
        comp_s4s = list(_sigma_k(swb, swe, m, 4))
        comp_s8s = list(_sigma_k(swb, swe, m, 8))
        if declared_s4si and declared_s4si != comp_s4s:
            errors.append(f"SGT_3w: sgt3_sigma4_image={declared_s4si} but computed={comp_s4s}")
        if declared_s8si and declared_s8si != comp_s8s:
            errors.append(f"SGT_3w: sgt3_sigma8_image={declared_s8si} but computed={comp_s8s}")
        if comp_s4s == [swb, swe]:
            errors.append(
                f"SGT_3w: sigma^4({swb},{swe})=({swb},{swe}); "
                "antipodal witness must show sigma^4 != id"
            )
        if comp_s8s != [swb, swe]:
            errors.append(
                f"SGT_3w: sigma^8({swb},{swe})={comp_s8s} != ({swb},{swe})"
            )

    # SGT_4: sigma^24 = identity on all 81 pairs
    declared_sgt4 = data.get("sgt4_sigma24_total_violations")
    declared_lcm = data.get("sgt4_lcm_cosmos_satellite")
    computed_sgt4 = sum(
        1
        for b in range(1, m + 1)
        for e in range(1, m + 1)
        if list(_sigma_k(b, e, m, 24)) != [b, e]
    )
    computed_lcm = math.lcm(_EXPECTED_COSMOS_PERIOD, _EXPECTED_SATELLITE_PERIOD)
    if declared_sgt4 != computed_sgt4:
        errors.append(
            f"SGT_4a: declared sgt4_sigma24_total_violations={declared_sgt4} "
            f"but computed={computed_sgt4}"
        )
    if computed_sgt4 != _EXPECTED_SGT4_VIOLATIONS:
        errors.append(
            f"SGT_4a: sigma^24 has {computed_sgt4} non-fixed pairs; expected 0"
        )
    if declared_lcm is not None and declared_lcm != computed_lcm:
        errors.append(
            f"SGT_4b: declared lcm={declared_lcm} but computed lcm(24,8)={computed_lcm}"
        )
    if declared_lcm is not None and declared_lcm != _EXPECTED_LCM:
        errors.append(
            f"SGT_4b: declared lcm={declared_lcm} expected {_EXPECTED_LCM}"
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
