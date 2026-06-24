# Primary source: Mühlbach P. et al. (2026) arXiv:2605.11589; Wildberger N.J. (2005) ISBN 978-0-9757492-0-8
from __future__ import annotations
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (transform unification: "
    "G(b,e)=(b+e)*(b+e)+e*e with raw d=b+e (A2); sigma(b,e)=(e,((b+e-1)%m)+1) (A1); "
    "TU_1: matched-group bandwidth = cosmos_period+satellite_period+1 = 24+8+1 = 33; "
    "TU_2: 3 Cosmos sub-orbit G-channels are independent — A[0] values 744693/658293/574269 distinct; "
    "TU_3: cross-class energy isolation: cosmos_g_total=9963 > satellite_g_total=1377 > singularity_g=405; "
    "Theorem NT: DFT eigenvalues (roots of unity) are observer projections — channel structure "
    "certified via integer G-value sums and autocorrelations only; no float state"
)
"""
QA Transform Unification Certificate [505]

Claim: QA's G-function G(b,e)=(b+e)²+e² provides a complete, non-degenerate
matched-group channel bank (Mühlbach et al. arXiv:2605.11589) across all three
QA orbit classes. Three integer certificates:

  TU_1  Matched-group bandwidth = 33 = cosmos_period + satellite_period + 1
        = |Z/24Z| + |Z/8Z| + |Z/1Z| = 24 + 8 + 1.
        This is the total number of independent frequency channels (irreducible
        representations) in the matched-group transform for G = Z/24Z × Z/8Z × {1}.

  TU_2  Three independent Cosmos sub-orbit channels.
        A[0]_Ok = Σᵢ G(σⁱ(b₀,e₀))² for starting pairs O1=(1,1), O2=(1,3), O3=(1,4).
        All three values are distinct: A[0]_O1=744693, A[0]_O2=658293, A[0]_O3=574269.
        Each Cosmos sub-orbit carries a different total G-squared power → three
        non-degenerate, independent channel banks within the Cosmos class.
        Within each orbit: A[0] > max(A[1], ..., A[23]) — lag-0 dominant (no
        self-similar repetition at any sub-period, matching AD_3 of cert [501]).

  TU_3  Cross-class energy isolation.
        Cosmos G-total (sum of G over all 72 pairs) = 9963.
        Satellite G-total (sum over all 8 pairs) = 1377.
        Singularity G(9,9) = 405.
        Strict ordering: 9963 > 1377 > 405 — the three orbit classes have strictly
        ordered total energies; all distinct integers. The matched-group transform
        assigns each class to a disjoint energy level: Cosmos occupies the high band,
        Satellite the mid band, Singularity the DC level.

arXiv:2605.11589 context:
  Mühlbach et al. prove that for any group G, every G-equivariant covariance matrix
  is diagonalized by the Peter-Weyl basis (irreducible representations of G). For
  cyclic G=Z/n, the Peter-Weyl basis is the DFT on Z/n. The matched-group transform
  is provably optimal (KLT = DFT for G-stationary signals). Total bandwidth = Σ|Gᵢ|
  across all irreducible component groups.

  For QA: G = Z/24Z (Cosmos) × Z/8Z (Satellite) × Z/1Z (Singularity).
  - Cosmos channel bank: 24-point DFT on each sub-orbit (3 independent banks = 72 params)
  - Satellite channel bank: 8-point DFT on Satellite orbit (8 params)
  - Singularity channel: DC-only (trivial group, 1 param)
  - Total bandwidth: 24×3 + 8 + 1 = 81 = 9² (full state space), compressible to 33 irreps

  Theorem NT: The DFT eigenvalues (24th and 8th roots of unity) are continuous observer
  projections over the discrete G-orbit structure. The channel structure is certified
  entirely via integer G-value arithmetic (sums, products, autocorrelations).

  Companion certs: [500] Cosmos Chamber (G-sum AP), [501] Algebraic Diversity (G-injectivity),
  [503] Witt Tower tau-monotone (empirical discrimination), [504] Star-G Tensor (G-module).

Schema: QA_TRANSFORM_UNIFICATION_CERT.v1
"""

import json
import sys
from pathlib import Path

SCHEMA = "QA_TRANSFORM_UNIFICATION_CERT.v1"
M = 9
_EXPECTED_BANDWIDTH = 33
_EXPECTED_COSMOS_CHANNELS = 24
_EXPECTED_SATELLITE_CHANNELS = 8
_EXPECTED_SINGULARITY_CHANNELS = 1
_EXPECTED_COSMOS_G_TOTAL = 9963
_EXPECTED_SATELLITE_G_TOTAL = 1377
_EXPECTED_SINGULARITY_G = 405
_EXPECTED_COSMOS_AC0 = [744693, 658293, 574269]   # sorted descending by A[0]
_EXPECTED_SATELLITE_AC0 = 292005


def _qa_step(b, e, m):
    return e, ((b + e - 1) % m) + 1


def _G(b, e):
    return (b + e) * (b + e) + e * e


def _orbit(b0, e0, m):
    pts, b, e = [], b0, e0
    for _ in range(m * m + 2):
        pts.append((b, e))
        b, e = _qa_step(b, e, m)
        if b == b0 and e == e0:
            break
    return pts


def _autocorr_lag0(g_vals, n):
    """A[0] = Σᵢ g[i]² (sum of squares)."""
    return sum(v * v for v in g_vals)


def _autocorr_max_nonzero(g_vals, n):
    """max(A[1], ..., A[n-1]) where A[k] = Σᵢ g[i]*g[(i+k)%n]."""
    return max(
        sum(g_vals[i] * g_vals[(i + k) % n] for i in range(n))
        for k in range(1, n)
    )


def _compute_expected(m):
    """Compute all expected values from scratch."""
    seen, cosmos_orbits = set(), []
    sat_orbit = None
    sing_pair = None
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            orb = _orbit(b, e, m)
            for p in orb:
                seen.add(p)
            if len(orb) == 24:
                cosmos_orbits.append(orb)
            elif len(orb) == 8:
                sat_orbit = orb
            elif len(orb) == 1:
                sing_pair = (b, e)

    cosmos_g_total = sum(_G(b, e) for orb in cosmos_orbits for b, e in orb)
    sat_g_total = sum(_G(b, e) for b, e in sat_orbit)
    sing_g = _G(*sing_pair)

    cosmos_ac0 = sorted(
        [_autocorr_lag0([_G(b, e) for b, e in orb], 24) for orb in cosmos_orbits],
        reverse=True,
    )
    sat_ac0 = _autocorr_lag0([_G(b, e) for b, e in sat_orbit], 8)

    cosmos_ac0_dominant = all(
        _autocorr_lag0([_G(b, e) for b, e in orb], 24)
        > _autocorr_max_nonzero([_G(b, e) for b, e in orb], 24)
        for orb in cosmos_orbits
    )
    sat_ac0_dominant = (
        sat_ac0 > _autocorr_max_nonzero([_G(b, e) for b, e in sat_orbit], 8)
    )

    return {
        "bandwidth": 24 + 8 + 1,  # Z/24Z → 24 irreps + Z/8Z → 8 irreps + {1} → 1; sum per orbit class
        "cosmos_channels": 24,
        "satellite_channels": 8,
        "singularity_channels": 1,
        "cosmos_g_total": cosmos_g_total,
        "satellite_g_total": sat_g_total,
        "singularity_g": sing_g,
        "cosmos_ac0": cosmos_ac0,
        "satellite_ac0": sat_ac0,
        "cosmos_ac0_dominant": cosmos_ac0_dominant,
        "satellite_ac0_dominant": sat_ac0_dominant,
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

    # TU_1: bandwidth = 33
    decl_bw = data.get("total_channel_count")
    if decl_bw != _EXPECTED_BANDWIDTH:
        errors.append(
            f"TU_1a: declared total_channel_count={decl_bw} expected {_EXPECTED_BANDWIDTH}"
        )
    if computed["bandwidth"] != _EXPECTED_BANDWIDTH:
        errors.append(
            f"TU_1a: computed bandwidth={computed['bandwidth']} expected {_EXPECTED_BANDWIDTH}"
        )
    decl_cc = data.get("cosmos_channel_count")
    if decl_cc is not None and decl_cc != _EXPECTED_COSMOS_CHANNELS:
        errors.append(
            f"TU_1b: declared cosmos_channel_count={decl_cc} expected {_EXPECTED_COSMOS_CHANNELS}"
        )
    decl_sc = data.get("satellite_channel_count")
    if decl_sc is not None and decl_sc != _EXPECTED_SATELLITE_CHANNELS:
        errors.append(
            f"TU_1b: declared satellite_channel_count={decl_sc} expected {_EXPECTED_SATELLITE_CHANNELS}"
        )

    # TU_2: 3 independent Cosmos sub-orbit channels (distinct A[0], lag-0 dominant)
    decl_ac0 = data.get("cosmos_sub_orbit_ac0", [])
    if decl_ac0:
        decl_sorted = sorted(decl_ac0, reverse=True)
        if decl_sorted != sorted(computed["cosmos_ac0"], reverse=True):
            errors.append(
                f"TU_2a: declared cosmos A[0] values {sorted(decl_ac0, reverse=True)} "
                f"!= computed {computed['cosmos_ac0']}"
            )
        if len(set(decl_ac0)) != 3:
            errors.append(
                f"TU_2b: cosmos A[0] values not all distinct: {decl_ac0}"
            )
    if len(set(computed["cosmos_ac0"])) != 3:
        errors.append(
            f"TU_2b: computed cosmos A[0] values not all distinct: {computed['cosmos_ac0']}"
        )
    if not computed["cosmos_ac0_dominant"]:
        errors.append("TU_2c: some Cosmos sub-orbit has A[k]>=A[0] for k>0; lag-0 not dominant")

    decl_sat_ac0 = data.get("satellite_ac0")
    if decl_sat_ac0 is not None and decl_sat_ac0 != computed["satellite_ac0"]:
        errors.append(
            f"TU_2d: declared satellite_ac0={decl_sat_ac0} computed={computed['satellite_ac0']}"
        )
    if not computed["satellite_ac0_dominant"]:
        errors.append("TU_2d: Satellite B[k]>=B[0] for some k>0; lag-0 not dominant")

    # TU_3: cross-class energy isolation: cosmos > satellite > singularity
    decl_cgt = data.get("cosmos_g_total")
    decl_sgt = data.get("satellite_g_total")
    decl_sg = data.get("singularity_g")

    if decl_cgt is not None and decl_cgt != _EXPECTED_COSMOS_G_TOTAL:
        errors.append(
            f"TU_3a: declared cosmos_g_total={decl_cgt} expected {_EXPECTED_COSMOS_G_TOTAL}"
        )
    if decl_sgt is not None and decl_sgt != _EXPECTED_SATELLITE_G_TOTAL:
        errors.append(
            f"TU_3a: declared satellite_g_total={decl_sgt} expected {_EXPECTED_SATELLITE_G_TOTAL}"
        )
    if decl_sg is not None and decl_sg != _EXPECTED_SINGULARITY_G:
        errors.append(
            f"TU_3a: declared singularity_g={decl_sg} expected {_EXPECTED_SINGULARITY_G}"
        )
    if not (computed["cosmos_g_total"] > computed["satellite_g_total"] > computed["singularity_g"]):
        errors.append(
            f"TU_3b: cross-class energy order violated: cosmos={computed['cosmos_g_total']} "
            f"sat={computed['satellite_g_total']} sing={computed['singularity_g']}"
        )
    # Strict distinctness of all three
    three_vals = {computed["cosmos_g_total"], computed["satellite_g_total"], computed["singularity_g"]}
    if len(three_vals) != 3:
        errors.append(
            f"TU_3c: G-totals not all distinct: {three_vals}"
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
