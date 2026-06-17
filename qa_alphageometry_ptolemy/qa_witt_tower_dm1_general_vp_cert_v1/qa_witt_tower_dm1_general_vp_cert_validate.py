#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=det=-1 companion matrix Witt tower lifting, p-adic valuation of discriminant t^2+4; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 ch.1-3 (ramification in p-adic fields, Witt vectors); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 ch.5,7 (quadratic residues, Hensel lifting, nilpotent mod p^k); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano-period structure for quadratic recurrences) -->
"""QA Witt Tower det=-1 General v_p Period Law Cert [440].

TWIN of [439] for the det=-1 companion-matrix family.

For the det=-1 companion matrix M=[[t,1],[1,0]] (char poly x^2-tx-1,
det(M)=-1), the orbit structure on (Z/p^k Z)^2 under v_p(t^2+4)=r is:

  STRUCTURAL CONSTANT: count(1) = 1 for ALL (p,t,k).
    Proof: M*x=x requires t*a=0 mod p^k (with a=b). Since p|t^2+4
    forces p not|t (for odd p), this gives a=0=b. Only zero is fixed.

  PERIOD SET: {1} ∪ {4} ∪ {4*p^L : L=1..k}  (periods 2 and other odd
    multiples never appear — the base cycle length is forced to 4).

  count(4) = (p^min(r,k) - 1) / 4
    = the number of period-4 orbits; grows until k=r, then saturates.

  k <= r (within ramification depth):
    count(4*p^L) = (p-1)/4 * p^(k-1)   for L=1..k   (k joint birth layers)

  k > r (beyond ramification depth):
    count(4*p^L) = (p^2-1)/4 * p^(L+r-2)  for L=1..k-r   (frozen)
    count(4*p^L) = (p-1)/4 * p^(k-1)       for L=k-r+1..k (r joint birth layers)

ALGEBRAIC MECHANISM:
  Let K = t*I - 2*M = [[-t,-2],[-2,t]]. Then det(K) = -(t^2+4) = -p^r*c_r
  (exact integer identity). The period-4 equation M^4*x=x reduces (using
  M^2=tM+I, M^4=(t^3+2t)M+(t^2+1)I) to K*x=0 mod p^k.
  Therefore |ker(K mod p^k)| = p^min(r,k), matching the Witt tower pattern.

COMPARISON WITH [439] (det=+1 family):
  det=+1 (v_p(t-2)=r): count(1)=p^min(r,k); frozen=(p^2-1)*p^(L+r-2); birth=(p-1)*p^(k-1)
  det=-1 (v_p(t^2+4)=r): count(4)=(p^min(r,k)-1)/4; frozen=(p^2-1)/4*p^(L+r-2); birth=(p-1)/4*p^(k-1)
  ALL non-trivial counts are divided by 4: the base cycle e=4 dilutes every
  orbit multiplicity by the factor 4. The structure is identical otherwise.

APPLICABILITY: p ≡ 1 mod 4 REQUIRED. For p ≡ 3 mod 4, t^2+4 ≡ t^2 + 1 mod p
  has no solution (since -4 is a non-residue), so the det=-1 family admits
  no ramified primes at those p. This is NOT an exception — it is a structural
  impossibility, not a stall analogue.

6 checks PASS; 7/7 fixtures PASS. Derived 2026-06-17.
"""

import json
import sys

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _vp(n, p):
    """Return v_p(n): p-adic valuation of n."""
    if n == 0:
        return 10**9
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def _orbit_counts_dm1(t, p, k):
    """Return {period: orbit_count} for det=-1 companion M=[[t,1],[1,0]] on (Z/p^k Z)^2."""
    m = p ** k
    period_of = {}
    for b in range(m):
        for e in range(m):
            if (b, e) in period_of:
                continue
            x = (b, e)
            path = []
            path_idx = {}
            while x not in path_idx:
                path_idx[x] = len(path)
                path.append(x)
                x = ((t * x[0] + x[1]) % m, x[0])
            period = len(path) - path_idx[x]
            for elem in path:
                period_of[elem] = period
    elem_counts = {}
    for per in period_of.values():
        elem_counts[per] = elem_counts.get(per, 0) + 1
    return {per: cnt // per for per, cnt in elem_counts.items()}


def _expected_dm1(p, r, k):
    """Expected orbit counts for det=-1, v_p(t^2+4)=r on (Z/p^k Z)^2."""
    exp = {1: 1}
    m = min(r, k)
    exp[4] = (p ** m - 1) // 4
    birth = (p - 1) // 4 * p ** (k - 1)
    if k <= r:
        for L in range(1, k + 1):
            exp[4 * p ** L] = birth
    else:
        for L in range(1, k - r + 1):
            exp[4 * p ** L] = (p * p - 1) // 4 * p ** (L + r - 2)
        for L in range(k - r + 1, k + 1):
            exp[4 * p ** L] = birth
    return exp


def _ker_size_dm1(t, p, k):
    """Brute-force |ker(K mod p^k)| where K = tI-2M = [[-t,-2],[-2,t]]."""
    m = p ** k
    count = 0
    for a in range(m):
        for b in range(m):
            if (-t * a - 2 * b) % m == 0 and (-2 * a + t * b) % m == 0:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_c1_period_set(results):
    """C1 DM1_PERIOD_SET: period set={1,4,4p^L:L=1..k} for p=5, r=1,2, k=1..4."""
    failures = 0
    for p in [5]:
        for r in [1, 2]:
            for c_r in range(1, p):
                t = None
                for cand in range(1, p ** (r + 1)):
                    if _vp(cand * cand + 4, p) == r:
                        q = (cand * cand + 4) // p ** r
                        if q % p == c_r % p:
                            t = cand
                            break
                if t is None:
                    continue
                for k in range(1, 5):
                    counts = _orbit_counts_dm1(t, p, k)
                    expected_periods = {1, 4} | {4 * p ** L for L in range(1, k + 1)}
                    if set(counts.keys()) != expected_periods:
                        failures += 1
    result = {"check": "C1_DM1_PERIOD_SET", "failures": failures,
              "ok": failures == 0}
    results.append(result)
    return result["ok"]


def _check_c2_count4_saturation(results):
    """C2 DM1_COUNT4_SATURATION: count(4)=(p^min(r,k)-1)/4; p=5,13, r=1..3, k=1..4."""
    failures = 0
    cases = []
    for r in range(1, 4):
        # p=5: find t with v_5(t^2+4)=r
        for t in range(1, 5 ** (r + 1)):
            if _vp(t * t + 4, 5) == r:
                cases.append((5, r, t))
                break
        # p=13: find t with v_13(t^2+4)=r
        for t in range(1, 13 ** (r + 1)):
            if _vp(t * t + 4, 13) == r:
                cases.append((13, r, t))
                break
    for p, r, t in cases:
        max_k = 4 if p == 5 else 2
        for k in range(1, max_k + 1):
            counts = _orbit_counts_dm1(t, p, k)
            expected_4 = (p ** min(r, k) - 1) // 4
            if counts.get(4, 0) != expected_4:
                failures += 1
    result = {"check": "C2_DM1_COUNT4_SATURATION", "failures": failures,
              "ok": failures == 0}
    results.append(result)
    return result["ok"]


def _check_c3_unified_formula(results):
    """C3 DM1_UNIFIED_FORMULA: full orbit law for p=5,13, r=1..3, k=1..4/2."""
    failures = 0
    cases = []
    for r in range(1, 4):
        for t in range(1, 5 ** (r + 1)):
            if _vp(t * t + 4, 5) == r:
                cases.append((5, r, t, 4))
                break
        for t in range(1, 13 ** (r + 1)):
            if _vp(t * t + 4, 13) == r:
                cases.append((13, r, t, 2))
                break
    for p, r, t, max_k in cases:
        for k in range(1, max_k + 1):
            counts = _orbit_counts_dm1(t, p, k)
            expected = _expected_dm1(p, r, k)
            if counts != expected:
                failures += 1
    result = {"check": "C3_DM1_UNIFIED_FORMULA", "failures": failures,
              "ok": failures == 0}
    results.append(result)
    return result["ok"]


def _check_c4_fixed_always_1(results):
    """C4 DM1_FIXED_ALWAYS_1: count(1)=1 for all t with p|t^2+4, p=5,13, k=1..3."""
    failures = 0
    for p in [5, 13]:
        for t in range(1, p * (p - 1) // 2 + p):
            if _vp(t * t + 4, p) == 0:
                continue
            for k in range(1, 4):
                if p ** k > 2200:
                    break
                counts = _orbit_counts_dm1(t, p, k)
                if counts.get(1, 0) != 1:
                    failures += 1
    result = {"check": "C4_DM1_FIXED_ALWAYS_1", "failures": failures,
              "ok": failures == 0}
    results.append(result)
    return result["ok"]


def _check_c5_algebraic_ker(results):
    """C5 DM1_ALGEBRAIC_KER: ker(tI-2M mod p^k)=p^min(r,k) for t=1..100, p=5,13, k=1..3."""
    failures = 0
    for p in [5, 13]:
        for t in range(1, 101):
            v = _vp(t * t + 4, p)
            if v == 0 or v > 4:
                continue
            r = v
            for k in range(1, 4):
                if p ** k > 2200:
                    break
                ker = _ker_size_dm1(t, p, k)
                expected = p ** min(r, k)
                if ker != expected:
                    failures += 1
    result = {"check": "C5_DM1_ALGEBRAIC_KER", "failures": failures,
              "ok": failures == 0}
    results.append(result)
    return result["ok"]


def _check_c6_p3mod4_no_ramification(results):
    """C6 DM1_NO_RAMIFIED_P3MOD4: p≡3 mod 4 admits no t with p|t^2+4."""
    failures = 0
    # Check p=7,11,19,23 (all ≡3 mod 4)
    for p in [7, 11, 19, 23]:
        for t in range(0, p):
            if (t * t + 4) % p == 0:
                failures += 1  # Should never happen
    result = {"check": "C6_DM1_NO_RAMIFIED_P3MOD4", "failures": failures,
              "ok": failures == 0}
    results.append(result)
    return result["ok"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = [
    # FIX1: p=5, r=1 (t=1), k=3: full orbit distribution
    {
        "id": "FIX1_P5_R1_K3",
        "desc": "p=5, r=1, t=1, k=3: {1:1, 4:1, 20:6, 100:30, 500:25}",
        "fn": lambda: _orbit_counts_dm1(1, 5, 3) == {1: 1, 4: 1, 20: 6, 100: 30, 500: 25},
    },
    # FIX2: p=5, r=2 (t=14), k=3 (k>r): frozen + 2 joint birth layers
    {
        "id": "FIX2_P5_R2_K3",
        "desc": "p=5, r=2, t=14, k=3: count(4)=6, count(20)=30 (frozen), count(100)=count(500)=25 (joint birth)",
        "fn": lambda: _orbit_counts_dm1(14, 5, 3) == {1: 1, 4: 6, 20: 30, 100: 25, 500: 25},
    },
    # FIX3: p=5, r=3 (t=11), k=3 (k=r): all joint birth, count(4) saturated
    {
        "id": "FIX3_P5_R3_K3",
        "desc": "p=5, r=3, t=11, k=3: count(4)=31=(5^3-1)/4, all birth layers (p-1)/4*p^2=25",
        "fn": lambda: (
            _orbit_counts_dm1(11, 5, 3).get(4, -1) == 31 and
            _orbit_counts_dm1(11, 5, 3).get(20, -1) == 25 and
            _orbit_counts_dm1(11, 5, 3).get(100, -1) == 25 and
            _orbit_counts_dm1(11, 5, 3).get(500, -1) == 25
        ),
    },
    # FIX4: p=5, r=3 (t=11), k=4 (k>r): frozen L=1, r=3 joint birth layers
    {
        "id": "FIX4_P5_R3_K4",
        "desc": "p=5, r=3, t=11, k=4: count(4)=31, frozen count(20)=(p^2-1)/4*p^2=150, birth=125",
        "fn": lambda: (
            _orbit_counts_dm1(11, 5, 4).get(4, -1) == 31 and
            _orbit_counts_dm1(11, 5, 4).get(20, -1) == 150 and  # frozen L=1: (p^2-1)/4*p^(1+3-2)=6*25=150
            _orbit_counts_dm1(11, 5, 4).get(100, -1) == 125 and  # birth: (p-1)/4*p^3=1*125=125
            _orbit_counts_dm1(11, 5, 4).get(500, -1) == 125 and
            _orbit_counts_dm1(11, 5, 4).get(2500, -1) == 125
        ),
    },
    # FIX5: p=13, r=1 (t=3), k=2: full distribution
    {
        "id": "FIX5_P13_R1_K2",
        "desc": "p=13, r=1, t=3, k=2: {1:1, 4:3, 52:42, 676:39}",
        "fn": lambda: _orbit_counts_dm1(3, 13, 2) == {1: 1, 4: 3, 52: 42, 676: 39},
    },
    # FIX6: ker mechanism: p=5, r=2, k=2 → ker=25=p^2
    {
        "id": "FIX6_KER_P5_R2_K2",
        "desc": "p=5, r=2, t=14, k=2: ker(tI-2M mod 25)=25=p^min(2,2)=p^2",
        "fn": lambda: _ker_size_dm1(14, 5, 2) == 25,
    },
    # FIX7: count(1)=1 structural check for p=13, k=2
    {
        "id": "FIX7_FIXED_1_P13",
        "desc": "p=13, r=1, t=3, k=2: count(1)=1 (only zero vector is M-fixed)",
        "fn": lambda: _orbit_counts_dm1(3, 13, 2).get(1, -1) == 1,
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []

    check_fns = [
        _check_c1_period_set,
        _check_c2_count4_saturation,
        _check_c3_unified_formula,
        _check_c4_fixed_always_1,
        _check_c5_algebraic_ker,
        _check_c6_p3mod4_no_ramification,
    ]
    ok_checks = 0
    total_checks = len(check_fns)
    for fn in check_fns:
        if fn(results):
            ok_checks += 1

    fixture_results = []
    for fix in FIXTURES:
        try:
            passed = fix["fn"]()
        except Exception:
            passed = False
        fixture_results.append({"id": fix["id"], "desc": fix["desc"], "passed": passed})

    fixture_pass = sum(1 for f in fixture_results if f["passed"])
    fixture_total = len(fixture_results)

    ok = (ok_checks == total_checks) and (fixture_pass == fixture_total)
    output = {
        "cert": "[440] QA Witt Tower det=-1 General v_p Period Law",
        "ok": ok,
        "checks_passed": f"{ok_checks}/{total_checks}",
        "fixture_summary": f"{fixture_pass}/{fixture_total} passed",
        "results": results,
        "fixtures": fixture_results,
    }
    print(json.dumps(output, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
