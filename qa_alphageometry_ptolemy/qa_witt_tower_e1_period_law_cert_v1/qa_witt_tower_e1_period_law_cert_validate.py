#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical nilpotent lifting theory over Witt vectors; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 (ramified p-adic extensions, nilpotent lifting); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (Hensel lifting, primitive roots); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano-period structure for quadratic recurrences) -->
"""QA Witt Tower Ramified Prime e=1 Period Law Cert [437].

When e=ord(lambda0 mod p)=1 (i.e. p divides t-2 in the det=+1 companion
family), the period-set/multiplicity formulas from [435] generalize with
a different structure than the e=2 and e=4 cases:

  Period set:  {p^L : L = 0, 1, ..., k}
  orbit count(1, k)         = p                    (NOT 1 = e)
  orbit count(p^L, L = k)   = (p-1) * p^(L-1)     (birth layer)
  orbit count(p^L, L < k)   = (p^2-1) * p^(L-1)   (frozen layer)

This holds universally for p >= 5 (any c = (t-2)/p mod p).
For p = 3: holds when c = (t-2)/3 ≡ 1 mod 3; a stall occurs at k=2
when c ≡ 2 mod 3 (no period-p^2 orbits form -- analogous to [435] p=2 stall).

Nilpotent mechanism: N = M-I satisfies N^2 = (t-2)*M as exact integer
identity. When p|(t-2), N^2 ≡ 0 mod p, so N is nilpotent degree 2 mod p.
This forces ker(N mod p) to be 1-dimensional (p elements), giving
count(1, k) = p for ALL k.

Checks
------
C1  E1_PERIOD_SET     -- period set = {p^L : L=0..k} for p=5, many (t,k)
C2  E1_FIXED_COUNT    -- orbit count(1) = p at every k, every valid p
C3  E1_ORBIT_FORMULA  -- full orbit-count formula, p=5 and p=7
C4  NILPOTENT         -- N^2 = (t-2)*M exact integer, N^2 ≡ 0 mod p
C5  P3_STALL          -- p=3 c≡2: stall at k=2; c≡1: formula holds
C6  LEGACY            -- [436] e=1 instances reproduce correctly
"""

import json
import sys


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _orbit_counts(M, p, k):
    """Return {period: orbit_count} for M acting on (Z/p^k Z)^2."""
    m = p ** k
    counts = {}
    visited = bytearray(m * m)
    for a in range(m):
        for b in range(m):
            if visited[a * m + b]:
                continue
            ca, cb = a, b
            n = 0
            while True:
                visited[ca * m + cb] = 1
                na = (M[0][0] * ca + M[0][1] * cb) % m
                nb = (M[1][0] * ca + M[1][1] * cb) % m
                ca, cb = na, nb
                n += 1
                if ca == a and cb == b:
                    break
            counts[n] = counts.get(n, 0) + 1
    return counts


def _expected_e1_orbits(p, k):
    """Expected orbit counts under the e=1 period law."""
    exp = {1: p}
    for L in range(1, k + 1):
        exp[p ** L] = (p - 1) * p ** (L - 1) if L == k else (p ** 2 - 1) * p ** (L - 1)
    return exp


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_c1_period_set():
    """C1: period set = {p^L : L=0..k} for p=5, k=1..3, many t values."""
    p = 5
    violations = []
    pairs = 0
    for t in range(p + 2, 200, p):
        assert (t - 2) % p == 0
        M = [[t, -1], [1, 0]]
        for k in range(1, 4):
            obs = _orbit_counts(M, p, k)
            expected_set = {p ** L for L in range(k + 1)}
            if set(obs.keys()) != expected_set:
                violations.append({"t": t, "k": k, "got": sorted(obs.keys()),
                                   "expected": sorted(expected_set)})
            pairs += 1
    return {"p": p, "t_range": "p+2..200 step p", "k_range": "1..3",
            "pairs": pairs, "violations": violations, "ok": not violations}


def _check_c2_fixed_count():
    """C2: orbit count(period=1) = p for every valid (p, t, k).

    Restricted to v_p(t-2) = 1 (simple ramification); doubly-ramified
    cases (p^2 | (t-2)) have a different fixed-point structure.

    Note: the p=3 stall only disrupts the p^2-level layer, not the
    fixed-point count. Fixed points satisfy N*x ≡ 0 mod p (not mod p^k),
    so their count = dim(ker N mod p) = p regardless of stall.
    """
    failures = []
    for p in [3, 5, 7, 11]:
        for mul in range(1, 6):
            t = p * mul + 2
            if (t - 2) % (p * p) == 0:
                continue  # doubly ramified: v_p(t-2) >= 2, out of scope
            M = [[t, -1], [1, 0]]
            for k in range(1, 5):
                obs = _orbit_counts(M, p, k)
                fp = obs.get(1, 0)
                if fp != p:
                    failures.append({"p": p, "t": t, "k": k, "got": fp, "expected": p})
    return {"failures": failures, "ok": not failures}


def _check_c3_orbit_formula():
    """C3: full orbit-count formula holds for p=5 (k=1..3) and p=7 (k=1..2).

    Restricted to v_p(t-2) = 1 (simple ramification).
    """
    violations = []
    pairs = 0
    for p, k_max in [(5, 3), (7, 2)]:
        t_limit = 150 if p == 5 else 80
        for t in range(p + 2, t_limit, p):
            if (t - 2) % (p * p) == 0:
                continue  # doubly ramified: out of scope
            M = [[t, -1], [1, 0]]
            for k in range(1, k_max + 1):
                obs = _orbit_counts(M, p, k)
                exp = _expected_e1_orbits(p, k)
                if obs != exp:
                    violations.append({"p": p, "t": t, "k": k,
                                       "got": obs, "expected": exp})
                pairs += 1
    return {"pairs": pairs, "violations": violations, "ok": not violations}


def _check_c4_nilpotent():
    """C4: N=M-I satisfies N^2=(t-2)*M exactly; N^2 ≡ 0 mod p when p|(t-2)."""
    identity_failures = []
    nilpotent_failures = []
    for t in range(3, 300):
        M = [[t, -1], [1, 0]]
        N = [[t - 1, -1], [1, -1]]
        N2 = [[N[0][0] * N[0][0] + N[0][1] * N[1][0],
               N[0][0] * N[0][1] + N[0][1] * N[1][1]],
              [N[1][0] * N[0][0] + N[1][1] * N[1][0],
               N[1][0] * N[0][1] + N[1][1] * N[1][1]]]
        tM = [[(t - 2) * M[i][j] for j in range(2)] for i in range(2)]
        if N2 != tM:
            identity_failures.append(t)
        for p in [3, 5, 7]:
            if (t - 2) % p == 0:
                if any(N2[i][j] % p != 0 for i in range(2) for j in range(2)):
                    nilpotent_failures.append((t, p))
    return {
        "t_range": "3..299",
        "identity_failures": identity_failures,
        "nilpotent_failures": nilpotent_failures,
        "ok": not identity_failures and not nilpotent_failures,
    }


def _check_c5_p3_stall():
    """C5: p=3 stall for c≡2 mod 3 at k=2; formula holds for c≡1 mod 3."""
    stall_failures = []
    for t in [8, 17, 26, 35]:          # c = 2, 5, 8, 11 — all ≡ 2 mod 3
        c = (t - 2) // 3
        assert c % 3 == 2
        M = [[t, -1], [1, 0]]
        if _orbit_counts(M, 3, 1) != _expected_e1_orbits(3, 1):
            stall_failures.append({"t": t, "k": 1, "issue": "k=1 should pass"})
        obs2 = _orbit_counts(M, 3, 2)
        if 9 in obs2:
            stall_failures.append({"t": t, "k": 2, "issue": "stall: period-9 absent"})
        if obs2 == _expected_e1_orbits(3, 2):
            stall_failures.append({"t": t, "k": 2, "issue": "formula should fail"})

    nonstall_failures = []
    for t in [5, 14, 23, 32]:          # c = 1, 4, 7, 10 — all ≡ 1 mod 3
        c = (t - 2) // 3
        assert c % 3 == 1
        M = [[t, -1], [1, 0]]
        for k in range(1, 4):
            obs = _orbit_counts(M, 3, k)
            exp = _expected_e1_orbits(3, k)
            if obs != exp:
                nonstall_failures.append({"t": t, "k": k})

    return {
        "stall_t": [8, 17, 26, 35],
        "nonstall_t": [5, 14, 23, 32],
        "stall_failures": stall_failures,
        "nonstall_failures": nonstall_failures,
        "ok": not stall_failures and not nonstall_failures,
    }


def _check_c6_legacy():
    """C6: [436]-established e=1 instances reproduce the period law."""
    # [436] C1: e=1 when p|(t-2) in det=+1 family. Three clean cases:
    #   t=7,  p=5: (t-2)=5=p, c=1 (c%5=1), p>=5 → formula holds
    #   t=5,  p=3: (t-2)=3=p, c=1 (c%3=1) → no stall
    #   t=12, p=5: (t-2)=10=2p, c=2 (c%5=2), p>=5 → formula holds
    cases = [(7, 5), (5, 3), (12, 5)]
    failures = []
    for t, p in cases:
        M = [[t, -1], [1, 0]]
        k_max = 3 if p <= 5 else 2
        for k in range(1, k_max + 1):
            obs = _orbit_counts(M, p, k)
            exp = _expected_e1_orbits(p, k)
            if obs != exp:
                failures.append({"t": t, "p": p, "k": k, "got": obs, "expected": exp})
    return {"cases": [[t, p] for t, p in cases], "failures": failures, "ok": not failures}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _run_fixtures():
    def _n2_eq_tm(t):
        M = [[t, -1], [1, 0]]
        N = [[t - 1, -1], [1, -1]]
        N2 = [[N[0][0]*N[0][0]+N[0][1]*N[1][0], N[0][0]*N[0][1]+N[0][1]*N[1][1]],
              [N[1][0]*N[0][0]+N[1][1]*N[1][0], N[1][0]*N[0][1]+N[1][1]*N[1][1]]]
        tM = [[(t-2)*M[i][j] for j in range(2)] for i in range(2)]
        return N2 == tM

    raw = [
        ("FIX1_FP_P5_K1",
         "p=5 t=7 k=1: exactly 5 fixed-point orbits",
         True,
         _orbit_counts([[7, -1], [1, 0]], 5, 1).get(1, 0) == 5),
        ("FIX2_FROZEN_P5_K2",
         "p=5 t=7 k=2: orbit count(period=5) = p^2-1 = 24",
         True,
         _orbit_counts([[7, -1], [1, 0]], 5, 2).get(5, -1) == 24),
        ("FIX3_BIRTH_P5_K2",
         "p=5 t=7 k=2: orbit count(period=25) = (p-1)*p^1 = 20",
         True,
         _orbit_counts([[7, -1], [1, 0]], 5, 2).get(25, -1) == 20),
        ("FIX4_P3_NONSTALL_K2",
         "p=3 t=5 c=1 k=2: orbit formula holds (no stall)",
         True,
         _orbit_counts([[5, -1], [1, 0]], 3, 2) == _expected_e1_orbits(3, 2)),
        ("FIX5_P3_STALL_K2",
         "p=3 t=8 c=2 k=2: stall -- no period-9 orbits",
         True,
         9 not in _orbit_counts([[8, -1], [1, 0]], 3, 2)),
        ("FIX6_NIL_T100",
         "t=100: N^2 = 98*M exactly as integer identity",
         True,
         _n2_eq_tm(100)),
        ("FIX7_WRONG_CLAIM_FP_ONE",
         "e=1 does NOT give 1 fixed-point orbit -- gives p=5",
         True,
         _orbit_counts([[7, -1], [1, 0]], 5, 3).get(1, 0) != 1),
    ]
    return [
        {"name": n, "desc": d, "expected": e, "actual": a, "passed": a == e}
        for n, d, e, a in raw
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checks = {
        "C1_PERIOD_SET": _check_c1_period_set(),
        "C2_FIXED_COUNT": _check_c2_fixed_count(),
        "C3_ORBIT_FORMULA": _check_c3_orbit_formula(),
        "C4_NILPOTENT": _check_c4_nilpotent(),
        "C5_P3_STALL": _check_c5_p3_stall(),
        "C6_LEGACY": _check_c6_legacy(),
    }
    all_ok = all(v["ok"] for v in checks.values())

    fixtures = _run_fixtures()
    n_pass = sum(1 for f in fixtures if f["passed"])

    out = {
        "ok": all_ok and n_pass == len(fixtures),
        "cert": "QA Witt Tower Ramified Prime e=1 Period Law",
        "family_id": 437,
        "checks": {k: v["ok"] for k, v in checks.items()},
        "checks_detail": checks,
        "fixture_summary": f"{n_pass}/{len(fixtures)} passed",
        "fixtures": fixtures,
    }
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
