#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=doubly-ramified lifting in p-adic number theory; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 (higher ramification, p-adic lifting); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (Hensel lifting, nilpotent elements mod p^k); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano-period structure for quadratic recurrences) -->
"""QA Witt Tower Doubly-Ramified Period Law Cert [438].

When v_p(t-2) = 2 (i.e. p^2 | (t-2) but p^3 does not) in the det=+1
companion family (M=[[t,-1],[1,0]], char poly x^2-tx+1), the period-set
and orbit-count structure differs from both the v_p=1 case ([437]) and
what naive extrapolation from [435] would predict.

KEY DISTINCTION from v_p=1 ([437]):
  At k=1 (mod p):     identical to v_p=1 — count(1) = p
  At k>=2 (mod p^k):  count(1) = p^2   <-- doubled fixed-point count

ORBIT-COUNT LAW for v_p(t-2) = 2, k >= 1:

  k=1:   period set {1, p},    counts {1: p, p: p-1}    (same as v_p=1)
  k>=2:  period set {p^L : L = 0, 1, ..., k}
         count(1)      = p^2
         count(p^L)    = (p^2-1) * p^L    for 1 <= L <= k-2  (frozen)
         count(p^(k-1)) = (p-1) * p^(k-1)                   (joint birth)
         count(p^k)    = (p-1) * p^(k-1)                    (birth)

Unlike [437]'s v_p=1 case, this formula holds universally for ALL primes p
and ALL values of c2 = (t-2)/p^2. There is no p=3 stall analogue.

MECHANISM (Double Nilpotency):
  N = M - I satisfies N^2 = (t-2)*M = p^2 * c2 * M exactly (integer).
  When p^2 | (t-2): N^2 ≡ 0 mod p^2 (doubly nilpotent, stronger than [437]).

  ker(N mod p^2) = {(a, a) : a in Z/p^2Z} = p^2 elements
  vs ker(N mod p^2) = p elements for v_p=1 ([437]).

  This explains the fixed-point jump: count(1,k) = |ker(N mod p^2)| = p^2
  for ALL k >= 2, rather than p as in [437].

JOINT BIRTH (top two layers share same count):
  The birth count (p-1)*p^(k-1) is split equally across L=k and L=k-1.
  This reflects the extra lifting freedom from the doubled ramification:
  the tower cannot distinguish L=k-1 from L=k in its birth structure.

Checks
------
C1  V2_PERIOD_SET    -- period set = {p^L : L=0..k}, v_p=2, p=5, many (t,k)
C2  V2_FIXED_JUMP    -- count(1)=p at k=1; count(1)=p^2 for k>=2
C3  V2_ORBIT_FORMULA -- full formula, p=3,5,7, all c2 in {1..p-1}, k=1..4
C4  DOUBLE_NILPOTENT -- N^2=p^2*c2*M exact; N^2≡0 mod p^2; ker size=p^2
C5  NO_P3_EXCEPTION  -- formula holds for p=3 at all c2 values (no stall)
C6  K1_INVISIBLE     -- at k=1, v_p=2 indistinguishable from v_p=1
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


def _expected_v2_orbits(p, k):
    """Expected orbit counts for v_p(t-2) = 2."""
    if k == 1:
        return {1: p, p: p - 1}
    exp = {1: p * p}
    for L in range(1, k - 1):          # frozen: L in 1..k-2
        exp[p ** L] = (p * p - 1) * p ** L
    exp[p ** (k - 1)] = (p - 1) * p ** (k - 1)   # joint birth: L=k-1
    exp[p ** k] = (p - 1) * p ** (k - 1)           # birth: L=k
    return exp


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_c1_period_set():
    """C1: period set = {p^L : L=0..k} for p=5, v_p=2, k=1..4."""
    p = 5
    violations = []
    pairs = 0
    for c2 in range(1, p):
        t = p * p * c2 + 2
        assert (t - 2) % (p * p) == 0
        assert (t - 2) % (p * p * p) != 0  # v_p exactly 2
        M = [[t, -1], [1, 0]]
        for k in range(1, 5):
            obs = _orbit_counts(M, p, k)
            expected_set = {p ** L for L in range(k + 1)}
            if set(obs.keys()) != expected_set:
                violations.append({"t": t, "k": k, "got": sorted(obs.keys()),
                                   "expected": sorted(expected_set)})
            pairs += 1
    return {"p": p, "c2_range": f"1..{p-1}", "k_range": "1..4",
            "pairs": pairs, "violations": violations, "ok": not violations}


def _check_c2_fixed_jump():
    """C2: count(1)=p at k=1; count(1)=p^2 for k>=2, all primes p in {3,5,7}."""
    failures = []
    for p in [3, 5, 7]:
        for c2 in range(1, p):
            t = p * p * c2 + 2
            M = [[t, -1], [1, 0]]
            # k=1: should equal p (same as v_p=1)
            obs1 = _orbit_counts(M, p, 1)
            if obs1.get(1, 0) != p:
                failures.append({"p": p, "c2": c2, "k": 1, "got": obs1.get(1, 0), "expected": p})
            # k=2,3,4: should equal p^2
            for k in range(2, 5):
                if p ** k > 5 * 10 ** 5:
                    break
                obs = _orbit_counts(M, p, k)
                if obs.get(1, 0) != p * p:
                    failures.append({"p": p, "c2": c2, "k": k,
                                     "got": obs.get(1, 0), "expected": p * p})
    return {"failures": failures, "ok": not failures}


def _check_c3_orbit_formula():
    """C3: full orbit formula, p=3,5,7, all c2 in {1..p-1}, k=1..4."""
    violations = []
    pairs = 0
    for p in [3, 5, 7]:
        for c2 in range(1, p):
            t = p * p * c2 + 2
            M = [[t, -1], [1, 0]]
            for k in range(1, 5):
                if p ** k > 5 * 10 ** 5:
                    break
                obs = _orbit_counts(M, p, k)
                exp = _expected_v2_orbits(p, k)
                if obs != exp:
                    violations.append({"p": p, "c2": c2, "t": t, "k": k,
                                       "got": obs, "expected": exp})
                pairs += 1
    return {"pairs": pairs, "violations": violations, "ok": not violations}


def _check_c4_double_nilpotent():
    """C4: N^2=p^2*c2*M exactly (integer); N^2≡0 mod p^2; ker(N mod p^2)=p^2."""
    identity_failures = []
    nilpotent_failures = []
    kernel_failures = []
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
            if (t - 2) % (p * p) == 0 and (t - 2) > 0:
                # N^2 ≡ 0 mod p^2
                if any(N2[i][j] % (p * p) != 0 for i in range(2) for j in range(2)):
                    nilpotent_failures.append((t, p, "N2 not ≡0 mod p^2"))
                # ker(N mod p^2) = {(a,a): a in Z/p^2Z} = p^2 elements
                p2 = p * p
                ker_size = sum(
                    1 for a in range(p2) for b in range(p2)
                    if ((N[0][0] * a + N[0][1] * b) % p2 == 0 and
                        (N[1][0] * a + N[1][1] * b) % p2 == 0)
                )
                if ker_size != p2:
                    kernel_failures.append((t, p, ker_size))
    return {
        "t_range": "3..299",
        "identity_failures": identity_failures,
        "nilpotent_failures": nilpotent_failures,
        "kernel_failures": kernel_failures,
        "ok": not identity_failures and not nilpotent_failures and not kernel_failures,
    }


def _check_c5_no_p3_exception():
    """C5: formula holds for p=3 at ALL c2 values — no stall unlike [437]."""
    failures = []
    p = 3
    for c2 in range(1, p):     # c2 in {1, 2}
        t = p * p * c2 + 2
        M = [[t, -1], [1, 0]]
        for k in range(1, 6):
            if p ** k > 10 ** 6:
                break
            obs = _orbit_counts(M, p, k)
            exp = _expected_v2_orbits(p, k)
            if obs != exp:
                failures.append({"c2": c2, "t": t, "k": k})
    return {"p": 3, "c2_tested": list(range(1, p)), "failures": failures,
            "ok": not failures}


def _check_c6_k1_invisible():
    """C6: at k=1, v_p=2 is identical to v_p=1 — {1:p, p:p-1}."""
    failures = []
    for p in [3, 5, 7, 11]:
        # v_p=1 representative
        t_v1 = p + 2         # t-2 = p → v_p = 1
        # v_p=2 representatives
        for c2 in range(1, min(p, 4)):
            t_v2 = p * p * c2 + 2    # t-2 = p^2*c2 → v_p = 2
            obs_v1 = _orbit_counts([[t_v1, -1], [1, 0]], p, 1)
            obs_v2 = _orbit_counts([[t_v2, -1], [1, 0]], p, 1)
            if obs_v1 != obs_v2:
                failures.append({"p": p, "c2": c2, "v1": obs_v1, "v2": obs_v2})
            if obs_v2 != {1: p, p: p - 1}:
                failures.append({"p": p, "c2": c2, "expected_form": {1: p, p: p - 1},
                                  "got": obs_v2})
    return {"failures": failures, "ok": not failures}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _run_fixtures():
    def _n2_doubly_nil(t, p):
        N = [[t - 1, -1], [1, -1]]
        N2 = [[N[0][0]*N[0][0]+N[0][1]*N[1][0], N[0][0]*N[0][1]+N[0][1]*N[1][1]],
              [N[1][0]*N[0][0]+N[1][1]*N[1][0], N[1][0]*N[0][1]+N[1][1]*N[1][1]]]
        return all(N2[i][j] % (p * p) == 0 for i in range(2) for j in range(2))

    raw = [
        # FIX-1: k=1 same for v_p=1 and v_p=2 (p=5)
        ("FIX1_K1_INVISIBLE_P5",
         "p=5: k=1 distribution identical for t=7 (v_p=1) and t=27 (v_p=2)",
         True,
         _orbit_counts([[7, -1], [1, 0]], 5, 1) == _orbit_counts([[27, -1], [1, 0]], 5, 1)),

        # FIX-2: fixed-point jump at k=2 (p=5)
        ("FIX2_FP_JUMP_P5_K2",
         "p=5 t=27 k=2: count(1) = p^2 = 25",
         True,
         _orbit_counts([[27, -1], [1, 0]], 5, 2).get(1, 0) == 25),

        # FIX-3: v_p=1 does NOT have p^2 fixed points at k=2
        ("FIX3_V1_STILL_P_NOT_P2",
         "p=5 t=7 (v_p=1) k=2: count(1) = p = 5, not p^2",
         True,
         _orbit_counts([[7, -1], [1, 0]], 5, 2).get(1, 0) == 5),

        # FIX-4: joint birth — L=k-1 and L=k have same count
        ("FIX4_JOINT_BIRTH_P5_K3",
         "p=5 t=27 k=3: count(period=25) == count(period=125)",
         True,
         (lambda d: d.get(25, -1) == d.get(125, -2))(
             _orbit_counts([[27, -1], [1, 0]], 5, 3))),

        # FIX-5: p=3 c2=2 formula holds (no stall unlike [437] v_p=1)
        ("FIX5_P3_C2_2_NO_STALL",
         "p=3 t=20 (c2=2 v_p=2) k=2: formula holds",
         True,
         _orbit_counts([[20, -1], [1, 0]], 3, 2) == _expected_v2_orbits(3, 2)),

        # FIX-6: double nilpotency for t=27, p=5
        ("FIX6_DOUBLE_NIL_T27_P5",
         "t=27 p=5: N^2 ≡ 0 mod 25",
         True,
         _n2_doubly_nil(27, 5)),

        # FIX-7: WRONG_CLAIM: v_p=2 does NOT give p^2 fixed points at k=1
        ("FIX7_WRONG_CLAIM_P2_AT_K1",
         "v_p=2 does NOT give p^2=25 fixed-point orbits at k=1 (only p=5)",
         True,
         _orbit_counts([[27, -1], [1, 0]], 5, 1).get(1, 0) != 25),
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
        "C2_FIXED_JUMP": _check_c2_fixed_jump(),
        "C3_ORBIT_FORMULA": _check_c3_orbit_formula(),
        "C4_DOUBLE_NILPOTENT": _check_c4_double_nilpotent(),
        "C5_NO_P3_EXCEPTION": _check_c5_no_p3_exception(),
        "C6_K1_INVISIBLE": _check_c6_k1_invisible(),
    }
    all_ok = all(v["ok"] for v in checks.values())

    fixtures = _run_fixtures()
    n_pass = sum(1 for f in fixtures if f["passed"])

    out = {
        "ok": all_ok and n_pass == len(fixtures),
        "cert": "QA Witt Tower Doubly-Ramified Period Law",
        "family_id": 438,
        "checks": {k: v["ok"] for k, v in checks.items()},
        "checks_detail": checks,
        "fixture_summary": f"{n_pass}/{len(fixtures)} passed",
        "fixtures": fixtures,
    }
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
