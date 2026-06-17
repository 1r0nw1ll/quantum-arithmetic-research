#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical quadratic-recurrence ramification and primitive-root order theory; Serre (1979) doi.org/10.1007/978-1-4757-5673-9 (ramified extensions, double roots mod p); Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 (primitive roots, Hensel lifting); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano-style period tables) -->
"""QA Witt Tower Ramified Prime e-VALUE RIGIDITY -- cert [436].

[435] generalized [434]'s p=5/Fibonacci-only closed form to "any odd
ramified prime p ... with e := ord(lambda0 mod p) as a free parameter,"
verified on exactly two data points: e=4 (Fibonacci, det=-1) and e=2
(D12, det=+1). This cert shows that framing OVERCLAIMS: for the
two-term companion-matrix families used throughout this chain
(det=+1 family: M=[[t,-1],[1,0]], char poly x^2-tx+1, D=t^2-4;
det=-1 family: M=[[t, 1],[1,0]], char poly x^2-tx-1, D=t^2+4),
e is NOT free at all -- it is mechanically forced:

    det(M)=+1:  e = 1  if p | (t-2)
                e = 2  if p | (t+2)          -- only {1, 2} ever occur

    det(M)=-1:  e = 4  for every odd ramified prime -- only {4} ever occurs

The mechanism is algebraic, not empirical: for det=+1, ramification
requires t === +/-2 mod p, so lambda0 = t/2 === +/-1 mod p, and
ord(+1)=1, ord(-1)=2 for any odd p.  For det=-1, ramification requires
t^2 === -4 mod p, so lambda0^2 === (t/2)^2 === -1 mod p; for any odd p,
-1 != 1, so ord(lambda0) cannot be 1 or 2 -- it must be exactly 4.

[434]'s p=5 (e=4) and [435]'s p=3 (e=2, 3|(t+2) at t=4) are not
independent free-parameter samples: they are the two only possible
outcomes of this law for their respective families.

CLAIM 1 (DET_PLUS1_E_RIGIDITY): for the det=+1 family (x^2-tx+1),
every odd prime p dividing D=t^2-4=(t-2)(t+2) divides EXACTLY one of
(t-2),(t+2) (never both nor neither, since gcd(t-2,t+2)|4 for any t),
and e:=ord(lambda0 mod p) equals 1 in the (t-2) case and 2 in the (t+2)
case -- no other value ever occurs.  Verified: t=3..T_MAX, all odd p|D.

CLAIM 2 (DET_MINUS1_E_RIGIDITY): for the det=-1 family (x^2-tx-1),
every odd prime p dividing D=t^2+4 gives e:=ord(lambda0 mod p)=4
exactly -- no other value ever occurs.  Verified: t=1..T_MAX, all
odd p|D.

CLAIM 3 (NO_THIRD_VALUE_GLOBAL): aggregating all (t,p) pairs from
Claims 1-2, observed e values are {1,2} for det=+1 and {4} for
det=-1 -- e=3,5,6,... never occur in either family across the full
sweep.  This directly falsifies [435]'s "free parameter" framing:
e is not free; it is determined by (det, Vieta-factor-membership)
alone.

CLAIM 4 (BRANCH_MECHANISM_EXACT): the residue-level mechanism, checked
as exact mod-p identities (not derived from the resulting order):
  det=+1, p|(t-2) branch:  lambda0 === +1 mod p exactly
  det=+1, p|(t+2) branch:  lambda0 === -1 mod p exactly
  det=-1, every odd p:     lambda0^2 === -1 mod p exactly
Verified over the same sweep as Claims 1-2.

CLAIM 5 (LEGACY_CONSISTENCY): [434]'s Fibonacci case (t=1, det=-1,
p=5, e=4) and [435]'s D12 case (t=4, det=+1, p=3, e=2, 3|(t+2)=6)
are both reproduced as direct instances of this rigidity law via the
SAME generic code path used for Claims 1-2, confirming this cert
refines rather than contradicts either prior result.

WHY THIS IS NOT A DUPLICATE: [434] and [435] treat e as an INPUT --
computed once per (M,p) and plugged into period-set/multiplicity
formulas; neither cert asks what values e can take or whether e is
actually free.  This cert characterizes e itself and shows it is
rigid, not free -- correcting [435]'s stated framing without altering
any prior PASS verdict.

THEOREM NT: all arithmetic is pure integer mod p throughout; lambda0
and lambda0^2 are computed by integer modular arithmetic; no float, no
observer projection enters QA state.

PRIMARY SOURCES:
  Serre (1979) doi.org/10.1007/978-1-4757-5673-9 -- ramification theory
  Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.7 -- primitive roots, Hensel
  Wall (1960) doi.org/10.1080/00029890.1960.11989541 -- Pisano-style period tables
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Sweep bounds (t ranges)
# ---------------------------------------------------------------------------

T_PLUS1_MAX = 3000   # det=+1 family: t = 3 .. T_PLUS1_MAX
T_MINUS1_MAX = 3000  # det=-1 family: t = 1 .. T_MINUS1_MAX


# ---------------------------------------------------------------------------
# Arithmetic helpers
# ---------------------------------------------------------------------------

def _prime_factors(n: int) -> set[int]:
    n = abs(n)
    fs: set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            fs.add(d)
            n //= d
        d += 1
    if n > 1:
        fs.add(n)
    return fs


def _order_mod_p(a: int, p: int) -> int:
    """Multiplicative order of a mod p (p prime, a != 0 mod p)."""
    e = 1
    cur = a % p
    if cur == 0:
        raise ValueError(f"{a} is zero mod {p}")
    while cur != 1:
        cur = (cur * a) % p
        e += 1
        if e > p:
            raise RuntimeError(f"order of {a} mod {p} exceeds p -- not a unit?")
    return e


def _lambda0(t: int, p: int) -> int:
    """Double root of char poly mod p: lambda0 = t * inverse(2) mod p."""
    return (t * pow(2, -1, p)) % p


# ---------------------------------------------------------------------------
# C1: det=+1 e rigidity
# ---------------------------------------------------------------------------

def _check_det_plus1_e_rigidity(t_max: int) -> dict[str, Any]:
    failures: list[str] = []
    pairs_tested = 0
    for t in range(3, t_max + 1):
        D = t * t - 4
        if D <= 0:
            continue
        for p in sorted(_prime_factors(D)):
            if p == 2:
                continue
            div_minus = (t - 2) % p == 0
            div_plus = (t + 2) % p == 0
            if div_minus == div_plus:
                failures.append(
                    f"t={t},p={p}: ambiguous Vieta membership "
                    f"(minus={div_minus}, plus={div_plus})"
                )
                continue
            lam0 = _lambda0(t, p)
            e = _order_mod_p(lam0, p)
            expected = 1 if div_minus else 2
            pairs_tested += 1
            if e != expected:
                failures.append(
                    f"t={t},p={p}: lambda0={lam0} e={e} "
                    f"expected={expected} "
                    f"(branch={'(t-2)' if div_minus else '(t+2)'})"
                )
    return {"ok": not failures, "failures": failures, "pairs_tested": pairs_tested}


# ---------------------------------------------------------------------------
# C2: det=-1 e rigidity
# ---------------------------------------------------------------------------

def _check_det_minus1_e_rigidity(t_max: int) -> dict[str, Any]:
    failures: list[str] = []
    pairs_tested = 0
    for t in range(1, t_max + 1):
        D = t * t + 4
        for p in sorted(_prime_factors(D)):
            if p == 2:
                continue
            lam0 = _lambda0(t, p)
            e = _order_mod_p(lam0, p)
            pairs_tested += 1
            if e != 4:
                failures.append(
                    f"t={t},p={p}: lambda0={lam0} e={e} expected=4"
                )
    return {"ok": not failures, "failures": failures, "pairs_tested": pairs_tested}


# ---------------------------------------------------------------------------
# C3: no third value globally
# ---------------------------------------------------------------------------

def _check_no_third_value_global(t1_max: int, t2_max: int) -> dict[str, Any]:
    plus1_e_values: set[int] = set()
    minus1_e_values: set[int] = set()

    for t in range(3, t1_max + 1):
        D = t * t - 4
        if D <= 0:
            continue
        for p in _prime_factors(D):
            if p == 2:
                continue
            plus1_e_values.add(_order_mod_p(_lambda0(t, p), p))

    for t in range(1, t2_max + 1):
        D = t * t + 4
        for p in _prime_factors(D):
            if p == 2:
                continue
            minus1_e_values.add(_order_mod_p(_lambda0(t, p), p))

    failures: list[str] = []
    unexpected_plus1 = sorted(plus1_e_values - {1, 2})
    unexpected_minus1 = sorted(minus1_e_values - {4})
    if unexpected_plus1:
        failures.append(
            f"det=+1 family: unexpected e values {unexpected_plus1} "
            f"(allowed only {{1, 2}})"
        )
    if unexpected_minus1:
        failures.append(
            f"det=-1 family: unexpected e values {unexpected_minus1} "
            f"(allowed only {{4}})"
        )

    return {
        "ok": not failures,
        "failures": failures,
        "observed_plus1_e_values": sorted(plus1_e_values),
        "observed_minus1_e_values": sorted(minus1_e_values),
    }


# ---------------------------------------------------------------------------
# C4: exact residue mechanism
# ---------------------------------------------------------------------------

def _check_branch_mechanism_exact(t1_max: int, t2_max: int) -> dict[str, Any]:
    failures: list[str] = []
    checked = 0

    for t in range(3, t1_max + 1):
        D = t * t - 4
        if D <= 0:
            continue
        for p in _prime_factors(D):
            if p == 2:
                continue
            lam0 = _lambda0(t, p)
            checked += 1
            if (t - 2) % p == 0:
                # p | (t-2)  =>  t === 2 mod p  =>  lambda0 = t/2 === 1 mod p
                if lam0 != 1:
                    failures.append(
                        f"t={t},p={p}: p|(t-2) but lambda0={lam0} != 1"
                    )
            else:
                # p | (t+2)  =>  t === -2 mod p  =>  lambda0 = t/2 === -1 mod p
                if lam0 != p - 1:
                    failures.append(
                        f"t={t},p={p}: p|(t+2) but lambda0={lam0} != {p-1} (=-1 mod p)"
                    )

    for t in range(1, t2_max + 1):
        D = t * t + 4
        for p in _prime_factors(D):
            if p == 2:
                continue
            lam0 = _lambda0(t, p)
            checked += 1
            lam0_sq = (lam0 * lam0) % p
            if lam0_sq != p - 1:
                # lambda0^2 must equal -1 mod p (= p-1)
                failures.append(
                    f"t={t},p={p}: lambda0^2={lam0_sq} != {p-1} (=-1 mod p)"
                )

    return {"ok": not failures, "failures": failures, "checked": checked}


# ---------------------------------------------------------------------------
# C5: legacy consistency with [434] and [435]
# ---------------------------------------------------------------------------

# [434] Fibonacci: M=[[1,1],[1,0]], trace t=1, det=-1, ramified prime p=5, e=4
FIB_T, FIB_DET, FIB_P, FIB_E = 1, -1, 5, 4
# [435] D12: M=[[4,-1],[1,0]], trace t=4, det=+1, ramified prime p=3, e=2
D12_T, D12_DET, D12_P, D12_E = 4, 1, 3, 2


def _check_legacy_consistency() -> dict[str, Any]:
    failures: list[str] = []

    # [434] Fibonacci: det=-1 family, t=1, p=5 must give e=4 by C2
    lam0_fib = _lambda0(FIB_T, FIB_P)
    e_fib = _order_mod_p(lam0_fib, FIB_P)
    if e_fib != FIB_E:
        failures.append(
            f"[434] Fibonacci p={FIB_P}: e={e_fib}, expected {FIB_E}"
        )
    if (lam0_fib * lam0_fib) % FIB_P != FIB_P - 1:
        failures.append(
            f"[434] Fibonacci p={FIB_P}: lambda0^2 != -1 mod {FIB_P}"
        )

    # [435] D12/p=3: det=+1 family, t=4, p=3; p|(t+2)=6 so expected e=2
    if (D12_T + 2) % D12_P != 0:
        failures.append(
            f"[435] D12: {D12_P} should divide (t+2)={D12_T+2} but does not"
        )
    lam0_d12 = _lambda0(D12_T, D12_P)
    e_d12 = _order_mod_p(lam0_d12, D12_P)
    if e_d12 != D12_E:
        failures.append(
            f"[435] D12 p={D12_P}: e={e_d12}, expected {D12_E}"
        )
    if lam0_d12 != D12_P - 1:
        failures.append(
            f"[435] D12 p={D12_P}: lambda0={lam0_d12} != {D12_P-1} (=-1 mod {D12_P})"
        )

    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    r1 = _check_det_plus1_e_rigidity(T_PLUS1_MAX)
    checks["DET_PLUS1_E_RIGIDITY"] = r1["ok"]
    details["DET_PLUS1_E_RIGIDITY_COVERAGE"] = {"pairs_tested": r1["pairs_tested"], "t_max": T_PLUS1_MAX}
    if not r1["ok"]:
        details["DET_PLUS1_E_RIGIDITY"] = r1["failures"]

    r2 = _check_det_minus1_e_rigidity(T_MINUS1_MAX)
    checks["DET_MINUS1_E_RIGIDITY"] = r2["ok"]
    details["DET_MINUS1_E_RIGIDITY_COVERAGE"] = {"pairs_tested": r2["pairs_tested"], "t_max": T_MINUS1_MAX}
    if not r2["ok"]:
        details["DET_MINUS1_E_RIGIDITY"] = r2["failures"]

    r3 = _check_no_third_value_global(T_PLUS1_MAX, T_MINUS1_MAX)
    checks["NO_THIRD_VALUE_GLOBAL"] = r3["ok"]
    details["NO_THIRD_VALUE_GLOBAL_SUMMARY"] = {
        "observed_plus1_e_values": r3["observed_plus1_e_values"],
        "observed_minus1_e_values": r3["observed_minus1_e_values"],
    }
    if not r3["ok"]:
        details["NO_THIRD_VALUE_GLOBAL"] = r3["failures"]

    r4 = _check_branch_mechanism_exact(T_PLUS1_MAX, T_MINUS1_MAX)
    checks["BRANCH_MECHANISM_EXACT"] = r4["ok"]
    details["BRANCH_MECHANISM_EXACT_COVERAGE"] = {"checked": r4["checked"]}
    if not r4["ok"]:
        details["BRANCH_MECHANISM_EXACT"] = r4["failures"]

    r5 = _check_legacy_consistency()
    checks["LEGACY_CONSISTENCY"] = r5["ok"]
    if not r5["ok"]:
        details["LEGACY_CONSISTENCY"] = r5["failures"]

    ok = all(checks.values())
    result: dict[str, Any] = {"ok": ok, "checks": checks}
    if details:
        result["details"] = details
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "id": "DET_PLUS1_FIB_P5_EXCLUDED",
        "description": "Fibonacci p=5 is det=-1, NOT det=+1; 5 does not divide (t+2) or (t-2) for t=1 (D=t^2+4=-3... irrelevant, it is det=-1 family)",
        "expected": True,
        "fn": lambda: (1 * 1 + 4) % 5 == 0 and (1 * 1 - 4) < 0,
    },
    {
        "id": "DET_PLUS1_D12_P3_BRANCH",
        "description": "D12 t=4, p=3: 3 divides (t+2)=6 NOT (t-2)=2 -> branch=(t+2) -> e=2",
        "expected": True,
        "fn": lambda: (4 + 2) % 3 == 0 and (4 - 2) % 3 != 0,
    },
    {
        "id": "DET_PLUS1_LAMBDA0_MINUS1",
        "description": "D12 t=4, p=3: lambda0 = 4*inv(2) mod 3 = 4*2 mod 3 = 8 mod 3 = 2 = -1 mod 3",
        "expected": True,
        "fn": lambda: _lambda0(4, 3) == 2,
    },
    {
        "id": "DET_MINUS1_FIB_LAMBDA0_SQ_NEG1",
        "description": "Fibonacci t=1, p=5: lambda0=3, lambda0^2=9=4=-1 mod 5",
        "expected": True,
        "fn": lambda: (_lambda0(1, 5) ** 2) % 5 == 4,
    },
    {
        "id": "DET_PLUS1_SPECIFIC_COLLAPSE",
        "description": "det+1 t=5, D=21=3*7; p=3 divides (t-2)=3 -> e=1; p=7 divides (t+2)=7 -> e=2",
        "expected": True,
        "fn": lambda: (
            _order_mod_p(_lambda0(5, 3), 3) == 1 and
            _order_mod_p(_lambda0(5, 7), 7) == 2
        ),
    },
    {
        "id": "DET_MINUS1_LARGER_SAMPLE",
        "description": "det=-1 t=7, D=53; 53 is prime, lambda0=7*inv(2)mod53=7*27mod53=189mod53=30; 30^2 mod 53=900 mod 53=900-16*53=900-848=52=-1 mod 53; e=4",
        "expected": True,
        "fn": lambda: (
            _lambda0(7, 53) == 30 and
            (30 * 30) % 53 == 52 and
            _order_mod_p(30, 53) == 4
        ),
    },
    {
        "id": "WRONG_CLAIM_E_FREE",
        "description": "DESIGNED FAIL: claim e=3 is ever observed in det=+1 sweep t=3..50 (it never is -- this fixture must fail)",
        "expected": False,
        "fn": lambda: any(
            _order_mod_p(_lambda0(t, p), p) == 3
            for t in range(3, 51)
            for p in _prime_factors(t * t - 4) if p != 2 and t * t - 4 > 0
        ),
    },
]


def run_fixtures() -> dict[str, Any]:
    results = {}
    for f in FIXTURES:
        try:
            actual = f["fn"]()
            passed = actual == f["expected"]
        except Exception as exc:
            actual = f"ERROR: {exc}"
            passed = False
        results[f["id"]] = {
            "description": f["description"],
            "expected": f["expected"],
            "actual": actual,
            "passed": passed,
        }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QA Witt Tower Ramified Prime e-Value Rigidity cert [436] validator"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixtures", action="store_true")
    args = parser.parse_args()

    if args.fixtures:
        out = {"fixtures": run_fixtures()}
        print(json.dumps(out, indent=2))
        sys.exit(0)

    result = run_self_test()
    if args.self_test:
        fixture_result = run_fixtures()
        result["fixtures"] = fixture_result
        fixture_pass = sum(1 for v in fixture_result.values() if v["passed"])
        fixture_total = len(fixture_result)
        result["fixture_summary"] = f"{fixture_pass}/{fixture_total} passed"

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
