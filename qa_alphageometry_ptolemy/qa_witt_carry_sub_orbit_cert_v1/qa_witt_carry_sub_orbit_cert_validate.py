# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical Witt vector theory and algebraic number theory; Serre (1979) Local Fields doi.org/10.1007/978-1-4757-5673-9 Ch.II §4 (Witt vectors, Teichmuller representatives); Neukirch (1999) ISBN 978-3-540-65399-8 §II.5 (formal groups, Witt vectors); Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano periods) -->
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — integer arithmetic in Z[phi]/(9); all ring state is exact integer; no float; no continuous QA inputs"
"""Cert [387]: QA Witt Carry Sub-Orbit Invariant — Teichmuller Coset Structure.

PRIMARY CLAIM:
  In Z[phi]/(9) = W_2(GF(9)) (Witt vectors of length 2 over GF(9)):

  (C1) THREE_SUB_ORBITS:
       The 72 Cosmos pairs decompose into exactly 3 sigma-orbits of period 24:
         C0 = sigma-orbit of (1,0), C1 = sigma-orbit of (2,0), C2 = sigma-orbit of (4,0).
       sigma: (a,b) -> (a+b mod 9, a)  [QA Fibonacci shift in ring coordinates]

  (C2) PHI_ORDER_24:
       phi = (0,1) has multiplicative order 24 in (Z[phi]/9)^*, and
       (Z[phi]/9)^* = <phi>  u  2*<phi>  u  4*<phi>  (disjoint, sizes 24 each).
       The sigma-orbits biject with the three right-cosets of <phi> via the
       coordinate swap rho: (a,b) <-> (b,a):
         C_i  =  {rho(x) : x in c_i * <phi>}  where (c0,c1,c2) = ((1,0),(2,0),(4,0)).

  (C3) DIRECT_PRODUCT:
       (Z[phi]/9)^* = T x U_1  (internal direct product; gcd(8,9)=1)
         T   = {x in (Z[phi]/9)^* : x^8 = (1,0)}  -- Teichmuller subgroup, |T|=8, T cyclic ~= Z/8
         U_1 = {x in (Z[phi]/9)^* : x = (1,0) mod 3} -- 1-unit group, |U_1|=9, ~= (Z/3)^2
       Every unit decomposes uniquely as x = tau(x) * u(x) with tau(x) in T, u(x) in U_1.

  (C4) WITT_CARRY_INVARIANT:
       Define J: (Z[phi]/9)^* -> {0,1,2} by
         J(a,b) = coset_idx(rho^{-1}(a,b)) = coset_idx((b,a))
       where coset_idx(x) labels which coset of <u_phi> in U_1 contains u(x):
         <u_phi> = {(1,0), (4,3), (7,6)}  (the U_1-component of phi, order 3)
         u(x) in {coset 0 <-> J=0, coset 1 <-> J=1, coset 2 <-> J=2}.
       J is CONSTANT on each sigma-orbit and takes a distinct value on each:
         J = 0 on C0 (orbit of (1,0)),  J = 1 on C1 (orbit of (2,0)),  J = 2 on C2 (orbit of (4,0)).

  (C5) TEICHMÜLLER_HIT:
       Each sigma-orbit hits each of the 8 Teichmuller classes (elements of GF(9)^* under
       x -> x mod 3) exactly 3 times. The T-class sequence (a mod 3, b mod 3) has period 8
       on all three orbits, and the THREE ORBITS ARE DISTINGUISHED ENTIRELY BY J (Witt carry),
       not by their T-class sequences (which are identical up to cyclic shift).

INTERPRETATION:
  phi = tau(phi) * u_phi where tau(phi) in T has order 8 and u_phi in U_1 has order 3.
  The three Cosmos sub-orbits correspond to the three cosets of the cyclic group <u_phi>
  in U_1 ~= (Z/3)^2. The 'Witt carry' J is the second ghost coordinate of x in the
  Witt ring W_2(GF(9)) = Z[phi]/(9), evaluated mod <u_phi>.

  This resolves why T alone cannot distinguish the sub-orbits: T ~ GF(9)^* images are
  identical across all three, and the distinction lives purely in the U_1 = 1-unit layer
  of the Witt vector.

LINEAGE: extends cert [385] (orbit = prime ideal filtration) and cert [386] (inert primes);
         lays groundwork for cert [388] (Witt invariant for split primes).
SOURCES: Serre (1979) Local Fields doi.org/10.1007/978-1-4757-5673-9;
         Neukirch (1999) ISBN 978-3-540-65399-8; Wall (1960) doi.org/10.1080/00029890.1960.11989541.
"""

import json
import sys
from typing import Any

M = 9  # QA modulus


# ---------------------------------------------------------------------------
# Ring Z[phi]/(9): elements (a,b), multiplication (a+b*phi)(c+d*phi)=(ac+bd)+(ad+bc+bd)*phi
# ---------------------------------------------------------------------------

def _ring_mul(x: tuple, y: tuple) -> tuple:
    a, b = x; c, d = y
    return ((a * c + b * d) % M, (a * d + b * c + b * d) % M)


def _ring_pow(x: tuple, n: int) -> tuple:
    r = (1, 0)
    for _ in range(n):
        r = _ring_mul(r, x)
    return r


def _ring_inv(x: tuple) -> tuple | None:
    for c in range(M):
        for d in range(M):
            if _ring_mul(x, (c, d)) == (1, 0):
                return (c, d)
    return None


def _all_units() -> frozenset:
    return frozenset((a, b) for a in range(M) for b in range(M) if _ring_inv((a, b)) is not None)


def _teich_group() -> frozenset:
    return frozenset(x for x in _all_units() if _ring_pow(x, 8) == (1, 0))


def _u1_group() -> frozenset:
    return frozenset((a, b) for (a, b) in _all_units() if (a % 3, b % 3) == (1, 0))


# ---------------------------------------------------------------------------
# Teichmuller decomposition and coset invariant
# ---------------------------------------------------------------------------

def _teich_lift(x: tuple, T: frozenset) -> tuple:
    x_mod3 = (x[0] % 3, x[1] % 3)
    for t in T:
        if (t[0] % 3, t[1] % 3) == x_mod3:
            return t
    raise ValueError(f"No Teichmuller lift for {x}")


def _coset_idx(x: tuple, T: frozenset, U1: frozenset) -> int:
    """Return coset index in {0,1,2} of x in (Z[phi]/9)^* / <phi>. x must be a unit."""
    t = _teich_lift(x, T)
    u = _ring_mul(_ring_inv(t), x)
    a, b = u
    a3 = (a - 1) // 3 % 3
    b3 = b // 3 % 3
    return (b3 - a3) % 3


def _orbit_idx(a: int, b: int, T: frozenset, U1: frozenset) -> int:
    """Sub-orbit index for (a,b): apply coset_idx to rho^{-1}(a,b) = (b,a)."""
    return _coset_idx((b, a), T, U1)


# ---------------------------------------------------------------------------
# QA step (sigma) and orbit computation
# ---------------------------------------------------------------------------

def _sigma(a: int, b: int) -> tuple:
    return ((a + b) % M, a)


def _sigma_orbit(start: tuple) -> list:
    orbit, cur = [], start
    while True:
        orbit.append(cur)
        cur = _sigma(*cur)
        if cur == start:
            break
    return orbit


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_three_sub_orbits() -> dict[str, Any]:
    """C1: Cosmos = 3 sigma-orbits of period 24."""
    units = _all_units()
    cosmos = list(units)
    seen: set = set()
    orbits = []
    for start in sorted(cosmos):
        if start in seen:
            continue
        orb = _sigma_orbit(start)
        orbits.append(orb)
        seen.update(orb)

    periods = [len(o) for o in orbits]
    errors = []
    if len(orbits) != 3:
        errors.append(f"expected 3 orbits, got {len(orbits)}")
    if set(periods) != {24}:
        errors.append(f"expected all periods=24, got {set(periods)}")
    if len(seen) != 72:
        errors.append(f"covered {len(seen)} pairs, expected 72")

    ok = len(errors) == 0
    return {
        "name": "THREE_SUB_ORBITS",
        "ok": ok,
        "num_orbits": len(orbits),
        "periods": periods,
        "total_pairs": len(seen),
        "errors": errors,
        "detail": "Cosmos = 3 sigma-orbits, each period 24" if ok else f"FAIL: {errors}",
    }


def _check_phi_order_24() -> dict[str, Any]:
    """C2: ord(phi)=24; (Z[phi]/9)^* = <phi> union 2<phi> union 4<phi>."""
    phi = (0, 1)
    T = _teich_group()
    U1 = _u1_group()

    ord_phi = next((k for k in range(1, 73) if _ring_pow(phi, k) == (1, 0)), -1)

    orbit_phi = {_ring_pow(phi, k) for k in range(1, ord_phi + 1)}
    orbit_2 = {_ring_mul((2, 0), x) for x in orbit_phi}
    orbit_4 = {_ring_mul((4, 0), x) for x in orbit_phi}
    union = orbit_phi | orbit_2 | orbit_4
    units = _all_units()

    errors = []
    if ord_phi != 24:
        errors.append(f"ord(phi)={ord_phi}, expected 24")
    if union != units:
        errors.append(f"union size={len(union)}, expected 72")
    if len(orbit_phi & orbit_2) > 0 or len(orbit_phi & orbit_4) > 0 or len(orbit_2 & orbit_4) > 0:
        errors.append("cosets not disjoint")

    ok = len(errors) == 0
    return {
        "name": "PHI_ORDER_24",
        "ok": ok,
        "ord_phi": ord_phi,
        "coset_sizes": [len(orbit_phi), len(orbit_2), len(orbit_4)],
        "covers_all_units": union == units,
        "errors": errors,
        "detail": "ord(phi)=24; (Z[phi]/9)^* = <phi> u 2<phi> u 4<phi>" if ok else f"FAIL: {errors}",
    }


def _check_direct_product() -> dict[str, Any]:
    """C3: (Z[phi]/9)^* = T x U1, internal direct product."""
    T = _teich_group()
    U1 = _u1_group()
    units = _all_units()

    errors = []
    if len(T) != 8:
        errors.append(f"|T|={len(T)}, expected 8")
    if len(U1) != 9:
        errors.append(f"|U1|={len(U1)}, expected 9")
    if T & U1 != {(1, 0)}:
        errors.append(f"T ∩ U1 = {T & U1}, expected {{(1,0)}}")

    tu1 = {_ring_mul(t, u) for t in T for u in U1}
    if tu1 != units:
        errors.append(f"T*U1 size={len(tu1)}, expected 72")

    t_orders = sorted({next(k for k in range(1, 9) if _ring_pow(x, k) == (1, 0)) for x in T})
    u1_orders = sorted({next(k for k in range(1, 10) if _ring_pow(x, k) == (1, 0)) for x in U1})

    ok = len(errors) == 0
    return {
        "name": "DIRECT_PRODUCT",
        "ok": ok,
        "T_size": len(T),
        "U1_size": len(U1),
        "T_order_set": t_orders,
        "U1_order_set": u1_orders,
        "T_times_U1_equals_units": tu1 == units,
        "T_intersect_U1": sorted(T & U1),
        "errors": errors,
        "detail": "(Z[phi]/9)^* = T(order-8) x U1(order-9, exponent-3)" if ok else f"FAIL: {errors}",
    }


def _check_witt_carry_invariant() -> dict[str, Any]:
    """C4: J(a,b) = coset_idx((b,a)) is constant on each sigma-orbit, distinct across orbits."""
    T = _teich_group()
    U1 = _u1_group()
    units = _all_units()

    seen: set = set()
    orbit_data = []
    for start in sorted(units):
        if start in seen:
            continue
        orb = _sigma_orbit(start)
        seen.update(orb)
        indices = {_orbit_idx(a, b, T, U1) for a, b in orb}
        orbit_data.append({"start": start, "size": len(orb), "J_values": sorted(indices)})

    errors = []
    for od in orbit_data:
        if len(od["J_values"]) != 1:
            errors.append(f"orbit {od['start']}: J not constant, got {od['J_values']}")

    j_values = {od["J_values"][0] for od in orbit_data if len(od["J_values"]) == 1}
    if j_values != {0, 1, 2}:
        errors.append(f"J values across orbits = {j_values}, expected {{0,1,2}}")

    ok = len(errors) == 0
    return {
        "name": "WITT_CARRY_INVARIANT",
        "ok": ok,
        "orbits": orbit_data,
        "distinct_J_values": sorted(j_values) if ok else [],
        "errors": errors,
        "detail": "J(a,b)=coset_idx((b,a)) constant on each orbit; takes values {0,1,2}" if ok else f"FAIL: {errors}",
    }


def _check_teichmuller_hit() -> dict[str, Any]:
    """C5: each sigma-orbit hits each of the 8 T-classes exactly 3 times."""
    T = _teich_group()
    U1 = _u1_group()
    units = _all_units()

    t_classes = {(t[0] % 3, t[1] % 3) for t in T}
    if len(t_classes) != 8:
        return {
            "name": "TEICHMÜLLER_HIT",
            "ok": False,
            "detail": f"FAIL: expected 8 T-classes, got {len(t_classes)}",
        }

    seen: set = set()
    orbit_results = []
    errors = []

    for start in sorted(units):
        if start in seen:
            continue
        orb = _sigma_orbit(start)
        seen.update(orb)
        orb_j = next(iter({_orbit_idx(a, b, T, U1) for a, b in orb}))

        class_counts = {}
        for a, b in orb:
            cls = (a % 3, b % 3)
            class_counts[cls] = class_counts.get(cls, 0) + 1

        hits = set(class_counts.values())
        if hits != {3}:
            errors.append(f"orbit J={orb_j}: hit counts not all 3, got {hits}")
        if set(class_counts.keys()) != t_classes:
            errors.append(f"orbit J={orb_j}: missing T-classes")

        orbit_results.append({"J": orb_j, "t_class_hits": sorted(class_counts.values())})

    ok = len(errors) == 0
    return {
        "name": "TEICHMÜLLER_HIT",
        "ok": ok,
        "t_classes_count": len(t_classes),
        "orbit_results": orbit_results,
        "errors": errors,
        "detail": "Each orbit hits each of 8 T-classes exactly 3 times" if ok else f"FAIL: {errors}",
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "id": "ORBIT1_J0",
        "expect": "PASS",
        "check": lambda T, U1: all(
            next(iter({_orbit_idx(a, b, T, U1) for a, b in _sigma_orbit((1, 0))})) == 0
            for _ in [0]
        ),
        "desc": "orbit of (1,0) has J=0",
    },
    {
        "id": "ORBIT2_J1",
        "expect": "PASS",
        "check": lambda T, U1: all(
            next(iter({_orbit_idx(a, b, T, U1) for a, b in _sigma_orbit((2, 0))})) == 1
            for _ in [0]
        ),
        "desc": "orbit of (2,0) has J=1",
    },
    {
        "id": "ORBIT3_J2",
        "expect": "PASS",
        "check": lambda T, U1: all(
            next(iter({_orbit_idx(a, b, T, U1) for a, b in _sigma_orbit((4, 0))})) == 2
            for _ in [0]
        ),
        "desc": "orbit of (4,0) has J=2",
    },
    {
        "id": "TCLASS_SAME_ALL_ORBITS",
        "expect": "PASS",
        "check": lambda T, U1: (
            {(a % 3, b % 3) for a, b in _sigma_orbit((1, 0))}
            == {(a % 3, b % 3) for a, b in _sigma_orbit((2, 0))}
            == {(a % 3, b % 3) for a, b in _sigma_orbit((4, 0))}
        ),
        "desc": "all 3 orbits hit the same 8 T-classes (J not T-class distinguishes them)",
    },
    {
        "id": "J_WRONG_ORBIT",
        "expect": "FAIL",
        "check": lambda T, U1: _orbit_idx(2, 0, T, U1) == 0,
        "desc": "(2,0) is NOT in orbit J=0 (fixture must FAIL)",
    },
    {
        "id": "U1_EXPONENT_3",
        "expect": "PASS",
        "check": lambda T, U1: all(
            _ring_pow(u, 3) == (1, 0) for u in U1 if u != (1, 0)
        ),
        "desc": "all non-identity elements of U1 have order 3 (U1 = (Z/3)^2)",
    },
]


def _run_self_test() -> dict[str, Any]:
    T = _teich_group()
    U1 = _u1_group()

    checks = [
        _check_three_sub_orbits(),
        _check_phi_order_24(),
        _check_direct_product(),
        _check_witt_carry_invariant(),
        _check_teichmuller_hit(),
    ]

    fixture_results = []
    for fix in FIXTURES:
        result = fix["check"](T, U1)
        passed = result if fix["expect"] == "PASS" else not result
        fixture_results.append({
            "id": fix["id"],
            "expect": fix["expect"],
            "actual": "PASS" if result else "FAIL",
            "ok": passed,
            "desc": fix["desc"],
        })

    all_checks_ok = all(c["ok"] for c in checks)
    all_fixtures_ok = all(f["ok"] for f in fixture_results)

    return {
        "ok": all_checks_ok and all_fixtures_ok,
        "checks": {c["name"]: c["ok"] for c in checks},
        "fixtures": fixture_results,
        "summary": {
            "checks_pass": sum(c["ok"] for c in checks),
            "checks_total": len(checks),
            "fixtures_pass": sum(f["ok"] for f in fixture_results),
            "fixtures_total": len(fixture_results),
        },
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = _run_self_test()
        print(json.dumps(result))
    else:
        result = _run_self_test()
        print(json.dumps(result, indent=2))
    sys.exit(0 if result["ok"] else 1)
