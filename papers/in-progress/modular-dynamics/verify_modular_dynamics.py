"""
verify_modular_dynamics.py
==========================
Machine-checkable verification of computational claims in:

  "Orbit Structure of the Fibonacci Matrix over Finite Quadratic Rings"

Run:  python3 verify_modular_dynamics.py
All assertions must pass (exit 0).  Takes < 2 seconds.  Requires Python 3.6+, stdlib only.

Claims verified
---------------
V1   Inert primes {3,7,13,17,23,37,43}: orbit count and sizes match theorem.
V2   ord(F mod p) = 2(p+1) for each inert prime p (= Pisano period = ord(phi in GF(p^2)*)).
V3   Norm identity: N(b+e*phi) = det(M(b+e*phi)) for b,e in {-4..4}.
V4   det(F) = N(phi) = -1; F represents x*phi in Z[phi] (not x*phi^2, not T=F^2).
V5   Split prime p=11: roots {4,8}; ord(4)=5, ord(8)=10; 14 orbits of sizes {5,5,10^12}.
V6   Ramified prime p=5: F^5 equiv 3I (mod 5); F^20 equiv I (mod 5); orbit sizes [1,4,20].
V7   Eigenspace of F mod 5: E = {(t, 3t) : t in F_5}; its nonzero part is the size-4 orbit.
V8   Hensel lift mod-3 -> mod-9: 1 cosmos orbit (size 8) at mod-3 lifts to 3 (size 24) at mod-9.
V9   Frobenius: phi^p = first column of F^p mod p equals (1, p-1) for inert p in {3,7,13}.
V10  Total orbit-size sums equal p^2 for all tested primes.
"""

import sys
from math import gcd

# ── helpers ──────────────────────────────────────────────────────────────────

def mat_mul_mod(A, B, m):
    return [
        [sum(A[i][k] * B[k][j] for k in range(2)) % m for j in range(2)]
        for i in range(2)
    ]

def mat_pow_mod(A, n, m):
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in A]
    while n > 0:
        if n % 2 == 1:
            result = mat_mul_mod(result, base, m)
        base = mat_mul_mod(base, base, m)
        n //= 2
    return result

F = [[0, 1], [1, 1]]

def apply_F(b, e, m):
    return e % m, (b + e) % m

def get_orbits(m):
    """Return list of all orbits of F on (Z/mZ)^2."""
    visited = {}
    orbits = []
    for b in range(m):
        for e in range(m):
            if (b, e) not in visited:
                orbit = []
                cur = (b, e)
                while cur not in visited:
                    visited[cur] = len(orbits)
                    orbit.append(cur)
                    cur = apply_F(*cur, m)
                orbits.append(orbit)
    return orbits

def mat_order(A, m, max_order=10000):
    """Order of matrix A in GL_2(Z/mZ)."""
    I = [[1, 0], [0, 1]]
    cur = mat_mul_mod(A, [[1,0],[0,1]], m)  # copy
    cur = [row[:] for row in A]
    for k in range(1, max_order + 1):
        if cur == I:
            return k
        cur = mat_mul_mod(cur, A, m)
    raise ValueError(f"Order > {max_order}")

def is_irreducible_mod_p(p):
    """True iff x^2 - x - 1 has no root in F_p (inert case)."""
    return all((x*x - x - 1) % p != 0 for x in range(p))

def roots_mod_p(p):
    """Roots of x^2 - x - 1 in F_p."""
    return [x for x in range(p) if (x*x - x - 1) % p == 0]

def mult_order(a, p):
    """Multiplicative order of a in (Z/pZ)*."""
    a = a % p
    assert a != 0, "a must be nonzero"
    cur = 1
    for k in range(1, p):
        cur = (cur * a) % p
        if cur == 1:
            return k
    raise ValueError(f"Order not found for {a} mod {p}")

INERT_PRIMES = [3, 7, 13, 17, 23, 37, 43]

# ── V1: inert prime orbit structure ──────────────────────────────────────────

def verify_V1():
    for p in INERT_PRIMES:
        assert is_irreducible_mod_p(p), f"p={p} is not inert"
        orbits = get_orbits(p)
        sizes = sorted(len(o) for o in orbits)
        expected_orbit_size = 2 * (p + 1)
        expected_n_cosmos   = (p - 1) // 2
        expected_sizes = sorted([1] + [expected_orbit_size] * expected_n_cosmos)
        assert sizes == expected_sizes, (
            f"p={p}: orbit sizes {sizes} != expected {expected_sizes}"
        )
    print(f"V1  PASS  inert {INERT_PRIMES}: orbit sizes match theorem (1 fixed + (p-1)/2 cosmos of size 2(p+1))")

# ── V2: ord(F mod p) = 2(p+1) ────────────────────────────────────────────────

def verify_V2():
    for p in INERT_PRIMES:
        expected = 2 * (p + 1)
        ord_F = mat_order(F, p, max_order=expected + 1)
        assert ord_F == expected, f"p={p}: ord(F mod p)={ord_F}, expected {expected}"
    print("V2  PASS  ord(F mod p) = 2(p+1) for all inert primes (= Pisano period = ord(phi in GF(p^2)*))")

# ── V3: norm = determinant of multiplication matrix ──────────────────────────

def norm(b, e):
    return b*b + b*e - e*e

def det_M(b, e):
    # M(b+e*phi) = [[b, e], [e, b+e]]; det = b*(b+e) - e*e = b^2+be-e^2
    return b * (b + e) - e * e

def verify_V3():
    for b in range(-4, 5):
        for e in range(-4, 5):
            assert norm(b, e) == det_M(b, e), (
                f"N({b},{e}) = {norm(b,e)} != det(M) = {det_M(b,e)}"
            )
    print("V3  PASS  N(b+e*phi) = det(M(b+e*phi)) = b^2+be-e^2 for b,e in {-4..4}")

# ── V4: det(F)=N(phi)=-1; F represents x*phi ─────────────────────────────────

def verify_V4():
    # det(F) = 0*1 - 1*1 = -1
    det_F = F[0][0]*F[1][1] - F[0][1]*F[1][0]
    assert det_F == -1, f"det(F) = {det_F}"
    # N(phi) = N(0 + 1*phi) = 0^2 + 0*1 - 1^2 = -1
    assert norm(0, 1) == -1
    # F represents x*phi: phi*(b+e*phi) = e + (b+e)*phi i.e. (b,e) -> (e, b+e)
    for b in range(6):
        for e in range(6):
            fb, fe = apply_F(b, e, m=100)   # large modulus to avoid wrap
            assert (fb, fe) == (e, b + e), f"F({b},{e}) = ({fb},{fe}), expected ({e},{b+e})"
    # T = F^2 represents x*phi^2 (distinct from x*phi)
    T = mat_mul_mod(F, F, 100)
    assert T == [[1,1],[1,2]], f"T=F^2 = {T}"
    print("V4  PASS  det(F)=N(phi)=-1; F=x*phi; T=F^2=x*phi^2 (distinct actions)")

# ── V5: split prime p=11 ──────────────────────────────────────────────────────

def verify_V5():
    p = 11
    # x^2-x-1 splits over F_11
    rs = roots_mod_p(p)
    assert sorted(rs) == [4, 8], f"Roots of x^2-x-1 mod 11: {rs}"
    phi0, psi0 = 4, 8
    # Vieta: phi0+psi0 = 1 mod 11, phi0*psi0 = -1 mod 11
    assert (phi0 + psi0) % p == 1
    assert (phi0 * psi0) % p == p - 1   # -1 mod p
    # Multiplicative orders
    d1 = mult_order(phi0, p)
    d2 = mult_order(psi0, p)
    assert d1 == 5,  f"ord(4) mod 11 = {d1}, expected 5"
    assert d2 == 10, f"ord(8) mod 11 = {d2}, expected 10"
    lcm12 = d1 * d2 // gcd(d1, d2)
    assert lcm12 == 10
    # Theorem orbit counts
    n_fixed = 1
    n_ax1   = (p - 1) // d1    # 2 orbits of size 5
    n_ax2   = (p - 1) // d2    # 1 orbit of size 10
    n_gen   = (p - 1)**2 // lcm12  # 10 orbits of size 10
    total_theorem = n_fixed + n_ax1 + n_ax2 + n_gen
    assert total_theorem == 14, f"Theorem predicts {total_theorem} orbits, expected 14"
    # Direct computation
    orbits = get_orbits(p)
    assert len(orbits) == 14, f"Direct count: {len(orbits)} orbits"
    sizes = sorted(len(o) for o in orbits)
    expected = sorted([1] + [d1]*n_ax1 + [d2]*n_ax2 + [lcm12]*n_gen)
    assert sizes == expected, f"Orbit sizes mismatch:\n  got      {sizes}\n  expected {expected}"
    print(f"V5  PASS  p=11 split: roots={{4,8}}, ord(4)=5, ord(8)=10; 14 orbits "
          f"({n_ax1} size-5 + {n_ax2+n_gen} size-10 + 1 fixed)")

# ── V6: ramified prime p=5 ────────────────────────────────────────────────────

def verify_V6():
    p = 5
    # F^5 equiv 3I mod 5
    F5 = mat_pow_mod(F, 5, p)
    assert F5 == [[3, 0], [0, 3]], f"F^5 mod 5 = {F5}"
    # F^20 equiv I mod 5
    F20 = mat_pow_mod(F, 20, p)
    assert F20 == [[1, 0], [0, 1]], f"F^20 mod 5 = {F20}"
    # Orbit sizes {1, 4, 20}
    orbits = get_orbits(p)
    sizes = sorted(len(o) for o in orbits)
    assert sizes == [1, 4, 20], f"Orbit sizes mod 5: {sizes}"
    # No orbit of size 2 or 5 exists
    assert all(s in {1, 4, 20} for s in sizes)
    print("V6  PASS  p=5 ramified: F^5 equiv 3I; F^20 equiv I; orbit sizes [1,4,20]")

# ── V7: eigenspace and size-4 orbit ──────────────────────────────────────────

def verify_V7():
    p = 5
    # Eigenspace: ker(F - 3I) mod 5 = {(b,e): e = 3b mod 5}
    eigenspace = {(b, (3*b) % p) for b in range(p)}
    # F maps each eigenspace element to 3 times itself (mod 5)
    for b, e in eigenspace:
        fb, fe = apply_F(b, e, p)
        assert (fb, fe) == ((3*b) % p, (3*e) % p), (
            f"F({b},{e}) = ({fb},{fe}), expected ({(3*b)%p},{(3*e)%p})"
        )
    # The nonzero eigenspace elements form the size-4 orbit
    cycle = []
    cur = (1, 3)
    for _ in range(4):
        cycle.append(cur)
        cur = apply_F(*cur, p)
    assert cur == (1, 3), f"Orbit of (1,3) has period != 4; cur after 4 steps = {cur}"
    orbit_4 = set(cycle)
    assert orbit_4 == {(1, 3), (3, 4), (4, 2), (2, 1)}, f"Size-4 orbit: {orbit_4}"
    assert orbit_4 == eigenspace - {(0, 0)}, "Nonzero eigenspace != size-4 orbit"
    print("V7  PASS  E={(t,3t): t in F_5}; nonzero E = {(1,3),(3,4),(4,2),(2,1)} is the size-4 orbit")

# ── V8: Hensel lift mod-3 -> mod-9 ───────────────────────────────────────────

def verify_V8():
    # mod-3: 1 cosmos orbit (size 8) + 1 fixed point
    orbits3 = get_orbits(3)
    sizes3 = sorted(len(o) for o in orbits3)
    assert sizes3 == [1, 8], f"mod-3 orbit sizes: {sizes3}"
    cosmos3_idx = next(i for i, o in enumerate(orbits3) if len(o) == 8)
    mod3_cosmos = set(orbits3[cosmos3_idx])
    # mod-9: 3 cosmos orbits (size 24) + 1 Tribonacci orbit (size 8) + 1 fixed
    orbits9 = get_orbits(9)
    sizes9 = sorted(len(o) for o in orbits9)
    assert sizes9 == [1, 8, 24, 24, 24], f"mod-9 orbit sizes: {sizes9}"
    cosmos9 = [o for o in orbits9 if len(o) == 24]
    assert len(cosmos9) == 3
    # Every mod-9 cosmos element reduces mod 3 into the mod-3 cosmos orbit
    reductions = {(b % 3, e % 3) for o in cosmos9 for (b, e) in o}
    assert reductions == mod3_cosmos, (
        f"Reduction mismatch:\n  got {reductions}\n  expected {mod3_cosmos}"
    )
    # Each mod-3 cosmos element is hit exactly 9 times (72 elements / 8 residues)
    counts = {}
    for o in cosmos9:
        for (b, e) in o:
            key = (b % 3, e % 3)
            counts[key] = counts.get(key, 0) + 1
    assert all(v == 9 for v in counts.values()), f"Unequal hit counts: {counts}"
    print("V8  PASS  Hensel: 1 cosmos orbit (size 8) at mod-3 lifts to 3 (size 24) at mod-9; "
          "each mod-3 residue hit exactly 9 times")

# ── V9: Frobenius phi^p = (1, p-1) = 1-phi for inert p ──────────────────────

def verify_V9():
    # phi^p in Z[phi]/p is represented by F^p * (1,0)^T = first column of F^p mod p.
    # (Since F^k * (1,0)^T represents phi^k * 1 = phi^k.)
    # Frobenius sends phi -> 1-phi = 1 + (-1)*phi, i.e. (b=1, e=p-1).
    for p in [3, 7, 13]:
        Fp = mat_pow_mod(F, p, p)
        b, e = Fp[0][0], Fp[1][0]   # first column
        assert b == 1 and e == p - 1, (
            f"p={p}: phi^p -> ({b},{e}) mod p, expected (1,{p-1})"
        )
    # Cross-check: phi^(p+1) = phi * phi^p = phi * (1-phi) = phi - phi^2
    #            = phi - (phi+1) = -1 -> (b=-1, e=0) = (p-1, 0) mod p
    for p in [3, 7, 13]:
        Fp1 = mat_pow_mod(F, p + 1, p)
        b, e = Fp1[0][0], Fp1[1][0]
        assert (b, e) == (p - 1, 0), (
            f"p={p}: phi^(p+1) -> ({b},{e}), expected ({p-1},0) = -1"
        )
    print("V9  PASS  phi^p equiv 1-phi (mod p); phi^(p+1) equiv -1 (mod p) for p in {3,7,13}")

# ── V10: total orbit sizes = p^2 ─────────────────────────────────────────────

def verify_V10():
    for p in INERT_PRIMES + [5, 11]:
        orbits = get_orbits(p)
        total = sum(len(o) for o in orbits)
        assert total == p * p, f"p={p}: sum of orbit sizes = {total}, expected {p*p}"
    print(f"V10 PASS  sum of orbit sizes = p^2 for all tested primes {INERT_PRIMES + [5, 11]}")

# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    checks = [
        verify_V1, verify_V2, verify_V3, verify_V4, verify_V5,
        verify_V6, verify_V7, verify_V8, verify_V9, verify_V10,
    ]
    failures = []
    for fn in checks:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failures.append(fn.__name__)

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print("All 10 verification claims PASS.")
        sys.exit(0)
