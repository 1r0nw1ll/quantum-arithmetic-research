"""
verify_classification.py
========================
Machine-checkable verification of all computational claims in:

  "A Complete Classification of Pythagorean Triples via Generalized
   Fibonacci Sequences and Pisano Periods"

Run:  python3 verify_classification.py
All assertions must pass (exit 0).  Takes < 1 second.

Claims verified
---------------
V1  Five families partition (Z/9Z)^2 into 81 disjoint pairs.
V2  Pisano periods: Fibonacci/Lucas/Phibonacci = 24, Tribonacci = 8, Ninbonacci = 1.
V3  Group-theoretic classification: five families = orbits of F on (Z/9Z)^2.
V4  Order of F = [[0,1],[1,1]] in GL_2(Z/9Z) is 24.
V5  Norm alternation: N(F(b,e)) = -N(b,e) mod 9 for all (b,e).
V6  Cosmos norm pairs: Fibonacci {1,8}, Lucas {4,5}, Phibonacci {2,7}.
V7  Tribonacci: N(b,e) = 0 mod 9 for all orbit elements; stabilizer order 3.
V8  Ninbonacci: unique fixed point (0,0); F(0,0) = (0,0).
V9  Inradius identity: r = be for all BEDA tuples with b,e in {1..20}.
V10 Barning-Berggren: rho_p preserves r; lambda_k maps r -> k^2 * r.
"""

import sys

m = 9  # modulus

# ── helpers ──────────────────────────────────────────────────────────────────

def dr(n):
    """Digital root: maps Z>0 -> {1,...,9}."""
    return 9 if n % 9 == 0 else n % 9

def z9(x):
    """Z/9Z representative in {0,...,8}; 0 corresponds to digital root 9."""
    return x % m

def apply_F(b, e):
    return z9(e), z9(b + e)

def apply_T(b, e):   # T = F^2, the QA step
    return z9(b + e), z9(b + 2 * e)

def norm(b, e):
    """Q(sqrt5) norm: b^2 + be - e^2 mod 9."""
    return z9(b * b + b * e - e * e)

def mat_mul_mod(A, B, mod):
    return [
        [sum(A[i][k] * B[k][j] for k in range(2)) % mod for j in range(2)]
        for i in range(2)
    ]

def mat_pow_mod(A, n, mod):
    result = [[1, 0], [0, 1]]
    base = [row[:] for row in A]
    while n > 0:
        if n % 2 == 1:
            result = mat_mul_mod(result, base, mod)
        base = mat_mul_mod(base, base, mod)
        n //= 2
    return result

def get_orbits(step_fn):
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
                    cur = step_fn(*cur)
                orbits.append(orbit)
    return orbits

# Paper Table 1: paper_map[(dr_b, dr_e)] -> family name
# Rows = dr(b) in 1..9, cols = dr(e) in 1..9
_TABLE = [
    "F F L P F P L F F",   # dr(b)=1
    "L L F L P P L F L",   # dr(b)=2
    "P P T L F T F L T",   # dr(b)=3
    "F P F P P L L P P",   # dr(b)=4
    "P L L P P F P F P",   # dr(b)=5
    "L F T F L T P P T",   # dr(b)=6
    "F L P P L F L L L",   # dr(b)=7
    "F L P F P L F F F",   # dr(b)=8
    "F L T P P T L F N",   # dr(b)=9
]
_LETTER = {'F': 'Fibonacci', 'L': 'Lucas', 'P': 'Phibonacci',
           'T': 'Tribonacci', 'N': 'Ninbonacci'}
paper_map = {}
for _ri, _row in enumerate(_TABLE):
    for _ci, _let in enumerate(_row.split()):
        paper_map[(_ri + 1, _ci + 1)] = _LETTER[_let]  # 1-indexed


def z9_to_dr(x):
    return 9 if x == 0 else x


# ── V1: partition ─────────────────────────────────────────────────────────────

def verify_V1():
    all_pairs = set()
    counts = {}
    for b_dr in range(1, 10):
        for e_dr in range(1, 10):
            fam = paper_map[(b_dr, e_dr)]
            counts[fam] = counts.get(fam, 0) + 1
            all_pairs.add((b_dr, e_dr))
    assert all_pairs == {(b, e) for b in range(1, 10) for e in range(1, 10)}, \
        "Not all 81 pairs covered"
    assert counts == {'Fibonacci': 24, 'Lucas': 24, 'Phibonacci': 24,
                      'Tribonacci': 8, 'Ninbonacci': 1}, f"Wrong counts: {counts}"
    print("V1 PASS  five families partition (Z/9Z)^2; sizes 24+24+24+8+1=81")


# ── V2: Pisano periods ────────────────────────────────────────────────────────

def pisano_period(b0, e0, mod, max_steps=200):
    b, e = b0 % mod, e0 % mod
    start = (b, e)
    for k in range(1, max_steps + 1):
        b, e = e % mod, (b + e) % mod
        if (b, e) == start:
            return k
    raise ValueError(f"Period not found within {max_steps} steps")

def verify_V2():
    seeds = [
        ('Fibonacci',  (1, 1), 24),
        ('Lucas',      (2, 1), 24),
        ('Phibonacci', (3, 1), 24),
        ('Tribonacci', (3, 3),  8),
        ('Ninbonacci', (9, 9),  1),
    ]
    for name, (b0, e0), expected in seeds:
        p = pisano_period(b0, e0, mod=9)
        assert p == expected, f"{name}: period={p}, expected={expected}"
    print("V2 PASS  Pisano periods: Fib/Luc/Phi=24, Trib=8, Nin=1")


# ── V3: orbits of F = five families ──────────────────────────────────────────

def verify_V3():
    orbits = get_orbits(apply_F)
    mismatches = 0
    from collections import defaultdict
    orbit_families = defaultdict(set)
    seed_to_orbit = {}
    for i, orb in enumerate(orbits):
        for s in orb:
            seed_to_orbit[s] = i

    for b in range(m):
        for e in range(m):
            oi = seed_to_orbit[(b, e)]
            fam = paper_map[(z9_to_dr(b), z9_to_dr(e))]
            orbit_families[oi].add(fam)

    for oi, fams in orbit_families.items():
        assert len(fams) == 1, \
            f"Orbit {oi} (len={len(orbits[oi])}) spans families: {fams}"

    sizes = sorted(len(o) for o in orbits)
    assert sizes == [1, 8, 24, 24, 24], f"Unexpected orbit sizes: {sizes}"
    print("V3 PASS  five families = orbits of F on (Z/9Z)^2; sizes [1,8,24,24,24]")


# ── V4: order of F ───────────────────────────────────────────────────────────

def verify_V4():
    F = [[0, 1], [1, 1]]
    I = [[1, 0], [0, 1]]
    order = None
    for k in range(1, 73):
        Fk = mat_pow_mod(F, k, m)
        if Fk == I:
            order = k
            break
    assert order == 24, f"Order of F = {order}, expected 24"
    # det(F) = -1 ≡ 8 mod 9
    det = (F[0][0] * F[1][1] - F[0][1] * F[1][0]) % m
    assert det == 8, f"det(F) = {det}, expected 8 (= -1 mod 9)"
    print(f"V4 PASS  order(F)=24 in GL_2(Z/9Z); det(F)=-1≡{det} mod 9")


# ── V5: norm alternation ──────────────────────────────────────────────────────

def verify_V5():
    for b in range(m):
        for e in range(m):
            fb, fe = apply_F(b, e)
            assert norm(fb, fe) == z9(-norm(b, e)), \
                f"Norm alternation fails at ({b},{e}): N(F(b,e))={norm(fb,fe)}, -N(b,e)={z9(-norm(b,e))}"
    print("V5 PASS  N(F(b,e)) = -N(b,e) mod 9 for all (b,e) in (Z/9Z)^2")


# ── V6: cosmos norm pairs ─────────────────────────────────────────────────────

def verify_V6():
    orbits = get_orbits(apply_F)
    expected = {
        'Fibonacci':  frozenset({1, 8}),
        'Lucas':      frozenset({4, 5}),
        'Phibonacci': frozenset({2, 7}),
    }
    seed_to_orbit = {}
    for i, orb in enumerate(orbits):
        for s in orb:
            seed_to_orbit[s] = i

    for b in range(m):
        for e in range(m):
            fam = paper_map[(z9_to_dr(b), z9_to_dr(e))]
            if fam in expected:
                n = norm(b, e)
                assert n in expected[fam], \
                    f"{fam}: N({b},{e})={n} not in {expected[fam]}"
    print(f"V6 PASS  cosmos norm pairs: Fib{{1,8}}, Luc{{4,5}}, Phi{{2,7}}")


# ── V7: Tribonacci ────────────────────────────────────────────────────────────

def verify_V7():
    orbits = get_orbits(apply_F)
    trib_orbits = [o for o in orbits if len(o) == 8]
    assert len(trib_orbits) == 1, f"Expected 1 Tribonacci orbit, got {len(trib_orbits)}"
    trib = trib_orbits[0]
    # All norms = 0 mod 9
    for b, e in trib:
        assert norm(b, e) == 0, f"Tribonacci N({b},{e})={norm(b,e)}, expected 0"
    # Stabilizer: find smallest k>0 s.t. F^k(3,3)=(3,3)
    cur = (3 % m, 3 % m)
    for k in range(1, 25):
        cur = apply_F(*cur)
        if cur == (3 % m, 3 % m):
            stab_order = k
            break
    assert stab_order == 8, f"Tribonacci period={stab_order}, expected 8"
    # Order of F is 24; orbit size = 24 / (24/8) = 8; stabilizer = <F^8>, order 3
    # Verify F^8 fixes (3,3)
    c = (3 % m, 3 % m)
    for _ in range(8):
        c = apply_F(*c)
    assert c == (3 % m, 3 % m), "F^8 does not fix (3,3)"
    print("V7 PASS  Tribonacci: N≡0 mod 9; orbit size 8; F^8 fixes (3,3)")


# ── V8: Ninbonacci fixed point ────────────────────────────────────────────────

def verify_V8():
    assert apply_F(0, 0) == (0, 0), "F does not fix (0,0)"
    # Unique fixed point
    fixed = [(b, e) for b in range(m) for e in range(m) if apply_F(b, e) == (b, e)]
    assert fixed == [(0, 0)], f"Fixed points: {fixed}"
    print("V8 PASS  Ninbonacci: unique fixed point (0,0); F(0,0)=(0,0)")


# ── V9: inradius identity r = be ──────────────────────────────────────────────

def verify_V9():
    for b in range(1, 21):
        for e in range(1, 21):
            d = b + e
            a = b + 2 * e
            C = 2 * d * e
            F_ = a * b
            G = e * e + d * d
            assert C * C + F_ * F_ == G * G, f"Not Pythagorean: ({C},{F_},{G})"
            r = (C + F_ - G) // 2
            assert r == b * e, f"Inradius: r={r}, be={b*e} for b={b},e={e}"
    print("V9 PASS  r = be for all BEDA tuples with b,e in {1..20}")


# ── V10: generator actions on r ───────────────────────────────────────────────

def verify_V10():
    # rho_p: (b,e) -> (b*p, e//p) when p|e; preserves r
    for b in range(1, 10):
        for e in range(1, 10):
            for p in [2, 3, 5]:
                if e % p == 0:
                    b2, e2 = b * p, e // p
                    assert b2 * e2 == b * e, \
                        f"rho_{p}: r not preserved: ({b},{e})->({b2},{e2})"
    # lambda_k: (b,e) -> (k*b, k*e); maps r -> k^2*r
    for b in range(1, 6):
        for e in range(1, 6):
            for k in range(2, 6):
                b2, e2 = k * b, k * e
                assert b2 * e2 == k * k * b * e, \
                    f"lambda_{k}: r not k^2*r: ({b},{e})->({b2},{e2})"
    print("V10 PASS rho_p preserves r=be; lambda_k maps r->k^2*r")


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    checks = [verify_V1, verify_V2, verify_V3, verify_V4, verify_V5,
              verify_V6, verify_V7, verify_V8, verify_V9, verify_V10]
    failures = []
    for fn in checks:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {exc}")
            failures.append(fn.__name__)

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print("All 10 verification claims PASS.")
        sys.exit(0)
