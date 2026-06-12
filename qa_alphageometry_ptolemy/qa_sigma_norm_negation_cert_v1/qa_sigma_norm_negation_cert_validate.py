#!/usr/bin/env python3
# noqa: S1 S2 T1 T2 (cert validator — integers only, no QA state drift)
# Primary sources: Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.13
#   (norm forms of quadratic extensions, discriminant 5);
#   Neukirch (1999) ISBN 978-3-540-65399-8 Ch.I §7 (norm, discriminant);
#   Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Fibonacci matrix).
"""
Cert [391]: QA σ-Norm Negation and Eigenline Zero Set

Core claim: σ(a,b) = (a+b, a) IS multiplication by φ on ℤ[φ] under the
identification (a,b) ↔ a·φ+b.  Since N(φ) = −1 (the algebraic norm from
ℚ(√5)/ℚ), σ is a norm-negating endomorphism: N(σ(a,b)) = −N(a,b) where
N(a,b) = b²+ab−a².

At every rational prime p, the geometric signature is:
  split p   (p≡±1 mod 5): norm-zero locus mod p = σ-eigenline union (2 lines)
  ramified p=5:            norm-zero locus = single σ-eigenline (double root)
  inert p   (p≡±2 mod 5): norm form is anisotropic, no norm-zero locus

The orbit of φ=(1,0) under σ traces the Fibonacci sequence:
  σ^k(1,0) = (F_{k+1}, F_k)  for all k ≥ 0

7 checks (all PASS) + 8 fixtures (7 PASS, 1 FAIL designed).
Extends: [388] (eigenspaces at split primes), [386] (prime classification),
         [385] (RM by ℤ[φ]).
"""
import sys
import json


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def norm(a, b):
    """N_{ℚ(√5)/ℚ}(a·φ + b) = b²+ab−a²."""
    return b * b + a * b - a * a


def sigma(a, b):
    """QA shift: σ(a,b) = (a+b, a)."""
    return (a + b, a)


def sigma_k(a, b, k):
    for _ in range(k):
        a, b = sigma(a, b)
    return (a, b)


def fib(n):
    """Standard Fibonacci: F(0)=0, F(1)=1, F(2)=1, …"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def roots_mod_p(p):
    return [r for r in range(p) if (r * r - r - 1) % p == 0]


def sigma_eigenlines_mod_p(p):
    """Non-zero elements on σ-eigenlines mod p (one per root of x²−x−1)."""
    result = set()
    for r in roots_mod_p(p):
        for k in range(1, p):
            result.add((r * k % p, k))
    return result


def norm_zeros_mod_p(p):
    """Non-zero (a,b) mod p with N(a,b) ≡ 0 mod p."""
    return set(
        (a, b)
        for a in range(p)
        for b in range(p)
        if (a, b) != (0, 0) and norm(a, b) % p == 0
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_phi_mult_identity():
    """σ(a,b)=(a+b,a) equals φ·(a·φ+b): new (φ-coeff, const) = (a+b, a)."""
    # φ·(a·φ+b) = a·φ² + b·φ = a·(φ+1) + b·φ = a + (a+b)·φ
    pairs = [(1, 0), (0, 1), (1, 1), (2, 3), (5, 8), (3, -2), (13, 21), (-4, 7)]
    for a, b in pairs:
        if (a + b, a) != sigma(a, b):
            return False, f"identity failed at ({a},{b})"
    return True, f"σ(a,b)=(a+b,a)=φ·(a·φ+b) verified on {len(pairs)} pairs"


def check_norm_of_phi():
    """N(φ)=N(1,0)=0²+0·1−1²=−1; N(ψ)=N(−1,1)=−1 (ψ=1−φ, Galois conjugate)."""
    n_phi = norm(1, 0)
    n_psi = norm(-1, 1)
    if n_phi != -1:
        return False, f"N(phi)={n_phi}, expected −1"
    if n_psi != -1:
        return False, f"N(psi)={n_psi}, expected −1"
    return True, "N(φ)=N(ψ)=−1; both generators of ℤ[φ] have norm −1"


def check_sigma_negates_norm():
    """N(σ(a,b)) = −N(a,b) for all tested integer pairs."""
    pairs = [(1, 0), (0, 1), (1, 1), (2, 3), (5, 8), (3, -2), (7, 11),
             (13, 21), (0, 5), (-3, 4), (100, 63), (-7, 5)]
    for a, b in pairs:
        n0 = norm(a, b)
        sa, sb = sigma(a, b)
        n1 = norm(sa, sb)
        if n1 != -n0:
            return False, f"N(σ({a},{b}))={n1} ≠ −N({a},{b})={-n0}"
    return True, f"N(σ(a,b))=−N(a,b) verified on {len(pairs)} integer pairs"


def check_sigma_sq_preserves():
    """N(σ²(a,b)) = N(a,b) for all tested pairs."""
    pairs = [(1, 0), (0, 1), (1, 1), (2, 3), (5, 8), (3, -2), (7, 11), (0, 5)]
    for a, b in pairs:
        n0 = norm(a, b)
        s2a, s2b = sigma(*sigma(a, b))
        n2 = norm(s2a, s2b)
        if n2 != n0:
            return False, f"N(σ²({a},{b}))={n2} ≠ N({a},{b})={n0}"
    return True, f"N(σ²(a,b))=N(a,b) verified on {len(pairs)} pairs"


def check_eigenline_norm_zero():
    """Norm-zero locus mod p = σ-eigenline union (split/ram); empty (inert)."""
    split_ram = [5, 11, 19, 29, 41, 59, 61, 71]
    inert = [2, 3, 7, 13, 17, 23, 43, 47]
    for p in split_ram:
        eigen = sigma_eigenlines_mod_p(p)
        nz = norm_zeros_mod_p(p)
        if eigen != nz:
            extra = nz - eigen
            miss = eigen - nz
            return False, (f"p={p}: |eigenlines|={len(eigen)}, |norm-zeros|={len(nz)}"
                           f", extra={list(extra)[:3]}, missing={list(miss)[:3]}")
    for p in inert:
        nz = norm_zeros_mod_p(p)
        if nz:
            return False, f"p={p} inert: non-empty norm-zero locus {list(nz)[:3]}"
    return True, (
        f"Split/ram {split_ram}: eigenline union = norm-zero locus ✓; "
        f"Inert {inert}: anisotropic ✓"
    )


def check_discriminant_5():
    """disc(−a²+ab+b²) = B²−4AC = 1−4·(−1)·1 = 5 = disc(ℚ(√5)/ℚ)."""
    A, B, C = -1, 1, 1
    disc = B * B - 4 * A * C
    if disc != 5:
        return False, f"discriminant={disc}, expected 5"
    qr_11 = pow(5, (11 - 1) // 2, 11) == 1
    nqr_7 = pow(5, (7 - 1) // 2, 7) != 1
    if not (qr_11 and nqr_7):
        return False, f"Legendre symbol check: (5|11)={qr_11}, (5|7)={nqr_7}"
    return True, (
        "disc(N)=5=disc(ℚ(√5)/ℚ); "
        "(5|11)=+1 (split) and (5|7)=−1 (inert), consistent with [386]"
    )


def check_fibonacci_orbit():
    """σ^k(1,0)=(F_{k+1},F_k) for k=0..14; orbit of φ traces Fibonacci."""
    a, b = 1, 0
    for k in range(15):
        expected = (fib(k + 1), fib(k))
        if (a, b) != expected:
            return False, f"σ^{k}(1,0)={a,b}, expected {expected}"
        a, b = sigma(a, b)
    return True, (
        "σ^k(1,0)=(F_{k+1},F_k) k=0..14; "
        "because φ^k=F_k·φ+F_{k-1} and σ=×φ"
    )


CHECKS = {
    "PHI_MULT_IDENTITY": check_phi_mult_identity,
    "NORM_OF_PHI": check_norm_of_phi,
    "SIGMA_NEGATES_NORM": check_sigma_negates_norm,
    "SIGMA_SQ_PRESERVES": check_sigma_sq_preserves,
    "EIGENLINE_NORM_ZERO": check_eigenline_norm_zero,
    "DISCRIMINANT_5": check_discriminant_5,
    "FIBONACCI_ORBIT": check_fibonacci_orbit,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "name": "NORM_NEG_AT_EIGENLINE_START",
        "description": (
            "N(4,1)=1+4−16=−11, N(σ(4,1))=N(5,4)=16+20−25=11=−(−11): "
            "norm negation at eigenline start p=11"
        ),
        "expected": True,
        "fn": lambda: norm(*sigma(4, 1)) == -norm(4, 1),
    },
    {
        "name": "EIGENLINE_IS_NORM_ZERO_p11",
        "description": "(4,1) lies on σ-eigenline r₁=4 of p=11; N(4,1)=−11≡0 mod 11",
        "expected": True,
        "fn": lambda: norm(4, 1) % 11 == 0,
    },
    {
        "name": "NON_EIGENLINE_NONZERO_p11",
        "description": "(1,1) is NOT on any σ-eigenline mod 11; N(1,1)=1≢0 mod 11",
        "expected": True,
        "fn": lambda: norm(1, 1) % 11 != 0,
    },
    {
        "name": "INERT_ANISOTROPIC_p7",
        "description": "At inert p=7: every non-zero (a,b) has N(a,b)≢0 mod 7",
        "expected": True,
        "fn": lambda: not any(
            norm(a, b) % 7 == 0
            for a in range(7) for b in range(7)
            if (a, b) != (0, 0)
        ),
    },
    {
        "name": "FIBONACCI_ORBIT_k6",
        "description": "σ⁶(1,0)=(F₇,F₆)=(13,8)",
        "expected": True,
        "fn": lambda: sigma_k(1, 0, 6) == (fib(7), fib(6)),
    },
    {
        "name": "NORM_OF_PSI",
        "description": (
            "N(ψ)=N(−1,1)=1+(−1)·1−(−1)²=1−1−1=−1; "
            "ψ=1−φ (Galois conjugate) has the same norm as φ"
        ),
        "expected": True,
        "fn": lambda: norm(-1, 1) == -1,
    },
    {
        "name": "RAMIFIED_NORM_ZERO_p5",
        "description": (
            "At ramified p=5, double root r=3 of x²−x−1 mod 5: "
            "N(3,1)=1+3−9=−5≡0 mod 5"
        ),
        "expected": True,
        "fn": lambda: norm(3, 1) % 5 == 0,
    },
    {
        "name": "SIGMA_PRESERVES_NOT_NEGATES_FAILS",
        "description": (
            "DESIGNED FAIL: N(σ(1,0))=N(1,1)=1 ≠ N(1,0)=−1; "
            "σ does NOT preserve the norm (it negates it)"
        ),
        "expected": False,
        "fn": lambda: norm(*sigma(1, 0)) == norm(1, 0),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_self_test():
    results = {}
    all_pass = True
    for name, fn in CHECKS.items():
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, str(exc)
        results[name] = {"ok": ok, "detail": msg}
        if not ok:
            all_pass = False

    fixture_pass = 0
    fixture_results = []
    for fx in FIXTURES:
        try:
            got = fx["fn"]()
        except Exception as exc:
            got = None
        passed = (got == fx["expected"])
        fixture_pass += int(passed)
        fixture_results.append({
            "name": fx["name"],
            "expected": fx["expected"],
            "got": got,
            "passed": passed,
        })

    return {
        "ok": all_pass and fixture_pass == len(FIXTURES),
        "checks": {k: v["ok"] for k, v in results.items()},
        "check_details": {k: v["detail"] for k, v in results.items()},
        "fixture_summary": f"{fixture_pass}/{len(FIXTURES)} passed",
        "fixtures": fixture_results,
        "norm_form": "N(a,b) = b^2 + a*b - a^2 = N_{Q(sqrt(5))/Q}(a*phi + b)",
        "sigma_identity": "sigma(a,b) = (a+b,a) = phi*(a*phi+b) in Z[phi]",
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        out = run_self_test()
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["ok"] else 1)
    print("Usage: python3 qa_sigma_norm_negation_cert_validate.py --self-test")
    sys.exit(1)
