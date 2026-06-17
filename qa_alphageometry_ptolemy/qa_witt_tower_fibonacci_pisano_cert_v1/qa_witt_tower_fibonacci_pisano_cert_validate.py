"""QA Witt Tower Fibonacci-Pisano Synthesis cert [441] validator."""
# <!-- PRIMARY-SOURCE-EXEMPT: reason=Fibonacci-Pisano synthesis applying Witt tower chain [437]-[440] to Wall (1960) Pisano period theorem; Wall (1960) doi:10.1080/00029890.1960.11989541; Ireland & Rosen (1990) ISBN 978-0-387-97329-6 ch.5,7; Serre (1979) doi:10.1007/978-1-4757-5673-9 ch.1-3 -->
import json, sys

# NO orbit enumeration anywhere in this file.
# All checks use only the Pisano recurrence (two integers, O(1) memory).
# Orbit structure for p=5 is certified by [440]; this cert synthesises the law.


def _pisano(n):
    """Period of Fibonacci sequence mod n. O(pi(n)) time, O(1) memory."""
    if n == 1:
        return 1
    a, b = 0, 1
    for t in range(1, 6 * n + 2):
        a, b = b, (a + b) % n
        if a == 0 and b == 1:
            return t
    raise RuntimeError(f"pi({n}) not found")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_c1_p5_pisano():
    """C1: pi(5^k) = 4*5^k for k=1..5. Pisano recurrence only."""
    failures = []
    for k in range(1, 6):
        pk = 5 ** k
        got = _pisano(pk)
        want = 4 * pk
        if got != want:
            failures.append(f"k={k}: pi(5^{k})={got} != {want}")
    return not failures, failures or ["PASS k=1..5: pi(5^k)=4*5^k"]


def check_c2_p5_dm1_formula():
    """C2: [440] formula for p=5 r=1 element totals equal 5^(2k). Pure arithmetic.

    [440]: count(1)=1, count(4)=1, count(4*5^L)=5^(k-1) for L=1..k (k<=r=1 case)
    or frozen/birth pattern for k>r. Verify total elements = 5^(2k).
    """
    failures = []
    for k in range(1, 6):
        total = 5 ** (2 * k)
        # [440] formula, p=5, r=1
        acc = 1                                         # count(1)*1
        acc += (5 - 1) // 4 * 4                        # count(4)*4 = 1*4
        birth = (5 - 1) // 4 * 5 ** (k - 1)            # = 5^(k-1)
        if k == 1:                                      # k <= r=1
            acc += birth * (4 * 5)                      # count(20)*20
        else:                                           # k > r=1
            for L in range(1, k):                       # frozen L=1..k-1
                acc += (25 - 1) // 4 * 5 ** (L - 1) * (4 * 5 ** L)
            acc += birth * (4 * 5 ** k)                 # birth layer L=k
        if acc != total:
            failures.append(f"k={k}: element sum={acc} != {total}")
    return not failures, failures or ["PASS k=1..5: [440] element sums = 5^(2k)"]


def check_c3_p2_pisano():
    """C3: pi(2^k) = 3*2^(k-1) for k=1..8. Pisano only."""
    failures = []
    for k in range(1, 9):
        pk = 2 ** k
        got = _pisano(pk)
        want = 3 * pk // 2
        if got != want:
            failures.append(f"k={k}: pi(2^{k})={got} != {want}")
    return not failures, failures or ["PASS p=2 k=1..8: pi(2^k)=3*2^(k-1)"]


def check_c4_unramified_wall():
    """C4: pi(p^k) = pi(p)*p^(k-1) for unramified p in {3,7,11,13,17,19}, k=2..4."""
    failures = []
    for p in [3, 7, 11, 13, 17, 19]:
        pi_p = _pisano(p)
        for k in range(2, 5):
            got = _pisano(p ** k)
            want = pi_p * (p ** (k - 1))
            if got != want:
                failures.append(f"p={p} k={k}: pi={got} wall={want}")
    return not failures, failures or ["PASS {3,7,11,13,17,19} k=2..4: Wall lift law"]


def check_c5_three_regimes():
    """C5: Three-regime cross-check at k=3.

    p=5 (ramified det=-1, [440]): pi(125)=500=4*125
    p=2 (exceptional):            pi(8)=12=3*4
    p=3 (unramified inert):       pi(27)=pi(3)*9
    p=11 (unramified split):      pi(1331)=pi(11)*121
    """
    failures = []
    cases = [
        (5, 3, 4 * 125),
        (2, 3, 12),
        (3, 3, _pisano(3) * 9),
        (11, 3, _pisano(11) * 121),
    ]
    for p, k, want in cases:
        got = _pisano(p ** k)
        if got != want:
            failures.append(f"p={p} k={k}: pi={got} expected={want}")
    return not failures, failures or ["PASS 4 regime cross-checks at k=3"]


def check_c6_split_vs_inert():
    """C6: pi(p) | p-1 (split) or | 2(p+1) (inert) for 12 unramified primes."""
    def legendre(a, p):
        v = pow(a, (p - 1) // 2, p)
        return 1 if v == 1 else (-1 if v != 0 else 0)

    failures = []
    for p in [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]:
        pi_p = _pisano(p)
        leg = legendre(5, p)
        if leg == 1 and (p - 1) % pi_p != 0:
            failures.append(f"p={p} split: pi={pi_p} not|{p-1}")
        elif leg == -1 and (2 * (p + 1)) % pi_p != 0:
            failures.append(f"p={p} inert: pi={pi_p} not|{2*(p+1)}")
    return not failures, failures or ["PASS 12 primes: split/inert divisibility"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = [
    ("FIX1_P5_K1_20",   "pi(5)=20=4*5",              lambda: _pisano(5) == 20),
    ("FIX2_P5_K2_100",  "pi(25)=100=4*25",           lambda: _pisano(25) == 100),
    ("FIX3_P5_K3_500",  "pi(125)=500=4*125",         lambda: _pisano(125) == 500),
    ("FIX4_P2_K3_12",   "pi(8)=12=3*4",              lambda: _pisano(8) == 12),
    ("FIX5_P3_WALL",    "pi(9)=pi(3)*3",             lambda: _pisano(9) == _pisano(3) * 3),
    ("FIX6_P7_WALL",    "pi(49)=pi(7)*7",            lambda: _pisano(49) == _pisano(7) * 7),
    ("FIX7_P5_UNIQUE",  "5 only prime | t^2+4, t=1", lambda: all(5 % p != 0 for p in [2,3,7,11,13,17,19,23])),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def validate(base_dir=None):
    checks = [
        ("C1_P5_PISANO",      check_c1_p5_pisano),
        ("C2_P5_DM1_FORMULA", check_c2_p5_dm1_formula),
        ("C3_P2_PISANO",      check_c3_p2_pisano),
        ("C4_WALL_LIFT",      check_c4_unramified_wall),
        ("C5_THREE_REGIMES",  check_c5_three_regimes),
        ("C6_SPLIT_INERT",    check_c6_split_vs_inert),
    ]
    results = {}
    all_ok = True
    for name, fn in checks:
        ok, detail = fn()
        results[name] = {"ok": ok, "detail": detail}
        if not ok:
            all_ok = False

    fix_pass, fix_results = 0, []
    for name, desc, fn in FIXTURES:
        try:
            ok = fn()
        except Exception as e:
            ok, desc = False, str(e)
        fix_pass += ok
        fix_results.append({"name": name, "ok": ok, "desc": desc})
    all_ok = all_ok and fix_pass == len(FIXTURES)

    out = {
        "ok": all_ok,
        "checks": results,
        "fixture_summary": f"{fix_pass}/{len(FIXTURES)} passed",
        "fixtures": fix_results,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    sys.exit(0 if validate()["ok"] else 1)
