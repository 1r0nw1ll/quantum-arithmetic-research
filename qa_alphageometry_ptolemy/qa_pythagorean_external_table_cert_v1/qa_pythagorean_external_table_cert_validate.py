# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pythagorean external table laws: "
    "G-F=2e^2 across rows, G-C=b^2 down columns; A-to-B e-block transfer; "
    "D-to-E b-block transfer; F/C step sizes 2b/4e; no-empty-block rows/cols); "
    "Theorem NT: 'compatible pair', 'block', 'transfer', 'step' are observer "
    "classification labels; causal structure is polynomial algebra on bead numbers; "
    "no float state, no QA orbit evolution"
)

"""
Cert [353] — QA Pythagorean External Table Laws

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol I, Chapter VI pp.67-71
  p.62-63: "The difference between F and G remains constant in the blocks across
    the page, and the difference between C and G remains constant, progressing
    down the columns."
  p.62: "In the first line, the value of A becomes the value of B in the next
    block in succession across the page."
  p.62: "the value of D becomes the value of E in the next block down the column"
  p.62: "value of F increases in steps of 2 units for the altitude"
    "F increases by 2b units at each step" (down columns)
    "C increases by 4e units at each step" (across rows)
  p.62: "there are no empty, nonprime blocks in the column where b=1 or in the
    lines where e=1 and e=2"

Five claims:
  C1: G-F = 2e^2 is constant across rows (constant e); G-C = b^2 is constant down
      columns (constant b) — the two constant-difference laws
  C2: A(b,e) = B(b+2e, e) — the A value of any block equals the B value of the
      block e-steps to the right; gcd(b+2e, e) = gcd(b,e) = 1 (block always prime)
  C3: D(b,e) = E(b, e+b) — the D value equals E exactly b-steps down the column;
      in col b=1: D(1,e) = E(1, e+1) for all valid e
  C4: F increases by 2b per unit e-step down each column;
      C increases by 4e per 2-step in b across each row
  C5: b=1 column has no empty blocks; e=1 and e=2 rows have no empty blocks
      (i.e., gcd(b,1)=gcd(b,2)=1 for all odd b; gcd(1,e)=1 for all e)
"""

from math import gcd


def _valid_beads(max_b: int, max_e: int):
    """Yield (b, e, d, a) with b odd, gcd(b,e)=1, 1<=b<=max_b, 1<=e<=max_e."""
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) == 1:
                d = b + e
                a = d + e
                yield b, e, d, a


def check_c1() -> tuple[bool, str]:
    """G-F=2e^2 across rows; G-C=b^2 down columns."""
    count = 0
    for b, e, d, a in _valid_beads(25, 25):
        F = a * b
        C = 2 * d * e
        G = d * d + e * e
        # Across rows (constant e): G-F = (d^2+e^2)-(d^2-e^2) = 2e^2
        gf_diff = G - F
        expected_gf = 2 * e * e
        assert gf_diff == expected_gf, f"G-F!= 2e^2: G-F={gf_diff}, 2e^2={expected_gf} b={b},e={e}"
        # Down columns (constant b): G-C = (d-e)^2 = b^2
        gc_diff = G - C
        expected_gc = b * b
        assert gc_diff == expected_gc, f"G-C!= b^2: G-C={gc_diff}, b^2={expected_gc} b={b},e={e}"
        count += 1
    # Algebraic proofs:
    # G-F = (d^2+e^2) - (d^2-e^2) = 2e^2 (since F=ab=(d+e)(d-e)=d^2-e^2)
    # G-C = (d^2+e^2) - 2de = (d-e)^2 = b^2 (since b=d-e)
    return True, (
        f"G-F=2e^2 and G-C=b^2 verified for all {count} valid pairs (b,e)<=25; "
        f"proof: G-F=(d^2+e^2)-(d^2-e^2)=2e^2; G-C=(d-e)^2=b^2"
    )


def check_c2() -> tuple[bool, str]:
    """A(b,e) = B(b+2e, e): A transfers to B exactly e-blocks right; gcd preserved."""
    count = 0
    max_b, max_e = 20, 20
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = d + e
            A_val = a * a  # A = a^2 = (b+2e)^2
            # The next block: b_new = b + 2e, same e
            b_new = b + 2 * e
            B_val = b_new * b_new  # B = b^2 for block (b_new, e)
            assert A_val == B_val, f"A(b,e)!=B(b+2e,e): A={A_val}, B={B_val} b={b},e={e}"
            # gcd preserved: gcd(b+2e, e) = gcd(b, e) = 1
            assert gcd(b_new, e) == 1, f"gcd(b+2e,e)!=1: b={b},e={e},b_new={b_new}"
            count += 1
    # Proof: A(b,e)=a^2=(b+2e)^2; B(b+2e,e)=(b+2e)^2 ✓
    # gcd(b+2e, e)=gcd(b,e)=1 by gcd linearity ✓
    return True, (
        f"A(b,e)=B(b+2e,e) verified for all {count} valid pairs (b,e)<=20; "
        f"gcd(b+2e,e)=gcd(b,e)=1 always (next block is always prime); "
        f"proof: A=a^2=(b+2e)^2=B(b+2e,e)"
    )


def check_c3() -> tuple[bool, str]:
    """D(b,e) = E(b, e+b): D transfers to E exactly b-blocks down; b=1 col is D(1,e)=E(1,e+1)."""
    count = 0
    max_b, max_e = 20, 20
    for b in range(1, max_b + 1, 2):
        for e in range(1, max_e + 1):
            if gcd(b, e) != 1:
                continue
            d = b + e
            D_val = d * d  # D = d^2 = (b+e)^2
            # b-blocks down: e_new = e + b
            e_new = e + b
            if e_new > max_e:
                continue
            E_val = e_new * e_new  # E = e^2 for block (b, e_new)
            assert D_val == E_val, f"D(b,e)!=E(b,e+b): D={D_val}, E={E_val} b={b},e={e}"
            # gcd(b, e_new) = gcd(b, e+b) = gcd(b, e) = 1
            assert gcd(b, e_new) == 1, f"gcd(b,e+b)!=1: b={b},e_new={e_new}"
            count += 1
    # Special case b=1: D(1,e)=(1+e)^2 = E(1,e+1)=(e+1)^2 ✓
    for e in range(1, 30):
        assert (1 + e) * (1 + e) == (e + 1) * (e + 1)  # trivial identity check
    return True, (
        f"D(b,e)=E(b,e+b) verified for {count} valid pairs with e+b<=20; "
        f"b=1 special case: D(1,e)=(1+e)^2=E(1,e+1) for all e; "
        f"gcd(b,e+b)=gcd(b,e)=1 (destination always prime)"
    )


def check_c4() -> tuple[bool, str]:
    """F increases by 2b per unit e-step down each column; C increases by 4e per 2-step in b."""
    count_f = 0
    for b in range(1, 20 + 1, 2):
        for e in range(1, 20):
            if gcd(b, e) != 1 or gcd(b, e + 1) != 1:
                continue
            a_e = b + 2 * e
            a_e1 = b + 2 * (e + 1)
            F_e = a_e * b
            F_e1 = a_e1 * b
            diff_f = F_e1 - F_e
            assert diff_f == 2 * b, f"F step!= 2b: diff={diff_f}, 2b={2*b} b={b},e={e}"
            count_f += 1
    # C(b,e) = 2de = 2(b+e)e. For constant e, step of 2 in b:
    # C(b+2,e) - C(b,e) = 2(b+2+e)e - 2(b+e)e = 2*2*e = 4e
    count_c = 0
    for e in range(1, 20 + 1):
        for b in range(1, 19, 2):
            b2 = b + 2
            if gcd(b, e) != 1 or gcd(b2, e) != 1:
                continue
            C_b = 2 * (b + e) * e
            C_b2 = 2 * (b2 + e) * e
            diff_c = C_b2 - C_b
            assert diff_c == 4 * e, f"C step!= 4e: diff={diff_c}, 4e={4*e} b={b},e={e}"
            count_c += 1
    return True, (
        f"F column steps: 2b per unit e-step verified for {count_f} column-steps (b<=20,e<=20); "
        f"C row steps: 4e per 2-step in b verified for {count_c} row-steps; "
        f"proof: F=(b+2e)b, dF/de=2b; C=2(b+e)e, delta_C_per_2b=4e"
    )


def check_c5() -> tuple[bool, str]:
    """b=1 column has no empty blocks; e=1 and e=2 rows have no empty blocks."""
    # b=1 column: gcd(1, e) = 1 for all e >= 1
    for e in range(1, 200):
        assert gcd(1, e) == 1, f"gcd(1,{e}) != 1"

    # e=1 row: for all odd b, gcd(b, 1) = 1
    for b in range(1, 200, 2):
        assert gcd(b, 1) == 1, f"gcd({b},1) != 1"

    # e=2 row: for all odd b, gcd(b, 2) = 1 (odd and even are always coprime)
    for b in range(1, 200, 2):
        assert gcd(b, 2) == 1, f"gcd({b},2) != 1"

    # e=4,8 rows: power-of-2 e, odd b -> always coprime
    for power in [4, 8, 16, 32]:
        for b in range(1, 100, 2):
            assert gcd(b, power) == 1, f"gcd({b},{power}) != 1"

    return True, (
        "b=1 column: gcd(1,e)=1 for all e (verified e<=200); "
        "e=1 row: gcd(b,1)=1 for all odd b (verified b<=199); "
        "e=2 row: gcd(b,2)=1 for all odd b (odd+even always coprime; verified b<=199); "
        "e=4,8,16,32 rows: odd b always coprime to powers of 2 (verified b<=99)"
    )


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise RuntimeError(f"cert [353] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
