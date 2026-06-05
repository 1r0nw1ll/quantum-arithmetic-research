# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (BABTHE dual bead chain: nested "
    "Fibonacci quadruples joined at junction Q; unit fraction decomposition 2/T); "
    "Theorem NT: 'Babylonian fraction', 'unit fraction' are observer labels; "
    "causal structure is integer arithmetic: S=O*P, R=S-Q, T=R+S; no float, no mod"
)

"""
Cert [345] — QA BABTHE Dual Bead Chain Identity

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol II, Chapter XIV pp.74-85
  p.77: "R+S=T, or 41+56=97 for the sum. Then 56-41=15, for the difference
   giving the four bead numbers for this fraction being 15, 41, 56, 97."
  p.77: "the first set is assigned the identities N, O, P, Q and the second set
   is assigned the identities Q, R, S, T."
  p.77: "the third number, S, of the higher bead numbers is also the product of
   the two intermediate numbers of the lower bead numbers."
  p.78: "O+P was equal to S-R." [i.e., Q = S-R]
  Ch.XIV BABTHE2 program: 2/T = 1/S + 1/(OT) + 1/(PT), S=OP, R=S-Q, T=R+S

Five claims:
  C1: (N, O, P=N+O, Q=O+P) is a Fibonacci-type bead quadruple: N+O=P, O+P=Q
  C2: (Q, R=S-Q, S=O*P, T=R+S) is a second Fibonacci-type bead quadruple: Q+R=S, R+S=T
  C3: Q is the shared junction: Q=O+P (4th of first set) = S-R (Q+R=S, 1st of second set)
  C4: S = O*P (product of inner pair of first set = the sum-element of second set)
  C5: 2/T = 1/S + 1/(OT) + 1/(PT) — algebraically guaranteed by R=S-Q (i.e., R+Q=S)

Algebraic proof of C5:
  1/S + 1/(OT) + 1/(PT) = 1/(OP) + (O+P)/(OPT) = [T + Q]/(ST)
  = [R+S+Q]/(ST)  since T=R+S
  = [R+Q+S]/(ST)  = S/(ST)  since R+Q=S  [from R=S-Q]
  = 1/T            ... multiply by 2: 2/T ✓

Verification: 7 distinct coprime pairs (N odd ≥ 1, O ≥ 2, gcd(N,O)=1)
Including Iverson's canonical example: N=1, O=7 → 2/97=1/56+1/679+1/776
"""

from math import gcd
from fractions import Fraction


def _dual_chain(N: int, O: int) -> dict:
    """Compute both bead quadruples for coprime (N, O)."""
    P = N + O
    Q = O + P         # = N + 2*O
    S = O * P         # product of inner pair of first set
    R = S - Q         # = (O-1)*(P-1) - 1 = O*P - O - P
    T = R + S
    return {"N": N, "O": O, "P": P, "Q": Q, "S": S, "R": R, "T": T}


def check_c1() -> tuple[bool, str]:
    """(N, O, P=N+O, Q=O+P) is Fibonacci-type: N+O=P, O+P=Q."""
    test_pairs = [(1, 2), (1, 4), (1, 6), (1, 7), (3, 4), (3, 8), (5, 6)]
    for N, O in test_pairs:
        assert gcd(N, O) == 1, f"gcd({N},{O}) != 1"
        d = _dual_chain(N, O)
        assert d["N"] + d["O"] == d["P"], (
            f"N={N} O={O}: N+O={N+O} != P={d['P']}"
        )
        assert d["O"] + d["P"] == d["Q"], (
            f"N={N} O={O}: O+P={d['O']+d['P']} != Q={d['Q']}"
        )
    return True, (
        f"First quadruple (N,O,P,Q) Fibonacci-type N+O=P, O+P=Q "
        f"verified for {len(test_pairs)} coprime pairs"
    )


def check_c2() -> tuple[bool, str]:
    """(Q, R, S, T) is Fibonacci-type: Q+R=S, R+S=T."""
    test_pairs = [(1, 2), (1, 4), (1, 6), (1, 7), (3, 4), (3, 8), (5, 6)]
    for N, O in test_pairs:
        d = _dual_chain(N, O)
        Q, R, S, T = d["Q"], d["R"], d["S"], d["T"]
        assert Q + R == S, f"N={N} O={O}: Q+R={Q+R} != S={S}"
        assert R + S == T, f"N={N} O={O}: R+S={R+S} != T={T}"
        # All four must be positive
        assert Q > 0 and R > 0 and S > 0 and T > 0, (
            f"N={N} O={O}: non-positive element in (Q,R,S,T)=({Q},{R},{S},{T})"
        )
    return True, (
        "Second quadruple (Q,R,S,T) Fibonacci-type Q+R=S, R+S=T "
        "verified for 7 coprime pairs; all elements positive"
    )


def check_c3() -> tuple[bool, str]:
    """Q is the shared junction: Q = O+P (4th of first set) = S-R (1st of second)."""
    test_pairs = [(1, 2), (1, 4), (1, 6), (1, 7), (3, 4), (3, 8), (5, 6)]
    iverson_example = None
    for N, O in test_pairs:
        d = _dual_chain(N, O)
        # Q as 4th of first set: Q = O+P
        Q_from_first = d["O"] + d["P"]
        # Q as 1st of second set: implies Q+R=S → Q = S-R
        Q_from_second = d["S"] - d["R"]
        assert Q_from_first == Q_from_second == d["Q"], (
            f"N={N} O={O}: junction mismatch: O+P={Q_from_first}, S-R={Q_from_second}, Q={d['Q']}"
        )
        if (N, O) == (1, 7):
            iverson_example = d
    assert iverson_example is not None
    # Iverson's canonical: N=1,O=7 → (1,7,8,15) and (15,41,56,97)
    d = iverson_example
    assert (d["N"], d["O"], d["P"], d["Q"]) == (1, 7, 8, 15), (
        f"Iverson's first quadruple: expected (1,7,8,15), got {(d['N'],d['O'],d['P'],d['Q'])}"
    )
    assert (d["Q"], d["R"], d["S"], d["T"]) == (15, 41, 56, 97), (
        f"Iverson's second quadruple: expected (15,41,56,97), got {(d['Q'],d['R'],d['S'],d['T'])}"
    )
    return True, (
        "Junction Q is shared 4th/1st element: Q=O+P=S-R for all 7 pairs; "
        "Iverson's example (1,7,8,15)+(15,41,56,97) ✓"
    )


def check_c4() -> tuple[bool, str]:
    """S = O*P: product of inner pair of first set = sum-element of second set."""
    test_pairs = [(1, 2), (1, 4), (1, 6), (1, 7), (3, 4), (3, 8), (5, 6)]
    for N, O in test_pairs:
        d = _dual_chain(N, O)
        assert d["S"] == d["O"] * d["P"], (
            f"N={N} O={O}: S={d['S']} != O*P={d['O']*d['P']}"
        )
        # Also verify R = (O-1)*(P-1) - 1
        O, P, R = d["O"], d["P"], d["R"]
        assert R == (O - 1) * (P - 1) - 1, (
            f"N={d['N']} O={O}: R={R} != (O-1)(P-1)-1={(O-1)*(P-1)-1}"
        )
    return True, (
        "S=O*P (product relation) verified for all 7 pairs; "
        "equivalently R=(O-1)(P-1)-1 verified"
    )


def check_c5() -> tuple[bool, str]:
    """2/T = 1/S + 1/(OT) + 1/(PT): unit fraction decomposition."""
    test_pairs = [(1, 2), (1, 4), (1, 6), (1, 7), (3, 4), (3, 8), (5, 6)]
    for N, O in test_pairs:
        d = _dual_chain(N, O)
        S, T, P = d["S"], d["T"], d["P"]
        # Compute using Fraction for exact arithmetic
        lhs = Fraction(2, T)
        rhs = Fraction(1, S) + Fraction(1, O * T) + Fraction(1, P * T)
        assert lhs == rhs, (
            f"N={N} O={O}: 2/{T} = {lhs} but 1/{S}+1/{O*T}+1/{P*T} = {rhs}"
        )
    # Algebraic proof: R+Q=S → (R+S+Q)/(ST)=S/(ST)=1/T → *2 gives 2/T
    # Direct verification of Iverson's example:
    d = _dual_chain(1, 7)
    assert Fraction(2, 97) == Fraction(1, 56) + Fraction(1, 679) + Fraction(1, 776), (
        "Iverson's 2/97 = 1/56 + 1/679 + 1/776 failed"
    )
    assert Fraction(2, 7) == Fraction(1, 6) + Fraction(1, 14) + Fraction(1, 21), (
        "Iverson's 2/7 = 1/6 + 1/14 + 1/21 failed"
    )
    return True, (
        "2/T = 1/S + 1/(OT) + 1/(PT) verified for all 7 pairs by exact Fraction arithmetic; "
        "Iverson's 2/7=1/6+1/14+1/21 and 2/97=1/56+1/679+1/776 ✓"
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
        raise RuntimeError(f"cert [345] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
