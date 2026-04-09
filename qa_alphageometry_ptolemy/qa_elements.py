#!/usr/bin/env python3
"""
qa_elements.py — QA Element Oracle

THE authoritative source of QA element computation for cert validators.
Zero external dependencies (stdlib only). Drop into any validator directory.

HARD RULES ENFORCED BY DESIGN:
  - Elements use RAW d = b+e, a = b+2e. No modulus parameter exists on
    element functions. The function signature makes mod-reduction impossible.
  - qa_step() is the ONLY function that uses modular arithmetic, and it is
    clearly separated from element computation.
  - All 9 structural invariants are asserted at compute time (fail-fast).
  - S1 compliant: b*b product form everywhere, no power operator.

AXIOM COMPLIANCE:
  A1 (No-Zero): b, e >= 1 enforced. States in {1,...,N}.
  A2 (Derived): d = b+e, a = b+2e always derived, never independent.
  S1 (No **2): All multiplications use product form.
  S2 (No float): All values are int. No np.zeros, no np.random.
  T1 (Path time): No continuous time variables.
  T2 (Firewall): No float*modulus→int casts. Observer projections only.

Certified against: [133] QA_SIXTEEN_IDENTITIES_CERT.v1
Cross-verified against: qa_arithmetic/qa_arithmetic/identities.py

Usage in validators:
    from qa_elements import qa_elements, qa_step

    elems = qa_elements(b=2, e=1)
    # elems.C == 6, elems.F == 8, elems.d == 3, etc.
    # All invariants pre-verified. Cannot be wrong.

    # For state evolution (T-operator), use qa_step:
    b_next, e_next = qa_step(b, e, m=9)
"""

QA_COMPLIANCE = "canonical_element_module — authoritative source; not an empirical script"

import json
import sys
from typing import NamedTuple


# ---------------------------------------------------------------------------
# QA Element Result (immutable)
# ---------------------------------------------------------------------------

class QAElements(NamedTuple):
    """Immutable 16-element result. All values are exact integers.

    Raw derived coordinates: d = b+e, a = b+2e. No modular reduction.
    """
    b: int
    e: int
    d: int
    a: int
    A: int   # a*a
    B: int   # b*b
    C: int   # 2*d*e (green quadrance)
    D: int   # d*d
    E: int   # e*e
    F: int   # b*a = d*d - e*e (red quadrance / semi-latus)
    G: int   # d*d + e*e (blue quadrance)
    H: int   # C + F
    I: int   # C - F (conic discriminant)
    J: int   # b*d
    K: int   # a*d
    L: int   # C*F // 12 (integer when 12 | C*F)
    X: int   # e*d = C/2
    W: int   # d*(e+a)
    Y: int   # A - D = a*a - d*d
    Z: int   # E + K = e*e + a*d


# ---------------------------------------------------------------------------
# Element computation — NO MODULUS PARAMETER
# ---------------------------------------------------------------------------

def qa_elements(b: int, e: int) -> QAElements:
    """Compute all 16 QA elements from state (b, e).

    d = b + e (RAW, A2). a = b + 2*e (RAW, A2).
    No modulus parameter exists. This is by design — it is impossible
    to accidentally mod-reduce element inputs through this function.

    Raises ValueError if b < 1 or e < 1 (A1 violation).
    Raises AssertionError if any structural invariant fails.
    """
    if b < 1 or e < 1:
        raise ValueError(f"A1 violation: b={b}, e={e}. States must be >= 1.")

    # A2: derived coordinates (raw, never reduced)
    d = b + e
    a = b + 2 * e

    # 16 elements (S1: product form, no power operator)
    A = a * a
    B = b * b
    C = 2 * d * e
    D = d * d
    E = e * e
    F = b * a           # also equals d*d - e*e (verified below)
    G = d * d + e * e
    H = C + F
    I_val = C - F
    J = b * d
    K = a * d
    X = e * d
    W = d * (e + a)
    Y = A - D
    Z = E + K

    # L: area element, integer when 12 divides C*F
    CF = C * F
    L = CF // 12 if CF % 12 == 0 else CF // 12  # always compute

    # --- INVARIANT ASSERTIONS (fail-fast) ---

    # 1. Chromogeometry: C*C + F*F == G*G
    assert C * C + F * F == G * G, (
        f"Chromogeometry violated at ({b},{e}): "
        f"{C}*{C}+{F}*{F}={C*C+F*F} != {G}*{G}={G*G}"
    )

    # 2. G = (A+B)/2
    assert 2 * G == A + B, (
        f"G=(A+B)/2 violated at ({b},{e}): 2*{G}={2*G} != {A}+{B}={A+B}"
    )

    # 3. A - B = 2*C
    assert A - B == 2 * C, (
        f"A-B=2C violated at ({b},{e}): {A}-{B}={A-B} != 2*{C}={2*C}"
    )

    # 4. F identity: b*a == d*d - e*e
    assert F == D - E, (
        f"F identity violated at ({b},{e}): b*a={F} != d*d-e*e={D-E}"
    )

    # 5. d > e (since b >= 1, d = b+e > e)
    assert d > e, (
        f"d > e violated at ({b},{e}): d={d}, e={e}"
    )

    # 6. F > 0 (since d > e and both positive)
    assert F > 0, (
        f"F > 0 violated at ({b},{e}): F={F}"
    )

    # 7. C >= 4 (since d >= 2, e >= 1)
    assert C >= 4, (
        f"C >= 4 violated at ({b},{e}): C={C}"
    )

    # 8. Eisenstein norm: F*F - F*W + W*W == Z*Z
    assert F * F - F * W + W * W == Z * Z, (
        f"Eisenstein norm violated at ({b},{e}): "
        f"F*F-F*W+W*W={F*F - F*W + W*W} != Z*Z={Z*Z}"
    )

    # 9. Eisenstein dual: Y*Y - Y*W + W*W == Z*Z
    assert Y * Y - Y * W + W * W == Z * Z, (
        f"Eisenstein dual violated at ({b},{e}): "
        f"Y*Y-Y*W+W*W={Y*Y - Y*W + W*W} != Z*Z={Z*Z}"
    )

    return QAElements(
        b=b, e=e, d=d, a=a,
        A=A, B=B, C=C, D=D, E=E, F=F, G=G,
        H=H, I=I_val, J=J, K=K, L=L,
        X=X, W=W, Y=Y, Z=Z,
    )


def qa_elements_from_direction(d: int, e: int) -> QAElements:
    """Compute elements from a Pythagorean direction (d, e).

    Derives b = d - e. Asserts d > e > 0.
    Use this when the validator's input is a direction, not a state.
    """
    if d <= e or e < 1:
        raise ValueError(
            f"Invalid direction: d={d}, e={e}. Need d > e > 0."
        )
    b = d - e
    return qa_elements(b, e)


# ---------------------------------------------------------------------------
# State evolution — the ONLY place modular arithmetic appears
# ---------------------------------------------------------------------------

def qa_step(b: int, e: int, m: int) -> tuple:
    """T-operator: one step of QA state evolution in S_m.

    Returns (b_next, e_next) = (e, ((b+e-1) % m) + 1).

    THIS IS FOR DYNAMICS ONLY. Do NOT use this for element computation.
    Use qa_elements(b, e) for elements.
    """
    return (e, ((b + e - 1) % m) + 1)


def qa_mod(x: int, m: int) -> int:
    """A1-compliant modular reduction: maps x to {1,...,m}.

    THIS IS FOR DYNAMICS ONLY. Elements use raw coordinates.
    """
    return ((x - 1) % m) + 1


# ---------------------------------------------------------------------------
# Fixture verification
# ---------------------------------------------------------------------------

def verify_fixture_witness(witness: dict) -> list:
    """Check a fixture witness dict against canonical element computation.

    Looks for 'state_be' or ('b','e') keys to find the input pair.
    Compares any element keys (C, F, G, d, a, etc.) against canonical values.

    Returns list of error strings. Empty list = all correct.
    """
    errors = []

    # Find (b, e)
    be = witness.get("state_be")
    if be and len(be) == 2:
        b, e = be
    elif "b" in witness and "e" in witness:
        b, e = witness["b"], witness["e"]
    else:
        return ["Cannot find (b,e) in witness — need 'state_be' or 'b'+'e' keys"]

    if b < 1 or e < 1:
        return [f"A1 violation: b={b}, e={e}"]

    canon = qa_elements(b, e)

    # Check all element keys that appear in the witness
    element_keys = [
        "d", "a", "A", "B", "C", "D", "E", "F", "G",
        "H", "I", "J", "K", "L", "X", "W", "Y", "Z",
    ]

    for key in element_keys:
        if key in witness:
            claimed = witness[key]
            actual = getattr(canon, key)
            if claimed != actual:
                errors.append(
                    f"{key}: claimed {claimed}, canonical {actual} "
                    f"at (b={b},e={e})"
                )

    # Check tuple_beda if present
    if "tuple_beda" in witness:
        claimed_tuple = witness["tuple_beda"]
        actual_tuple = [canon.b, canon.e, canon.d, canon.a]
        if claimed_tuple != actual_tuple:
            errors.append(
                f"tuple_beda: claimed {claimed_tuple}, "
                f"canonical {actual_tuple}"
            )

    return errors


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """Exhaustive verification on all (b,e) in [1..24] x [1..24].

    Checks:
    1. All 9 invariants hold (via assertions in qa_elements)
    2. Known reference values match
    3. C >= 4 for every pair
    4. d > e for every pair
    5. F > 0 for every pair
    """
    # Known reference values
    ref = qa_elements(1, 1)
    assert ref.d == 2 and ref.a == 3, f"(1,1) d,a wrong: {ref.d},{ref.a}"
    assert ref.C == 4 and ref.F == 3, f"(1,1) C,F wrong: {ref.C},{ref.F}"
    assert ref.G == 5, f"(1,1) G wrong: {ref.G}"

    ref2 = qa_elements(2, 1)
    assert ref2.d == 3 and ref2.a == 4, f"(2,1) d,a wrong: {ref2.d},{ref2.a}"
    assert ref2.C == 6 and ref2.F == 8, f"(2,1) C,F wrong: {ref2.C},{ref2.F}"

    ref3 = qa_elements(5, 2)
    assert ref3.d == 7 and ref3.a == 9, f"(5,2) d,a wrong: {ref3.d},{ref3.a}"
    assert ref3.C == 28 and ref3.F == 45, f"(5,2) C,F wrong: {ref3.C},{ref3.F}"

    ref4 = qa_elements(9, 9)
    assert ref4.d == 18 and ref4.a == 27, f"(9,9) d,a wrong: {ref4.d},{ref4.a}"
    assert ref4.C == 324 and ref4.F == 243, f"(9,9) C,F wrong: {ref4.C},{ref4.F}"

    # Exhaustive sweep
    count = 0
    min_C = None
    for b in range(1, 25):
        for e in range(1, 25):
            elems = qa_elements(b, e)  # assertions fire inside
            if min_C is None or elems.C < min_C:
                min_C = elems.C
            count += 1

    assert min_C == 4, f"min_C should be 4, got {min_C}"
    assert count == 576, f"expected 576 pairs, got {count}"

    # qa_elements_from_direction consistency
    for d in range(2, 20):
        for e_dir in range(1, d):
            b = d - e_dir
            from_state = qa_elements(b, e_dir)
            from_dir = qa_elements_from_direction(d, e_dir)
            assert from_state == from_dir, (
                f"State vs direction mismatch at d={d},e={e_dir}"
            )

    # qa_step smoke test
    assert qa_step(1, 1, 9) == (1, 2), f"qa_step(1,1,9) wrong"
    assert qa_step(9, 9, 9) == (9, 9), f"qa_step(9,9,9) wrong"

    print(json.dumps({
        "ok": True,
        "pairs_verified": count,
        "min_C": min_C,
        "invariants": 9,
        "reference_values_checked": 4,
        "direction_consistency": "171 pairs verified",
    }, indent=2))
    return True


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        try:
            ok = self_test()
            sys.exit(0 if ok else 1)
        except (AssertionError, ValueError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            sys.exit(1)
    else:
        # Interactive: compute elements for given b, e
        if len(sys.argv) >= 3:
            b, e = int(sys.argv[1]), int(sys.argv[2])
            elems = qa_elements(b, e)
            print(json.dumps(elems._asdict(), indent=2))
        else:
            print("Usage:")
            print("  python qa_elements.py --self-test")
            print("  python qa_elements.py <b> <e>")
            sys.exit(1)
