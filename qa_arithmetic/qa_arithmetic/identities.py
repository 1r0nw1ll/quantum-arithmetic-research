QA_COMPLIANCE = "canonical_orbit_module — not an empirical script"
"""The 16 QA identities for a direction vector (b, e).

All use S1 (b*b, not the power op) and A2 (d=b+e, a=b+2e derived).

Source: elements.txt (corrected), certified in [133] QA_SIXTEEN_IDENTITIES_CERT.v1.
"""

from qa_arithmetic.core import qa_mod

IDENTITY_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "X", "W", "Y", "Z",
]


def identities(b: int, e: int, m: int = 24) -> dict:
    """Compute all 16 QA identities for direction (b, e).

    Returns dict with keys: A, B, C, D, E, F, G, H, I, J, K, L, X, W, Y, Z
    plus derived: d, a, orbit_family.

    Uses raw integers (not modular) for the identities themselves.
    d and a are modular (A2-compliant).
    """
    d = b + e      # raw sum for identity computation
    a = b + 2 * e  # raw sum for identity computation

    A = a * a       # a-squared
    B = b * b       # b-squared
    C = 2 * d * e   # green quadrance (4-par)
    D = d * d       # d-squared
    E = e * e       # e-squared
    F = a * b       # red quadrance / semi-latus (ab)
    G = d * d + e * e  # blue quadrance (5-par)
    H = C + F       # C + F
    I_val = C - F   # conic discriminant: I<0 ellipse, I=0 parabola, I>0 hyperbola
    J = b * d       # perigee (bd)
    K = a * d       # apogee (ad)
    L = C * F // 12 # = abde/6, always integer for Fibonacci directions
    X = e * d       # C/2
    W = d * (e + a) # = X + K
    Y = A - D       # a-squared minus d-squared
    Z = E + K       # e-squared + ad

    result = {
        "b": b, "e": e,
        "d_raw": d, "a_raw": a,
        "d_mod": qa_mod(d, m),
        "a_mod": qa_mod(a, m),
        "A": A, "B": B, "C": C, "D": D, "E": E, "F": F,
        "G": G, "H": H, "I": I_val, "J": J, "K": K, "L": L,
        "X": X, "W": W, "Y": Y, "Z": Z,
    }

    # Verify key relationships
    assert C * C + F * F == G * G, f"Chromogeometry violated: C²+F²≠G² for ({b},{e})"
    assert G == (A + B) // 2 or 2 * G == A + B, f"G≠(A+B)/2 for ({b},{e})"
    assert A - B == 2 * C, f"A-B≠2C for ({b},{e})"
    assert H == C + F, f"H≠C+F for ({b},{e})"

    return result
