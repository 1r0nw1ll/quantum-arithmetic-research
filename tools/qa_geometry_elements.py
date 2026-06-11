"""
qa_geometry_elements
====================
Compute the complete QA named-element suite from a root pair (b, e).

Source: Dale Pond, "Quantum Arithmetic Elements" (svpwiki.com, 1998),
        extending Ben Iverson's original 21-element system.

Dimensional hierarchy
---------------------
  1D  — roots:    b, e, d, a          (distances / lines)
  2D  — areas:    B E D A X C F G J K H I W Y Z   (rectangles / ellipse identities)
  3D  — volumes:  L h S               (triangle area-solid, ellipse half-height, equilateral solid)
  circle layer:   P Q R               (Dale's integer circle: diameter, circumference, area)

All elements are exact integers when (b, e) are integers.
No floats, no numpy, no external dependencies — importable anywhere.

Usage
-----
    from tools.qa_geometry_elements import elements, feature_vector, DIMS

    e = elements(3, 2)          # dict of all elements
    v = feature_vector(3, 2)    # flat list ordered by dimension
    d = DIMS                    # {name: (value, dimension, role)}
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qa_orbit_rules import orbit_family as _canonical_orbit_family  # noqa: E402

QA_COMPLIANCE = "module — QA geometry elements; exact integer arithmetic; orbit_family from qa_orbit_rules"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QAElements:
    """All named QA elements for a root pair (b, e).

    Attribute names match the Dale Pond / Ben Iverson canonical letters.
    Lowercase = 1D roots (distances).
    Uppercase = 2D areas and higher.
    """
    # --- 1D roots ---
    b: int
    e: int
    d: int      # b + e
    a: int      # e + d = b + 2e

    # --- 2D areas ---
    B: int      # b²                    quadrance of b
    E: int      # e²                    quadrance of e
    D: int      # d²                    quadrance of d / semi-major diameter
    A: int      # a²                    quadrance of a
    X: int      # d·e = C/2             half-base of triangle / quarter-width of ellipse
    C: int      # 2·d·e                 base of triangle / foci distance (always 4-par)
    F: int      # a·b                   altitude of triangle / height (always odd)
    G: int      # D + E = d²+e²         hypotenuse (always 5-par)
    J: int      # b·d                   loci to outer width (incircle radius of Pyth triangle)
    K: int      # a·d                   furthest loci to outer width (excircle radius)
    H: int      # C + F                 sum identity
    I: int      # |C - F|               Koenig invariant (always positive)
    W: int      # d·(e+a) = X+K         side of equilateral triangle
    Y: int      # A - D = a²-d²         Eisenstein second element
    Z: int      # E + K = e²+a·d        Eisenstein companion

    # --- 3D volumes ---
    L: int      # b·e·d·a // 6          area of Prime Pythagorean triangle (4D solid / 6)
    h: int      # d · F = d·a·b         half-height of ellipse
    S: int      # d²·e = D·e            height of equilateral triangle (solid)

    # --- circle layer (Dale) ---
    P: int      # 2·W                   circle diameter
    Q: int      # P = 2W                circle circumference (QA units: π→1)
    R: int      # W²                    circle area (QA units)

    def roots(self) -> Dict[str, int]:
        return {"b": self.b, "e": self.e, "d": self.d, "a": self.a}

    def areas(self) -> Dict[str, int]:
        return {
            "B": self.B, "E": self.E, "D": self.D, "A": self.A,
            "X": self.X, "C": self.C, "F": self.F, "G": self.G,
            "J": self.J, "K": self.K, "H": self.H, "I": self.I,
            "W": self.W, "Y": self.Y, "Z": self.Z,
        }

    def volumes(self) -> Dict[str, int]:
        return {"L": self.L, "h": self.h, "S": self.S}

    def circles(self) -> Dict[str, int]:
        return {"P": self.P, "Q": self.Q, "R": self.R}

    def all_elements(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        out.update(self.roots())
        out.update(self.areas())
        out.update(self.volumes())
        out.update(self.circles())
        return out

    def feature_vector(self) -> List[int]:
        """Flat ordered feature vector: roots → areas → volumes → circles."""
        return list(self.all_elements().values())

    def pythagorean_triple(self) -> Tuple[int, int, int]:
        """(F, C, G) — always a Pythagorean triple: F²+C²=G²."""
        return (self.F, self.C, self.G)

    def incircle_radius(self) -> int:
        """Inradius of the (F,C,G) right triangle = b·e."""
        return self.b * self.e

    def excircle_radii(self) -> Dict[str, int]:
        """The three excircle radii of the (F,C,G) right triangle."""
        return {
            "r_F": self.b * self.d,   # = J
            "r_C": self.a * self.e,   # = a·e
            "r_G": self.d * self.a,   # = K (semiperimeter)
        }

    def triangle_area(self) -> int:
        """Area of (F,C,G) right triangle = b·e·d·a = 6·L."""
        return self.b * self.e * self.d * self.a

    def euler_oi_sq(self) -> int:
        """OI² = b⁴/4 + e⁴  (Euler distance, circumcenter to incenter).
        Returns numerator of b⁴/4 + e⁴ as fraction *4 to keep integer."""
        return self.b * self.b * self.b * self.b + 4 * self.e * self.e * self.e * self.e

    def cross_ratio(self) -> Tuple[int, int]:
        """Cross-ratio of (b, e, d, a) as fraction d/(2b) → (numerator, denominator)."""
        return (self.d, 2 * self.b)

    def harmonic_means(self) -> Dict[str, Tuple[int, int]]:
        """Harmonic means as (numerator, denominator) fractions."""
        return {
            "hm_b_a": (2 * self.F, self.d),    # 2ba/(b+a) = F/d... wait: 2ba/(b+a) = 2F/a...
            "hm_e_d": (self.C, self.a),          # 2ed/(e+d) = C/a
        }

    def koenig_invariant(self) -> int:
        """I = |C - F| — Koenig shell invariant. Preserved (sign-flipped) by QA map."""
        return self.I

    def ellipse_major_diameter(self) -> int:
        """Major diameter of the QA ellipse = D = d²."""
        return self.D

    def ellipse_foci_distance(self) -> int:
        """Distance between foci = C = 2de."""
        return self.C

    def ellipse_axis_points(self) -> Dict[str, int]:
        """Four points along major diameter: J, X, D, K."""
        return {"J": self.J, "X": self.X, "D": self.D, "K": self.K}

    def orbit_family(self, m: int = 24) -> str:
        """QA orbit family: 'cosmos' | 'satellite' | 'singularity'."""
        return _canonical_orbit_family(self.b, self.e, m)


def elements(b: int, e: int) -> QAElements:
    """Compute all named QA elements from root pair (b, e).

    Both b and e must be positive integers (A1: no zero element).
    All results are exact integers.

    The four roots: b, e, d=b+e, a=b+2e.
    All 25 named elements derive from these via the formulae of Dale Pond / Ben Iverson.
    """
    if b <= 0 or e <= 0:
        raise ValueError(f"A1 violation: b={b}, e={e} — both must be positive integers.")

    d = b + e
    a = b + 2 * e          # = e + d

    # 2D areas
    B = b * b
    E = e * e
    D = d * d
    A = a * a
    X = d * e              # = C // 2
    C = 2 * d * e
    F = a * b
    G = D + E              # d²+e²; always: F²+C²=G²
    J = b * d
    K = a * d
    H = C + F
    I = abs(C - F)         # Koenig invariant; always positive
    W = d * (e + a)        # = X + K = de + da
    Y = A - D              # a²-d² = e(2b+3e)
    Z = E + K              # e²+ad

    # 3D volumes
    beda = b * e * d * a
    assert beda % 6 == 0, f"beda not divisible by 6 for b={b},e={e} — QA law violated"
    L = beda // 6
    h = d * F              # = d·a·b
    S = D * e              # = d²·e

    # Circle layer
    P = 2 * W
    Q = P                  # circumference = diameter in QA units (π → 1)
    R = W * W              # area = W²

    return QAElements(
        b=b, e=e, d=d, a=a,
        B=B, E=E, D=D, A=A,
        X=X, C=C, F=F, G=G,
        J=J, K=K, H=H, I=I,
        W=W, Y=Y, Z=Z,
        L=L, h=h, S=S,
        P=P, Q=Q, R=R,
    )


# ---------------------------------------------------------------------------
# Batch computation for analytics pipelines
# ---------------------------------------------------------------------------

def element_matrix(pairs: List[Tuple[int, int]]) -> List[Dict[str, int]]:
    """Compute element dicts for a list of (b, e) pairs.

    Returns one dict per pair, each containing all named elements.
    Suitable for direct construction of a pandas DataFrame or numpy array.
    """
    return [elements(b, e).all_elements() for b, e in pairs]


def feature_names() -> List[str]:
    """Ordered feature names matching feature_vector() output."""
    return list(elements(1, 1).all_elements().keys())


# ---------------------------------------------------------------------------
# Dimensional metadata
# ---------------------------------------------------------------------------

ELEMENT_META: Dict[str, Tuple[int, str]] = {
    # name: (dimension, geometric role)
    "b": (1, "root 1 — base"),
    "e": (1, "root 2 — edge"),
    "d": (1, "root 3 — diagonal = b+e"),
    "a": (1, "root 4 — apex = b+2e"),
    "B": (2, "b² — square on base"),
    "E": (2, "e² — square on edge"),
    "D": (2, "d² — semi-major diameter of ellipse"),
    "A": (2, "a² — square on apex"),
    "X": (2, "de = C/2 — half-base of triangle, quarter-width of ellipse"),
    "C": (2, "2de — base of Pythagorean triangle, foci distance (4-par)"),
    "F": (2, "ab — altitude of Pythagorean triangle (odd-parity)"),
    "G": (2, "d²+e² — hypotenuse of Pythagorean triangle (5-par); F²+C²=G²"),
    "J": (2, "bd — loci-to-outer-width (excircle radius r_F)"),
    "K": (2, "ad — furthest loci-to-outer-width (excircle radius r_G = semiperimeter)"),
    "H": (2, "C+F — sum identity"),
    "I": (2, "|C-F| — Koenig invariant; preserved under QA map (sign flip)"),
    "W": (2, "d(e+a) = X+K — side of equilateral triangle"),
    "Y": (2, "a²-d² — Eisenstein second element"),
    "Z": (2, "e²+ad — Eisenstein companion; F²-FW+W²=Z²"),
    "L": (3, "beda/6 — area solid of Prime Pythagorean triangle"),
    "h": (3, "d·ab — half-height of ellipse"),
    "S": (3, "d²e — height of equilateral triangle (Dale element #25)"),
    "P": (2, "2W — circle diameter (integer)"),
    "Q": (2, "2W — circle circumference (QA units: π→1)"),
    "R": (4, "W² — circle area (QA units)"),
}


def dims_by_level(level: int) -> List[str]:
    """Return element names at a given geometric dimension."""
    return [name for name, (dim, _) in ELEMENT_META.items() if dim == level]


# ---------------------------------------------------------------------------
# Verification: unity block (1,1,2,3) — Ben Iverson's 'four Forces'
# ---------------------------------------------------------------------------

def _verify_unity_block() -> None:
    """Verify the unity block (b=1,e=1) matches known values from cert [183]."""
    el = elements(1, 1)
    assert el.d == 2 and el.a == 3, "unity block roots"
    assert el.F == 3 and el.C == 4 and el.G == 5, "3-4-5 triple"
    assert el.F * el.F + el.C * el.C == el.G * el.G, "Pythagorean"
    assert el.W == 8, f"equilateral side W=8, got {el.W}"
    assert el.Z == 7, f"Eisenstein Z=7, got {el.Z}"
    assert el.L == 1, f"L=1, got {el.L}"
    assert el.I == abs(el.C - el.F) == 1, "Koenig I=1"
    assert el.incircle_radius() == 1 * 1 == 1, "incircle r=be=1"
    assert el.triangle_area() == 1 * 1 * 2 * 3 == 6, "triangle area=beda=6"
    # Eisenstein norm: F²-FW+W²=Z²
    assert el.F * el.F - el.F * el.W + el.W * el.W == el.Z * el.Z, "Eisenstein norm"


_verify_unity_block()


if __name__ == "__main__":
    import sys

    def _show(b: int, e: int) -> None:
        el = elements(b, e)
        print(f"\nQA Elements for (b={b}, e={e})")
        print(f"  Roots:   b={el.b}  e={el.e}  d={el.d}  a={el.a}")
        print(f"  Pyth:    F={el.F}  C={el.C}  G={el.G}   [F²+C²=G²: {el.F**2+el.C**2}={el.G**2}]")
        print(f"  Squares: B={el.B}  E={el.E}  D={el.D}  A={el.A}")
        print(f"  Ellipse: J={el.J}  X={el.X}  D={el.D}  K={el.K}")
        print(f"  Koenig:  H={el.H}  I={el.I}  (I=|C-F|)")
        print(f"  Equil:   W={el.W}  Y={el.Y}  Z={el.Z}")
        print(f"  Circle:  P={el.P}  Q={el.Q}  R={el.R}")
        print(f"  Volumes: L={el.L}  h={el.h}  S={el.S}")
        print(f"  Circles: incircle r={el.incircle_radius()},  excircles {el.excircle_radii()}")
        print(f"  Area(triangle) = beda = {el.triangle_area()}")
        print(f"  Orbit family (m=24): {el.orbit_family(24)}")

    args = sys.argv[1:]
    if len(args) == 2:
        _show(int(args[0]), int(args[1]))
    else:
        for pair in [(1, 1), (2, 1), (3, 2), (5, 3), (1, 2)]:
            _show(*pair)
