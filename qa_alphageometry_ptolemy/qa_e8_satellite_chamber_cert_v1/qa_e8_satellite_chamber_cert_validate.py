from __future__ import annotations
# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wildberger 2005 ISBN 978-0-9757492-0-8; Bourbaki 1968; Humphreys 1972 ISBN 978-0-387-90053-7) -->

QA_COMPLIANCE = (
    "cert_validator -- integer orbit arithmetic on QA Satellite (m=9); "
    "E8 roots are integer-scaled; all QA state b,e,d,a,C,F,G are integers; "
    "float projections are observer outputs only (Theorem NT)"
)
"""Cert [496]: QA-E8 Satellite Chamber Theorem.

DOWNGRADED 2026-07-04 -- see RETRACTION NOTE below. 8 of the original 11
checks remain valid; 3 (ESC_BRANCH, ESC_GRANT, ESC_ELEM_UNIQUE) are
retracted and no longer gate this cert's pass/fail status.

RETRACTION NOTE:
  branch_and_distances() returns an index into the *filtered list of
  simple roots* (order depends on which of the 240 globally-enumerated
  roots happen to be positive/simple for a given height vector, and
  where they fall in that fixed enumeration) -- NOT a coordinate-axis
  index. check_branch()/check_grant_distance()/check_elementary_
  uniqueness() all conflated "list position 0" with "Satellite axis 0
  = (6,3)", which are different things.

  Verified directly (2026-07-04): for h=G_VALS, the actual branch
  (degree-3) simple root is (0,0,2,0,-2,0,0,0) -- a Type-1 root
  touching axes 2 and 4, i.e. SAT[2]=(9,3) and SAT[4]=(3,6). It does
  NOT touch axis 0 = (6,3) at all. "branch==0" was a coincidence
  between two unrelated indices, not a geometric fact.

  This is not fixable by correcting the index: 7 of the 8 simple roots
  in this chamber are Type-2 (nonzero at all 8 coordinates
  simultaneously), so there is no well-defined sense in which a single
  Satellite axis "is" the branch node or sits "at distance k" from it --
  the graph structure does not decompose per-axis the way the original
  theorem assumed. ESC_BRANCH, ESC_GRANT, and ESC_ELEM_UNIQUE are
  retracted, not corrected, because their premise is ill-posed.

  The wall-bound / chamber-selection results (ESC_WALL_LOWER/UPPER,
  ESC_ISO_INTERVAL, ESC_G2_EXITS) do NOT depend on this axis-to-branch
  mapping (they compare root-projection signs or full simple-root SETS
  directly) and remain valid, independently re-verified 2026-07-04.

PRIMARY CLAIM (narrowed):
  For QA mod m=9, the Satellite orbit has 8 states anchored at (6,3)
  (the unique step whose reduced triple is the fundamental (3,4,5),
  verified independently against all 8 orbit members -- not merely
  the first one satisfying C<F, since 3 of the 8 satisfy that alone).

  For the height function h = alpha*d^2 + beta*e^2 (alpha, beta > 0),
  define positive roots as those with h-projection > 0, simple roots as
  the minimal positive roots under the partial order r > r-s (s positive).

  THEOREM (still valid): h lies in the same E8 Weyl chamber as h=G if
  and only if 7/12 < alpha/beta < 13/12. The unique ISOTROPIC choice
  (alpha=beta) gives ratio 1, in the interior of that interval.
  Normalizing alpha=beta=1 gives h = G = d^2+e^2.

SUB-CLAIMS:
  (A) QA PYTHAGOREAN TRIPLE: C^2 + F^2 = G^2 for all (b,e) in {1..9}^2
      where d=b+e, a=b+2e (raw), C=2*e*d, F=a*b, G=d*d+e*e.

  (B) PARITY REDUCTION: b+e+d+a+C+F+G ≡ b (mod 2) for all 576 pairs.

  (C) E8 ROOT SYSTEM: 240 roots (112 Type-1 + 128 Type-2) constructed
      from Satellite axis labels satisfy E8 inner-product axioms.
      Using the x2-scaled convention (Type-1: +-2 entries; Type-2: +-1
      entries), every root has squared norm 8 and adjacent simple roots
      have inner product -2. Gram/Cartan matrix has determinant 1.

  (D) WALL BOUNDS: The three non-trivial Type-2 wall roots separating
      the G-chamber from neighboring chambers are:
        W1 = [-1,-1,+1,-1,+1,-1,+1,+1]  G-proj = +45
        W2 = [+1,-1,+1,-1,-1,-1,+1,+1]  G-proj = -9
        W3 = [+1,+1,+1,+1,+1,-1,-1,+1]  G-proj = +243
      For h = alpha*d^2 + beta*e^2: W1 forces alpha/beta > 7/12;
      W2 forces alpha/beta < 13/12; W3 is auto-satisfied.

  (E) G^2 EXITS: G^2 = (d^2+e^2)^2 preserves Satellite axis ordering
      but crosses the W1 and W3 walls (projections change sign) --
      proving chamber selection depends on metric values, not axis order.

  (F) ELEMENTARY UNIQUENESS -- RETRACTED 2026-07-04: this claimed G was
      the unique invariant in {b,e,d,a,C,F,G} "placing (6,3) at the E8
      branch node." See RETRACTION NOTE -- the underlying premise
      (individual axes correspond to Dynkin-diagram positions) does not
      hold, so this claim is retracted rather than corrected.

CHECKS (ESC = E8 Satellite Chamber):
  ESC_PYTH          C^2+F^2=G^2 for all 576 QA pairs
  ESC_PARITY        b+e+d+a+C+F+G ≡ b (mod 2) for all 576 pairs
  ESC_ROOTS         240 roots: 112 Type-1 + 128 Type-2
  ESC_GRAM          G-chamber simple system: Cartan matrix det = 1
  ESC_WALL_LOWER    W1-proj(G) = +45 > 0  (alpha/beta > 7/12)
  ESC_WALL_UPPER    W2-proj(G) = -9 < 0   (alpha/beta < 13/12)
  ESC_ISO_INTERVAL  alpha=beta=1: ratio 1 in (7/12, 13/12) exactly
  ESC_G2_EXITS      G^2 has opposite sign on W1 and/or W3
  ESC_CLOSURE_NO_RESCUE  closes the retraction's open question (see CLOSURE NOTE below)

  RETRACTED (computed and reported for audit, but do not gate PASS/FAIL):
  ESC_BRANCH        (was) h=G places (6,3) at E8 branch node
  ESC_GRANT         (was) (3,6) [Grant LRT] is at distance 4 from branch
  ESC_ELEM_UNIQUE   (was) G unique in {b,e,d,a,C,F,G} for branch=(6,3)

CLOSURE NOTE (2026-07-04): the retraction above left an open question --
is the axis-to-branch mapping merely miscomputed, or is there NO natural
QA-derived axis assignment that gives a well-defined per-axis Dynkin
structure at all? Checked directly: the "textbook" per-axis diagram
(exactly 1 Type-2 simple root, 7 Type-1 roots forming the classical
D8-extended-to-E8 chain+fork) exists, but ONLY for height vectors with
super-increasing spacing (each axis value strictly exceeding the sum of
all previous ones) -- confirmed for 3 independent super-increasing
witnesses. NONE of the 7 elementary QA invariants {b,e,d,a,C,F,G}, in raw
orbit order or sorted, reach that chamber (b,e,d,a fail distinctness; C is
degenerate; F and G land in messy mostly-Type-2 chambers, sorting G does
not rescue it -- lands in a DIFFERENT messy chamber instead). QA's
Satellite invariants are all within roughly the same order of magnitude
(G_VALS spans 45-306, a ~7x range); super-increasing spacing requires
exponential separation, which no natural bounded polynomial function of
QA state can produce over 8 terms. This closes the question: the branch
concept is structurally unrecoverable via any elementary QA invariant, not
merely awaiting a corrected index.

Primary sources (mathematical):
  Wildberger, N.J. (2005). Divine Proportions. Wild Egg Books.
  Bourbaki (1968). Groupes et algebres de Lie, Ch. 4-6.
  Humphreys, J.E. (1972). Introduction to Lie Algebras. Springer.
"""

from collections import deque
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import List, Tuple
import json
import sys

M = 9  # QA modulus


# ---------------------------------------------------------------------------
# QA arithmetic  (A1: states in {1..9}; A2: d,a derived; S1: no **2)
# ---------------------------------------------------------------------------

def qa_step(b: int, e: int) -> Tuple[int, int]:
    """One QA Fibonacci step on {1..m}^2 (A1-compliant)."""
    nb = e
    ne = ((b + e - 1) % M) + 1
    return nb, ne


def qa_inv(b: int, e: int) -> Tuple[int, int, int, int, int, int, int]:
    """Return (b,e,d,a,C,F,G) with d,a raw (no mod reduction) per A2."""
    d = b + e
    a = b + 2 * e
    C = 2 * e * d
    F = a * b
    G = d * d + e * e
    return b, e, d, a, C, F, G


def satellite_orbit() -> List[Tuple[int, int]]:
    """8-state Satellite orbit starting from canonical anchor (6,3)."""
    anchor = (6, 3)
    orbit = [anchor]
    cur = qa_step(*anchor)
    while cur != anchor:
        orbit.append(cur)
        cur = qa_step(*cur)
    assert len(orbit) == 8, f"Satellite orbit length {len(orbit)} != 8"
    return orbit


# ---------------------------------------------------------------------------
# E8 root system  (x2-scaled: Type-1 entries +-2; Type-2 entries +-1)
# ---------------------------------------------------------------------------

def e8_roots_scaled() -> List[Tuple[int, ...]]:
    """All 240 E8 roots in x2-scaled convention. Integer entries only."""
    roots: List[Tuple[int, ...]] = []
    # Type-1: two nonzero entries each +-2 (112 roots = 4 * C(8,2))
    for i, j in combinations(range(8), 2):
        for si in (2, -2):
            for sj in (2, -2):
                r = [0] * 8
                r[i] = si
                r[j] = sj
                roots.append(tuple(r))
    # Type-2: all +-1 with even number of -1s (128 roots = 2^7)
    for bits in range(256):
        neg_count = bin(bits).count("1")
        if neg_count % 2 == 0:
            roots.append(tuple(-1 if (bits >> k) & 1 else 1 for k in range(8)))
    return roots


def dot(r: Tuple[int, ...], s: Tuple[int, ...]) -> int:
    return sum(r[k] * s[k] for k in range(8))


# ---------------------------------------------------------------------------
# Weyl chamber computation
# ---------------------------------------------------------------------------

def simple_roots(h_vals: Tuple[int, ...], roots: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """
    Simple roots for the Weyl chamber containing h_vals.
    Returns None if h_vals is on a chamber wall (some projection = 0).
    Uses integer arithmetic; h_vals must be a generic height vector.
    """
    proj = [dot(r, h_vals) for r in roots]
    if any(p == 0 for p in proj):
        return None  # degenerate
    pos = [roots[i] for i in range(len(roots)) if proj[i] > 0]
    pos_set = set(pos)
    simple = []
    for r in pos:
        is_simple = True
        for s in pos:
            diff = tuple(r[k] - s[k] for k in range(8))
            if diff != (0,) * 8 and diff in pos_set:
                is_simple = False
                break
        if is_simple:
            simple.append(r)
    return simple


def cartan_matrix(simple: List[Tuple[int, ...]]) -> List[List[int]]:
    """Cartan matrix A_ij = dot(simple[i], simple[j]) // 4."""
    n = len(simple)
    return [[dot(simple[i], simple[j]) // 4 for j in range(n)] for i in range(n)]


def det_int(m: List[List[int]]) -> int:
    """Integer determinant via expansion (exact, works for small n)."""
    n = len(m)
    if n == 1:
        return m[0][0]
    total = 0
    for j in range(n):
        sub = [[m[r][c] for c in range(n) if c != j] for r in range(1, n)]
        total += ((-1) ** j) * m[0][j] * det_int(sub)
    return total


def branch_and_distances(simple: List[Tuple[int, ...]]) -> Tuple[int | None, List[int]]:
    """
    Returns (branch_index, distances_from_branch) for the simple root system.
    branch_index is the unique degree-3 node (E8 branch node), or None if absent.
    """
    g = cartan_matrix(simple)
    n = len(simple)
    degs = [sum(1 for j in range(n) if j != i and g[i][j] == -1) for i in range(n)]
    if degs.count(3) != 1:
        return None, []
    branch = degs.index(3)
    adj = {i: [j for j in range(n) if j != i and g[i][j] == -1] for i in range(n)}
    dist = [-1] * n
    dist[branch] = 0
    q: deque = deque([branch])
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                q.append(nb)
    return branch, dist


# ---------------------------------------------------------------------------
# Satellite orbit canonical data
# ---------------------------------------------------------------------------

SAT = satellite_orbit()  # [(6,3),(3,9),(9,3),(3,3),(3,6),(6,9),(9,6),(6,6)]
INV = [qa_inv(b, e) for b, e in SAT]
# G values for the 8 Satellite steps
G_VALS = tuple(inv[6] for inv in INV)     # (90,225,153,45,117,306,261,180)
D_VALS = tuple(inv[2] for inv in INV)
E_VALS = tuple(inv[1] for inv in INV)

# Binding chamber walls for h = alpha*d^2 + beta*e^2:
#   WALL_ROOT_LOWER: A=108, B=-63,  G-proj=+45 -> lower bound 7/12
#   WALL_ROOT_UPPER: A=108, B=-117, G-proj=-9  -> upper bound 13/12 (tightest)
WALL_ROOT_LOWER = (-1, -1, +1, -1, +1, -1, +1, +1)
WALL_ROOT_UPPER = (+1, -1, +1, -1, -1, -1, +1, +1)
# Walls that flip sign between G-chamber and G^2-chamber (for ESC_G2_EXITS):
WALL_ROOT_1 = (-1, -1, +1, -1, +1, -1, +1, +1)   # G-proj=+45,  G^2-proj=-16767
WALL_ROOT_2 = (-1, -1, +1, +1, -1, +1, -1, +1)    # G-proj=-9,   G^2-proj=+10935
WALL_ROOT_3 = (+1, +1, +1, +1, +1, -1, -1, +1)    # G-proj=+243, G^2-proj=-31509

ROOTS = e8_roots_scaled()


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------

def check_pythagorean() -> bool:
    """ESC_PYTH: C^2+F^2=G^2 for all 576 QA pairs in {1..9}^2."""
    for b in range(1, M + 1):
        for e in range(1, M + 1):
            _, _, d, a, C, F, G = qa_inv(b, e)
            if C * C + F * F != G * G:
                return False
    return True


def check_parity() -> bool:
    """ESC_PARITY: b+e+d+a+C+F+G ≡ b (mod 2) for all 576 pairs."""
    for b in range(1, M + 1):
        for e in range(1, M + 1):
            bv, ev, d, a, C, F, G = qa_inv(b, e)
            if (bv + ev + d + a + C + F + G) % 2 != bv % 2:
                return False
    return True


def check_roots_count() -> Tuple[bool, int, int]:
    """ESC_ROOTS: 240 roots split 112 Type-1 + 128 Type-2."""
    type1 = [r for r in ROOTS if r.count(0) == 6]
    type2 = [r for r in ROOTS if 0 not in r]
    ok = len(type1) == 112 and len(type2) == 128 and len(ROOTS) == 240
    return ok, len(type1), len(type2)


def check_gram() -> Tuple[bool, int]:
    """ESC_GRAM: G-chamber simple system has Cartan matrix det = 1."""
    simple = simple_roots(G_VALS, ROOTS)
    if simple is None or len(simple) != 8:
        return False, 0
    det_val = det_int(cartan_matrix(simple))
    return det_val == 1, det_val


def check_branch() -> Tuple[bool, int, Tuple[int, int]]:
    """ESC_BRANCH -- RETRACTED 2026-07-04, does not gate PASS/FAIL.

    `branch` here is an index into the filtered *list* of simple roots
    (order depends on which of the 240 globally-enumerated roots pass the
    positivity filter, and their position in that fixed enumeration) --
    it is NOT a Satellite-axis index. Verified directly: for h=G_VALS the
    actual branch (degree-3) root is (0,0,2,0,-2,0,0,0), touching axes 2
    and 4 (SAT[2]=(9,3), SAT[4]=(3,6)) -- not axis 0=(6,3). The
    `sat_axis = branch` line below conflates two unrelated indices; kept
    only so the historical computation is reproducible for audit.
    """
    simple = simple_roots(G_VALS, ROOTS)
    if simple is None:
        return False, -1, (-1, -1)
    branch, _ = branch_and_distances(simple)
    if branch is None:
        return False, -1, (-1, -1)
    sat_axis = branch  # NOT a valid axis index -- see retraction note above
    step = SAT[sat_axis] if 0 <= sat_axis < len(SAT) else (-1, -1)
    return sat_axis == 0, sat_axis, step


def check_grant_distance() -> Tuple[bool, int]:
    """ESC_GRANT -- RETRACTED 2026-07-04, does not gate PASS/FAIL.

    Same issue as check_branch(): `dist` is indexed by position in the
    filtered simple-roots list, not by Satellite axis, so `dist[4]` is
    not "distance from axis 4." 7 of the 8 simple roots for this chamber
    are Type-2 (touch all 8 axes at once), so "distance from branch to a
    single axis" has no well-defined meaning here regardless of indexing.
    """
    simple = simple_roots(G_VALS, ROOTS)
    if simple is None:
        return False, -1
    _, dist = branch_and_distances(simple)
    if not dist:
        return False, -1
    d = dist[4] if len(dist) > 4 else -1  # NOT axis 4 -- see retraction note above
    return d == 4, d


def check_wall_projections() -> Tuple[bool, bool, int, int]:
    """
    ESC_WALL_LOWER / ESC_WALL_UPPER:
    WALL_ROOT_LOWER-proj(G) > 0  (alpha/beta > 7/12)
    WALL_ROOT_UPPER-proj(G) < 0  (alpha/beta < 13/12)
    Returns (lower_ok, upper_ok, lower_proj, upper_proj).
    """
    wl = dot(WALL_ROOT_LOWER, G_VALS)
    wu = dot(WALL_ROOT_UPPER, G_VALS)
    return wl > 0, wu < 0, wl, wu


def check_iso_interval() -> Tuple[bool, Fraction, Fraction, Fraction]:
    """
    ESC_ISO_INTERVAL: alpha/beta = 1 lies strictly in (7/12, 13/12).
    Returns (ok, lower_bound, upper_bound, ratio).
    """
    # Lower bound from WALL_ROOT_LOWER: 108*alpha - 63*beta > 0 => alpha/beta > 7/12
    lower = Fraction(7, 12)
    # Upper bound from WALL_ROOT_UPPER: 108*alpha - 117*beta < 0 => alpha/beta < 13/12
    upper = Fraction(13, 12)
    ratio = Fraction(1, 1)
    ok = lower < ratio < upper
    return ok, lower, upper, ratio


def check_g2_exits() -> Tuple[bool, int]:
    """
    ESC_G2_EXITS: G^2 values have opposite sign on W1 and/or W3
    compared to G, proving G and G^2 are in different Weyl chambers.
    Returns (ok, n_sign_changes) where ok = at least 1 sign change.
    """
    g2_vals = tuple(v * v for v in G_VALS)
    sign_changes = 0
    for wall in (WALL_ROOT_1, WALL_ROOT_2, WALL_ROOT_3):
        proj_g  = dot(wall, G_VALS)
        proj_g2 = dot(wall, g2_vals)
        if (proj_g > 0) != (proj_g2 > 0):
            sign_changes += 1
    return sign_changes > 0, sign_changes


def check_elementary_uniqueness() -> Tuple[bool, List[str]]:
    """
    ESC_ELEM_UNIQUE -- RETRACTED 2026-07-04, does not gate PASS/FAIL.

    Criterion (iii) below ("branch == 0") inherits the same conflation
    documented in check_branch(): list-index 0 is not Satellite axis 0.
    Kept only so the historical computation is reproducible for audit.
    """
    inv_names = ["b", "e", "d", "a", "C", "F", "G"]
    inv_vals = [
        tuple(inv[0] for inv in INV),  # b
        tuple(inv[1] for inv in INV),  # e
        tuple(inv[2] for inv in INV),  # d
        tuple(inv[3] for inv in INV),  # a
        tuple(inv[4] for inv in INV),  # C
        tuple(inv[5] for inv in INV),  # F
        tuple(inv[6] for inv in INV),  # G
    ]
    passing = []
    for name, vals in zip(inv_names, inv_vals):
        # (i) strict ordering: all 8 values distinct
        if len(set(vals)) < 8:
            continue
        # (ii) generic: no zero projection on any E8 root
        if any(dot(r, vals) == 0 for r in ROOTS):
            continue
        # (iii) branch = axis 0 = (6,3)
        simple = simple_roots(vals, ROOTS)
        if simple is None or len(simple) != 8:
            continue
        branch, _ = branch_and_distances(simple)
        if branch == 0:
            passing.append(name)
    return passing == ["G"], passing


# Super-increasing test vectors: each axis value strictly exceeds the sum
# of all previously assigned values. This is the chamber shape (7 Type-1 +
# 1 Type-2 simple root, classical D8-extended-to-E8 chain+fork) where an
# axis-based Dynkin story would actually be well-defined.
SUPER_INCREASING_WITNESSES = (
    (1, 2, 4, 8, 16, 32, 64, 128),
    (1, 3, 7, 15, 31, 63, 127, 255),
    (5, 11, 23, 50, 105, 220, 450, 905),
)


def _chamber_type_counts(h_vals: Tuple[int, ...]) -> Tuple[int, int] | None:
    """Return (type1_count, type2_count) for the chamber containing h_vals,
    or None if h_vals is degenerate (zero projection on some root)."""
    simple = simple_roots(h_vals, ROOTS)
    if simple is None:
        return None
    type1 = sum(1 for r in simple if r.count(0) == 6)
    type2 = sum(1 for r in simple if 0 not in r)
    return type1, type2


def check_closure_no_natural_rescue() -> Tuple[bool, dict]:
    """
    ESC_CLOSURE_NO_RESCUE: closes the open question left by the 2026-07-04
    retraction -- is there ANY natural QA-derived axis assignment that
    gives the well-defined per-axis Dynkin structure (7 Type-1 + 1 Type-2
    simple roots), or is the axis-to-branch premise structurally
    unrecoverable?

    (1) POSITIVE CONTROL: super-increasing height vectors reliably give
        Type1=7, Type2=1 (the chamber where axis-based structure would be
        legitimate) -- verified for 3 independent witnesses.
    (2) NEGATIVE RESULT: none of the 7 elementary QA invariants
        {b,e,d,a,C,F,G}, in raw orbit-sequence order OR sorted, reach that
        chamber. b,e,d,a fail even the distinctness precondition; C is
        degenerate; F and G (raw or sorted) land in messy mostly-Type-2
        chambers, never Type1=7/Type2=1.

    Returns (ok, detail) where ok = both (1) holds for all witnesses AND
    (2) holds (no elementary invariant reaches Type1=7/Type2=1).
    """
    detail = {"super_increasing": [], "elementary_invariants": []}

    positive_ok = True
    for h in SUPER_INCREASING_WITNESSES:
        counts = _chamber_type_counts(h)
        hit = counts == (7, 1)
        detail["super_increasing"].append({"h": h, "counts": counts, "hit_clean_diagram": hit})
        if not hit:
            positive_ok = False

    inv_names = ["b", "e", "d", "a", "C", "F", "G"]
    inv_vals = [tuple(inv[i] for inv in INV) for i in range(7)]
    negative_ok = True
    for name, vals in zip(inv_names, inv_vals):
        for order_label, v in (("raw", vals), ("sorted", tuple(sorted(vals)))):
            if len(set(v)) < 8:
                detail["elementary_invariants"].append(
                    {"invariant": name, "order": order_label, "counts": None, "reason": "not distinct"}
                )
                continue
            counts = _chamber_type_counts(v)
            reached_clean = counts == (7, 1)
            detail["elementary_invariants"].append(
                {"invariant": name, "order": order_label, "counts": counts, "reached_clean_diagram": reached_clean}
            )
            if reached_clean:
                negative_ok = False  # an invariant DID rescue it -- closure claim would be wrong

    return positive_ok and negative_ok, detail


# ---------------------------------------------------------------------------
# Fixture validation
# ---------------------------------------------------------------------------

def validate_fixture(fixture: dict) -> dict:
    kind = fixture.get("kind", "wall_constraint")

    if kind == "wall_constraint":
        alpha = fixture["alpha"]
        beta = fixture["beta"]
        expected_in_chamber = fixture["expected_in_chamber"]
        h_vals = tuple(alpha * D_VALS[i] * D_VALS[i] + beta * E_VALS[i] * E_VALS[i]
                       for i in range(8))
        simple = simple_roots(h_vals, ROOTS)
        if simple is None or len(simple) != 8:
            in_chamber = False
            actual_branch = -1
        else:
            b, _ = branch_and_distances(simple)
            in_chamber = (b == 0)
            actual_branch = b if b is not None else -1
        return {
            "ESC_WALL_CONSTRAINT": in_chamber == expected_in_chamber,
            "actual_in_chamber": in_chamber,
            "actual_branch_axis": actual_branch,
        }

    if kind == "pythagorean":
        b, e = fixture["state"]
        expected_C = fixture["expected_C"]
        expected_F = fixture["expected_F"]
        expected_G = fixture["expected_G"]
        _, _, _, _, C, F, G = qa_inv(b, e)
        return {
            "ESC_PYTH_VALS": C == expected_C and F == expected_F and G == expected_G,
            "ESC_PYTH_IDENTITY": C * C + F * F == G * G,
        }

    if kind == "parity_sample":
        results = {}
        for b, e in fixture["pairs"]:
            bv, ev, d, a, C, F, G = qa_inv(b, e)
            ok = (bv + ev + d + a + C + F + G) % 2 == bv % 2
            results[f"parity_{b}_{e}"] = ok
        return {"ESC_PARITY_SAMPLE": all(results.values()), **results}

    return {"ESC_UNKNOWN_KIND": False}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> bool:  # noqa: PLR0912
    failures: List[str] = []

    # ESC_PYTH
    if not check_pythagorean():
        failures.append("ESC_PYTH FAIL: C^2+F^2!=G^2 for some pair")

    # ESC_PARITY
    if not check_parity():
        failures.append("ESC_PARITY FAIL: parity reduction violated")

    # ESC_ROOTS
    ok_r, n1, n2 = check_roots_count()
    if not ok_r:
        failures.append(f"ESC_ROOTS FAIL: got Type-1={n1}, Type-2={n2}, total={n1+n2}")

    # ESC_GRAM
    ok_g, det_val = check_gram()
    if not ok_g:
        failures.append(f"ESC_GRAM FAIL: det={det_val} (expected 1)")

    # ESC_WALL_LOWER / ESC_WALL_UPPER
    lower_ok, upper_ok, w1, w2 = check_wall_projections()
    if not lower_ok:
        failures.append(f"ESC_WALL_LOWER FAIL: W1-proj(G)={w1}, expected >0")
    if not upper_ok:
        failures.append(f"ESC_WALL_UPPER FAIL: W2-proj(G)={w2}, expected <0")

    # ESC_ISO_INTERVAL
    ok_iso, lo, hi, ratio = check_iso_interval()
    if not ok_iso:
        failures.append(f"ESC_ISO_INTERVAL FAIL: {ratio} not in ({lo}, {hi})")

    # ESC_G2_EXITS
    ok_g2, n_changes = check_g2_exits()
    if not ok_g2:
        failures.append("ESC_G2_EXITS FAIL: G and G^2 in same Weyl chamber")

    # RETRACTED 2026-07-04 -- computed for audit only, do NOT gate PASS/FAIL.
    # See module docstring RETRACTION NOTE and check_branch()/
    # check_grant_distance()/check_elementary_uniqueness() docstrings.
    ok_b, sat_axis, step = check_branch()
    ok_gr, dist_val = check_grant_distance()
    ok_eu, passing = check_elementary_uniqueness()

    # ESC_CLOSURE_NO_RESCUE (2026-07-04) -- DOES gate PASS/FAIL: closes the
    # retraction's open question by direct computation (see module docstring
    # CLOSURE NOTE below).
    ok_closure, closure_detail = check_closure_no_natural_rescue()
    if not ok_closure:
        failures.append(f"ESC_CLOSURE_NO_RESCUE FAIL: {closure_detail}")

    # Verify specific numeric values (regression guards)
    assert G_VALS == (90, 225, 153, 45, 117, 306, 261, 180), f"G_VALS mismatch: {G_VALS}"
    assert SAT[0] == (6, 3), f"Canonical anchor wrong: {SAT[0]}"
    assert SAT[4] == (3, 6), f"Grant LRT axis wrong: {SAT[4]}"
    lower_ok2, upper_ok2, wl, wu = check_wall_projections()
    assert wl == 45,  f"WALL_LOWER-proj(G) should be 45, got {wl}"
    assert wu == -9,  f"WALL_UPPER-proj(G) should be -9, got {wu}"
    assert dot(WALL_ROOT_1, G_VALS) == 45,   "W1-proj(G) should be 45"
    assert dot(WALL_ROOT_3, G_VALS) == 243,  "W3-proj(G) should be 243"

    if failures:
        for f in failures[:15]:
            print("FAIL:", f, file=sys.stderr)
    return len(failures) == 0


# ---------------------------------------------------------------------------
# Cert family validation
# ---------------------------------------------------------------------------

FAMILY_ID = 496
CERT_SLUG = "qa_e8_satellite_chamber_cert_v1"


def validate_cert_family(cert_dir: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    mp = cert_dir / "mapping_protocol_ref.json"
    if not mp.exists():
        errors.append("mapping_protocol_ref.json missing")
    else:
        data = json.loads(mp.read_text())
        if data.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("mapping_protocol_ref: wrong protocol_version")
        if not data.get("scope_note", "").strip():
            errors.append("mapping_protocol_ref: empty scope_note")

    fixture_dir = cert_dir / "fixtures"
    if not fixture_dir.is_dir():
        errors.append("fixtures/ directory missing")
    else:
        fix_files = list(fixture_dir.glob("*.json"))
        pass_files = [f for f in fix_files if f.name.startswith("pass_")]
        fail_files = [f for f in fix_files if f.name.startswith("fail_")]
        if not pass_files:
            errors.append("no pass_*.json fixtures found")
        if not fail_files:
            errors.append("no fail_*.json fixtures found")
        fail_count = 0
        for path in sorted(fix_files):
            try:
                fixture = json.loads(path.read_text())
            except Exception as exc:
                errors.append(f"{path.name}: JSON parse error: {exc}")
                continue
            expect_pass = fixture.get("expected", "PASS") == "PASS"
            checks = validate_fixture(fixture)
            all_pass = all(v for v in checks.values() if isinstance(v, bool))
            if all_pass != expect_pass:
                fail_count += 1
                errors.append(f"{path.name}: expected {'PASS' if expect_pass else 'FAIL'}, got {'PASS' if all_pass else 'FAIL'}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"QA E8 Satellite Chamber Cert validator [{FAMILY_ID}]"
    )
    parser.add_argument("cert_dir", nargs="?", default=str(Path(__file__).parent))
    parser.add_argument("--self-test", action="store_true", dest="selftest")
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir)
    fixture_dir = cert_dir / "fixtures"

    if args.selftest:
        st_ok = self_test()
        fam_ok, fam_errors = validate_cert_family(cert_dir)
        fix_files = list(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
        pass_files = [f for f in fix_files if f.name.startswith("pass_")]
        fail_files = [f for f in fix_files if f.name.startswith("fail_")]
        errors = ([] if st_ok else ["self_test FAIL"]) + fam_errors
        payload = {
            "ok": st_ok and fam_ok,
            "family_id": FAMILY_ID,
            "slug": CERT_SLUG,
            "pass_fixtures": len(pass_files),
            "fail_fixtures": len(fail_files),
            "errors": errors,
            "checks_summary": {
                "ESC_PYTH":       check_pythagorean(),
                "ESC_PARITY":     check_parity(),
                "ESC_ROOTS":      check_roots_count()[0],
                "ESC_GRAM":       check_gram()[0],
                "ESC_WALL_LOWER": check_wall_projections()[0],
                "ESC_WALL_UPPER": check_wall_projections()[1],
                "ESC_ISO_INTERVAL": check_iso_interval()[0],
                "ESC_G2_EXITS":   check_g2_exits()[0],
                "ESC_CLOSURE_NO_RESCUE": check_closure_no_natural_rescue()[0],
            },
            "retracted_checks_2026_07_04": {
                "reason": "axis-to-branch index conflation; see module docstring RETRACTION NOTE",
                "ESC_BRANCH":      check_branch()[0],
                "ESC_GRANT":       check_grant_distance()[0],
                "ESC_ELEM_UNIQUE": check_elementary_uniqueness()[0],
            },
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
        sys.exit(0 if payload["ok"] else 1)

    if not self_test():
        print("SELF_TEST FAIL")
        sys.exit(1)
    print("SELF_TEST PASS")

    pass_count = fail_count = 0
    for path in sorted(fixture_dir.glob("*.json")):
        with path.open() as fh:
            fixture = json.load(fh)
        expect_pass = fixture.get("expected", "PASS") == "PASS"
        checks = validate_fixture(fixture)
        all_pass = all(v for v in checks.values() if isinstance(v, bool))
        ok = all_pass == expect_pass
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        status = "PASS" if ok else "FAIL"
        bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
        print(f"{status} {path.name}: {bool_checks}")

    print(f"\nFixtures: {pass_count} PASS, {fail_count} FAIL")
    if fail_count:
        sys.exit(1)
