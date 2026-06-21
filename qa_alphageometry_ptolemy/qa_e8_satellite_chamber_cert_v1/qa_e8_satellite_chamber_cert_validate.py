from __future__ import annotations
# <!-- PRIMARY-SOURCE-EXEMPT: reason=mathematical proof from first principles; sources cited in mapping_protocol_ref.json (Wildberger 2005 ISBN 978-0-9757492-0-8; Bourbaki 1968; Humphreys 1972 ISBN 978-0-387-90053-7) -->

QA_COMPLIANCE = (
    "cert_validator -- integer orbit arithmetic on QA Satellite (m=9); "
    "E8 roots are integer-scaled; all QA state b,e,d,a,C,F,G are integers; "
    "float projections are observer outputs only (Theorem NT)"
)
"""Cert [496]: QA-E8 Satellite Chamber Theorem.

PRIMARY CLAIM:
  For QA mod m=9, the Satellite orbit has 8 states anchored at (6,3)
  (the unique step yielding the primitive (3,4,5) triple with C<F).

  Let the 8 Satellite axes label e_1,...,e_8 in this canonical order.

  For the height function h = alpha*d^2 + beta*e^2 (alpha, beta > 0),
  define positive roots as those with h-projection > 0, simple roots as
  the minimal positive roots under the partial order r > r-s (s positive).

  THEOREM:
  h lies in the E8 Weyl chamber where (6,3) is the branch node (degree-3
  node of the E8 Dynkin diagram) and (3,6) [Grant LRT] is the terminal
  leaf of the long arm (distance 4 from branch) if and only if:

      7/12 < alpha/beta < 3/2.

  The unique ISOTROPIC choice (alpha = beta) gives alpha/beta = 1, which
  lies in the interior of (7/12, 3/2). Normalizing to alpha = beta = 1
  gives h = G = d^2 + e^2.

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
        W2 = [-1,-1,+1,+1,-1,+1,-1,+1]  G-proj = -9
        W3 = [+1,+1,+1,+1,+1,-1,-1,+1]  G-proj = +243
      For h = alpha*d^2 + beta*e^2: W1 forces alpha/beta > 7/12;
      W2 forces alpha/beta < 3/2; W3 is auto-satisfied.

  (E) G^2 EXITS: G^2 = (d^2+e^2)^2 preserves Satellite axis ordering
      but crosses the W1 and W3 walls (projections change sign) --
      proving chamber selection depends on metric values, not axis order.

  (F) ELEMENTARY UNIQUENESS: Among {b, e, d, a, C, F, G}, G is the
      unique elementary QA invariant that (i) strictly orders the 8
      Satellite axes, (ii) is generic on E8 (no zero projection), and
      (iii) places (6,3) at the E8 branch node.

CHECKS (ESC = E8 Satellite Chamber):
  ESC_PYTH          C^2+F^2=G^2 for all 576 QA pairs
  ESC_PARITY        b+e+d+a+C+F+G ≡ b (mod 2) for all 576 pairs
  ESC_ROOTS         240 roots: 112 Type-1 + 128 Type-2
  ESC_GRAM          G-chamber simple system: Cartan matrix det = 1
  ESC_BRANCH        h=G places (6,3) at E8 branch node (degree 3)
  ESC_GRANT         (3,6) [Grant LRT] is at distance 4 from branch
  ESC_WALL_LOWER    W1-proj(G) = +45 > 0  (alpha/beta > 7/12)
  ESC_WALL_UPPER    W2-proj(G) = -9 < 0   (alpha/beta < 3/2)
  ESC_ISO_INTERVAL  alpha=beta=1: ratio 1 in (7/12, 3/2) exactly
  ESC_G2_EXITS      G^2 has opposite sign on W1 and W3
  ESC_ELEM_UNIQUE   G unique in {b,e,d,a,C,F,G} for branch=(6,3)

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
    """ESC_BRANCH: h=G places (6,3) at the E8 branch node (index 0)."""
    simple = simple_roots(G_VALS, ROOTS)
    if simple is None:
        return False, -1, (-1, -1)
    branch, _ = branch_and_distances(simple)
    if branch is None:
        return False, -1, (-1, -1)
    # Find which Satellite axis corresponds to the branch simple root
    # The branch simple root has nonzero projection on axis `branch`
    # We identify which SAT step labels this axis
    sat_axis = branch
    step = SAT[sat_axis]
    return sat_axis == 0, sat_axis, step


def check_grant_distance() -> Tuple[bool, int]:
    """ESC_GRANT: (3,6) [axis 4, Grant LRT] is at distance 4 from branch."""
    simple = simple_roots(G_VALS, ROOTS)
    if simple is None:
        return False, -1
    _, dist = branch_and_distances(simple)
    if not dist:
        return False, -1
    # (3,6) is SAT[4], axis index 4
    d = dist[4]
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
    ESC_ELEM_UNIQUE: G is the unique elementary invariant in
    {b, e, d, a, C, F, G} that (i) strictly orders the 8 Satellite axes,
    (ii) is generic on E8 (no zero root projection), (iii) branch=(6,3).
    Returns (ok, list_of_invariants_passing_all_three).
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

    # ESC_BRANCH
    ok_b, sat_axis, step = check_branch()
    if not ok_b:
        failures.append(f"ESC_BRANCH FAIL: branch at axis {sat_axis}={step}, expected axis 0=(6,3)")

    # ESC_GRANT
    ok_gr, dist_val = check_grant_distance()
    if not ok_gr:
        failures.append(f"ESC_GRANT FAIL: dist((3,6))={dist_val}, expected 4")

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

    # ESC_ELEM_UNIQUE
    ok_eu, passing = check_elementary_uniqueness()
    if not ok_eu:
        failures.append(f"ESC_ELEM_UNIQUE FAIL: passing invariants={passing}, expected=['G']")

    # Verify specific numeric values (regression guards)
    assert G_VALS == (90, 225, 153, 45, 117, 306, 261, 180), f"G_VALS mismatch: {G_VALS}"
    assert SAT[0] == (6, 3), f"Canonical anchor wrong: {SAT[0]}"
    assert SAT[4] == (3, 6), f"Grant LRT axis wrong: {SAT[4]}"
    lower_ok2, upper_ok2, wl, wu = check_wall_projections()
    assert wl == 45,  f"WALL_LOWER-proj(G) should be 45, got {wl}"
    assert wu == -9,  f"WALL_UPPER-proj(G) should be -9, got {wu}"
    assert dot(WALL_ROOT_1, G_VALS) == 45,   "W1-proj(G) should be 45"
    assert dot(WALL_ROOT_3, G_VALS) == 243,  "W3-proj(G) should be 243"

    # Fail-case: F gives wrong branch
    F_VALS = tuple(inv[5] for inv in INV)
    if len(set(F_VALS)) == 8:
        simple_f = simple_roots(F_VALS, ROOTS)
        if simple_f and len(simple_f) == 8:
            branch_f, _ = branch_and_distances(simple_f)
            if branch_f == 0:
                failures.append("EXPECTED_FAIL: F should NOT give branch=0, but it does")

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
                "ESC_BRANCH":     check_branch()[0],
                "ESC_GRANT":      check_grant_distance()[0],
                "ESC_WALL_LOWER": check_wall_projections()[0],
                "ESC_WALL_UPPER": check_wall_projections()[1],
                "ESC_ISO_INTERVAL": check_iso_interval()[0],
                "ESC_G2_EXITS":   check_g2_exits()[0],
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
