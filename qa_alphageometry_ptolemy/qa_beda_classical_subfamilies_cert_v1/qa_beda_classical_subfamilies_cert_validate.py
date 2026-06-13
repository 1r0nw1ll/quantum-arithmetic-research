# PRIMARY-SOURCE-EXEMPT: reason=pure-algebra cert validator; primary sources cited in mapping_protocol_ref.json: Iverson (1993) Pythagorean Arithmetic Vols I-III, Sierpinski (1962) ISBN 978-0-486-43293-4, Dale (2026) Five Families paper
"""
Cert [400]: QA BEDA Classical Subfamilies
Computational proof of Theorem 3 (BEDA Characterizations of Classical Subfamilies) in the
Pythagorean Five Families paper. Three classical subfamilies of primitive Pythagorean triples
admit elegant BEDA characterizations:

  Fermat:     |C - F| = 1  ↔  |b² − 2e²| = 1  (Pell-boundary condition)
  Pythagoras: (d − e)² = 1  ↔  b = 1
  Plato:      |G − F| = 2  ↔  e = 1 (with b odd for primitive)

Five checks:
  C1  Fermat algebraic identity: C − F = 2de − ab = 2e² − b², hence |C−F|=1 ↔ |b²−2e²|=1
  C2  Pythagoras identity: d−e = b, so (d−e)²=1 ↔ b=1
  C3  Plato identity: G − F = (d²+e²) − (d²−e²) = 2e², so |G−F|=2 ↔ e=1
  C4  Verified examples match paper Table (classical subfamilies cut across five families)
  C5  Exhaustive boundary check: for b,e in {1,...,30}, Pell solutions |b²−2e²|=1 are
      exactly those on the Fermat sub-lattice; b=1 cells are exactly Pythagoras;
      e=1 cells with b odd are exactly primitive Plato

All arithmetic is exact integer arithmetic (no floats, no mod).
"""

import json
import hashlib
import sys

# BEDA triple generation (raw — no mod reduction, Theorem NT compliant)
def beda(b, e):
    d = b + e
    a = b + 2 * e
    return (b, e, d, a)

def triple(b, e):
    b_, e_, d, a = beda(b, e)
    C = 2 * d * e_   # C = 2de
    F = a * b_       # F = ab
    G = e_ * e_ + d * d  # G = e² + d²
    return (C, F, G)

def gcd(x, y):
    while y:
        x, y = y, x % y
    return abs(x)

def is_primitive(C, F, G):
    return gcd(gcd(C, F), G) == 1


# --- Paper examples table (Table: Examples of Classical Subfamilies) ---
PAPER_EXAMPLES = [
    # (subfamily, five_family, b, e, C, F, G, property_check)
    ("Fermat",     "Fibonacci",   1, 1,  4,   3,   5,   "4-3=1"),
    ("Fermat",     "Phibonacci",  3, 2,  20,  21,  29,  "|20-21|=1"),
    ("Fermat",     "Lucas",       7, 5,  120, 119, 169, "120-119=1"),
    ("Pythagoras", "Fibonacci",   1, 1,  4,   3,   5,   "(2-1)^2=1"),
    ("Pythagoras", "Fibonacci",   1, 2,  12,  5,   13,  "(3-2)^2=1"),
    ("Pythagoras", "Lucas",       1, 3,  24,  7,   25,  "(4-3)^2=1"),
    ("Plato",      "Fibonacci",   1, 1,  4,   3,   5,   "5-3=2"),
    ("Plato",      "Phibonacci",  3, 1,  8,   15,  17,  "17-15=2"),
    ("Plato",      "Fibonacci",   7, 1,  16,  63,  65,  "65-63=2"),
]


def self_test():
    results = {"ok": True, "checks": 5, "failures": [], "detail": {}}

    # C1: Fermat algebraic identity C − F = 2e² − b²
    c1_pass = True
    c1_examples = []
    for b, e in [(1, 1), (3, 2), (7, 5), (17, 12), (41, 29)]:
        C, F, G = triple(b, e)
        diff_CF = C - F
        diff_beda = 2 * e * e - b * b
        if diff_CF != diff_beda:
            c1_pass = False
            results["ok"] = False
            results["failures"].append(f"C1: (b={b},e={e}): C-F={diff_CF} ≠ 2e²-b²={diff_beda}")
        c1_examples.append({"b": b, "e": e, "C_minus_F": diff_CF, "2e2_minus_b2": diff_beda, "ok": diff_CF == diff_beda})
    # Also verify |b²-2e²|=1 iff Pell solution (spot-check known Pell solutions)
    pell_solutions = [(1,1),(3,2),(7,5),(17,12),(41,29),(99,70),(239,169)]
    for b, e in pell_solutions:
        if abs(b*b - 2*e*e) != 1:
            c1_pass = False
            results["ok"] = False
            results["failures"].append(f"C1: (b={b},e={e}) should be Pell solution but |b²-2e²|={abs(b*b-2*e*e)}≠1")
    results["detail"]["C1"] = {
        "examples": c1_examples[:3],
        "pell_solutions_verified": len(pell_solutions),
        "pass": c1_pass,
    }

    # C2: Pythagoras identity d − e = b, so (d−e)²=1 ↔ b=1
    c2_pass = True
    c2_detail = []
    for b, e in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (3, 1)]:
        _, _, d, _ = beda(b, e)
        d_minus_e = d - e
        is_pythagoras_beda = (d_minus_e * d_minus_e == 1)
        is_pythagoras_b = (b == 1)
        if is_pythagoras_beda != is_pythagoras_b:
            c2_pass = False
            results["ok"] = False
            results["failures"].append(
                f"C2: b={b},e={e}: (d-e)^2={d_minus_e*d_minus_e} but b={'1' if is_pythagoras_b else '!=1'}: mismatch"
            )
        c2_detail.append({"b": b, "e": e, "d": d, "d_minus_e": d_minus_e, "pythagoras_beda": is_pythagoras_beda, "pythagoras_b": is_pythagoras_b, "ok": is_pythagoras_beda == is_pythagoras_b})
    results["detail"]["C2"] = {"examples": c2_detail, "pass": c2_pass}

    # C3: Plato identity G − F = 2e², so |G−F|=2 ↔ e=1
    c3_pass = True
    c3_detail = []
    for b, e in [(1,1),(3,1),(7,1),(9,1),(11,1),(1,2),(3,2),(1,3)]:
        C, F, G = triple(b, e)
        G_minus_F = G - F
        twice_e_sq = 2 * e * e
        if G_minus_F != twice_e_sq:
            c3_pass = False
            results["ok"] = False
            results["failures"].append(f"C3: (b={b},e={e}): G-F={G_minus_F} ≠ 2e²={twice_e_sq}")
        is_plato_gf = (G_minus_F == 2)
        is_plato_e = (e == 1)
        if is_plato_gf != is_plato_e:
            c3_pass = False
            results["ok"] = False
            results["failures"].append(
                f"C3: (b={b},e={e}): |G-F|=2 is {is_plato_gf} but e=1 is {is_plato_e}: mismatch"
            )
        c3_detail.append({"b": b, "e": e, "G_minus_F": G_minus_F, "2e_sq": twice_e_sq, "plato": is_plato_e, "ok": G_minus_F == twice_e_sq and is_plato_gf == is_plato_e})
    results["detail"]["C3"] = {"examples": c3_detail[:5], "pass": c3_pass}

    # C4: Paper examples match
    c4_pass = True
    c4_detail = []
    for row in PAPER_EXAMPLES:
        subfamily, five_fam, b, e, C_paper, F_paper, G_paper, prop = row
        C, F, G = triple(b, e)
        match = (C == C_paper and F == F_paper and G == G_paper)
        if not match:
            c4_pass = False
            results["ok"] = False
            results["failures"].append(
                f"C4: {subfamily} (b={b},e={e}): computed ({C},{F},{G}) ≠ paper ({C_paper},{F_paper},{G_paper})"
            )
        # Verify the classifying property
        if subfamily == "Fermat":
            prop_ok = abs(C - F) == 1
        elif subfamily == "Pythagoras":
            _, _, d, _ = beda(b, e)
            prop_ok = ((d - e) * (d - e) == 1)
        else:  # Plato
            prop_ok = abs(G - F) == 2
        if not prop_ok:
            c4_pass = False
            results["ok"] = False
            results["failures"].append(f"C4: {subfamily} (b={b},e={e}): property check failed")
        c4_detail.append({"subfamily": subfamily, "b": b, "e": e, "ok": match and prop_ok})
    results["detail"]["C4"] = {"examples_checked": len(PAPER_EXAMPLES), "detail": c4_detail, "pass": c4_pass}

    # C5: Exhaustive boundary check for b,e in {1,...,30}
    # Fermat: |b²-2e²|=1 ↔ |C-F|=1 (using C1 identity)
    # Pythagoras: b=1 ↔ (d-e)²=1
    # Plato primitive: e=1 and b odd ↔ |G-F|=2 and gcd(C,F,G)=1
    c5_pass = True
    fermat_pairs = 0
    pyth_pairs = 0
    plato_prim_pairs = 0
    mismatches = []
    for b in range(1, 31):
        for e in range(1, 31):
            C, F, G = triple(b, e)
            # Fermat check
            lhs_fermat = abs(C - F)
            rhs_fermat = abs(b * b - 2 * e * e)
            if lhs_fermat != rhs_fermat:
                c5_pass = False
                mismatches.append(f"Fermat identity fail at b={b},e={e}")
            if lhs_fermat == 1:
                fermat_pairs += 1
            # Pythagoras check
            _, _, d, _ = beda(b, e)
            is_pyth_beda = ((d - e) * (d - e) == 1)
            is_pyth_b = (b == 1)
            if is_pyth_beda != is_pyth_b:
                c5_pass = False
                mismatches.append(f"Pythagoras mismatch at b={b},e={e}")
            if is_pyth_b:
                pyth_pairs += 1
            # Plato check
            is_plato_gf = (abs(G - F) == 2)
            is_plato_e = (e == 1)
            if is_plato_gf != is_plato_e:
                c5_pass = False
                mismatches.append(f"Plato mismatch at b={b},e={e}")
            if is_plato_e and b % 2 == 1:
                # primitive Plato
                if is_primitive(C, F, G):
                    plato_prim_pairs += 1
    if not c5_pass:
        results["ok"] = False
        results["failures"].extend(mismatches[:5])
    results["detail"]["C5"] = {
        "range": "b,e in {1,...,30}",
        "fermat_pairs": fermat_pairs,
        "pythagoras_pairs_b_eq_1": pyth_pairs,
        "plato_primitive_pairs": plato_prim_pairs,
        "mismatches": mismatches[:5],
        "pass": c5_pass,
    }

    return results


if __name__ == "__main__":
    result = self_test()
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    sha = hashlib.sha256(output.encode()).hexdigest()
    print(f"SHA-256: {sha}", file=sys.stderr)
    if not result["ok"]:
        sys.exit(1)
