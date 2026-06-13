# PRIMARY-SOURCE-EXEMPT: reason=pure-algebra cert validator; primary sources in mapping_protocol_ref.json: Iverson (1993) Pythagorean Arithmetic Vols I-III, Sierpinski (1962) ISBN 978-0-486-43293-4, Dale (2026) Five Families paper
"""
Cert [401]: QA Octave Transformation
Computational proof of Theorem 4 (Octave Transformation) in the Pythagorean Five Families paper.

For a BEDA tuple (b,e,d,a) generating a primitive triple (C,F,G):
  Transform: (b',e',d',a') = (2e, b, a, 2d)
  Female triple: (C',F',G') = (2F, 2C, 2G) — the first octave multiple

Key identities:
  d' = b' + e' = 2e + b = a  (A2 for new tuple)
  a' = b' + 2e' = 2e + 2b = 2d  (A2 for new tuple)
  C' = 2d'e' = 2ab = 2F
  F' = a'b' = 2d·2e = 4de = 2C
  G' = (e')² + (d')² = b² + a² = 2(e²+d²) = 2G
  gcd(C',F',G') = 2 when (C,F,G) is primitive

Five checks:
  C1  Transform preserves BEDA structure: d'=b'+e' and a'=b'+2e'
  C2  Female triple = (2F, 2C, 2G): legs interchange and all components double
  C3  gcd = 2: female triple has gcd exactly 2 for primitive males
  C4  Paper examples: 4 male-female pairs from Table 4 match exactly
  C5  Exhaustive b,e in {1..20}²: G'=2G identity holds for all 400 pairs;
      all primitive males produce female gcd=2
"""

import json
import hashlib
import sys

def beda(b, e):
    d = b + e
    a = b + 2 * e
    return (b, e, d, a)

def triple(b, e):
    b_, e_, d, a = beda(b, e)
    C = 2 * d * e_
    F = a * b_
    G = e_ * e_ + d * d
    return (C, F, G)

def octave_transform(b, e):
    """(b,e,d,a) -> (b',e',d',a') = (2e, b, a, 2d)"""
    _, _, d, a = beda(b, e)
    b2 = 2 * e
    e2 = b
    d2 = a        # = b + 2e; verify: b2 + e2 = 2e + b = a = d2 ✓
    a2 = 2 * d    # = 2(b+e); verify: b2 + 2*e2 = 2e + 2b = 2(b+e) = 2d = a2 ✓
    return (b2, e2, d2, a2)

def gcd(x, y):
    while y:
        x, y = y, x % y
    return abs(x)

def triple_gcd(C, F, G):
    return gcd(gcd(abs(C), abs(F)), abs(G))

def is_primitive(C, F, G):
    return triple_gcd(C, F, G) == 1

# Paper Table 4: Male Primitives and Their Female Octaves
PAPER_TABLE = [
    # (five_fam, b_m, e_m, C_m, F_m, G_m, b_f, e_f, C_f, F_f, G_f)
    ("Fibonacci",  1, 1,   4,   3,   5,   2, 1,   6,   8,   10),
    ("Phibonacci", 3, 1,   8,   15,  17,  2, 3,   30,  16,  34),
    ("Lucas",      1, 2,   12,  5,   13,  4, 1,   10,  24,  26),
    ("Lucas",      1, 3,   24,  7,   25,  6, 1,   14,  48,  50),
]


def self_test():
    results = {"ok": True, "checks": 5, "failures": [], "detail": {}}

    # C1: Transform preserves BEDA structure
    c1_pass = True
    c1_detail = []
    for b, e in [(1,1),(3,1),(1,2),(7,5),(3,2),(1,3)]:
        b2, e2, d2, a2 = octave_transform(b, e)
        d2_check = b2 + e2
        a2_check = b2 + 2 * e2
        ok = (d2 == d2_check) and (a2 == a2_check)
        if not ok:
            c1_pass = False
            results["ok"] = False
            results["failures"].append(
                f"C1: b={b},e={e}: d'={d2}≠{d2_check} or a'={a2}≠{a2_check}"
            )
        c1_detail.append({"b": b, "e": e, "b2": b2, "e2": e2, "d2_ok": d2 == d2_check, "a2_ok": a2 == a2_check})
    results["detail"]["C1"] = {"examples": c1_detail, "pass": c1_pass}

    # C2: Female triple = (2F, 2C, 2G)
    c2_pass = True
    c2_detail = []
    for b, e in [(1,1),(3,1),(1,2),(7,5),(3,2),(1,3),(7,1)]:
        C, F, G = triple(b, e)
        b2, e2, _, _ = octave_transform(b, e)
        Cf, Ff, Gf = triple(b2, e2)
        ok = (Cf == 2 * F) and (Ff == 2 * C) and (Gf == 2 * G)
        if not ok:
            c2_pass = False
            results["ok"] = False
            results["failures"].append(
                f"C2: b={b},e={e}: female ({Cf},{Ff},{Gf}) ≠ (2F={2*F},2C={2*C},2G={2*G})"
            )
        c2_detail.append({"b": b, "e": e, "male": [C,F,G], "female": [Cf,Ff,Gf], "ok": ok})
    results["detail"]["C2"] = {"examples": c2_detail, "pass": c2_pass}

    # C3: gcd(female) = 2 for primitive males
    c3_pass = True
    c3_detail = []
    for b, e in [(1,1),(3,1),(1,2),(7,5),(3,2),(1,3),(7,1),(5,2),(5,4),(11,2)]:
        C, F, G = triple(b, e)
        if not is_primitive(C, F, G):
            continue
        b2, e2, _, _ = octave_transform(b, e)
        Cf, Ff, Gf = triple(b2, e2)
        g = triple_gcd(Cf, Ff, Gf)
        ok = (g == 2)
        if not ok:
            c3_pass = False
            results["ok"] = False
            results["failures"].append(f"C3: primitive b={b},e={e}: female gcd={g}≠2")
        c3_detail.append({"b": b, "e": e, "female_gcd": g, "ok": ok})
    results["detail"]["C3"] = {"examples": c3_detail, "pass": c3_pass}

    # C4: Paper Table 4 examples
    c4_pass = True
    c4_detail = []
    for row in PAPER_TABLE:
        fam, b_m, e_m, C_m, F_m, G_m, b_f, e_f, C_f, F_f, G_f = row
        C_comp, F_comp, G_comp = triple(b_m, e_m)
        male_ok = (C_comp, F_comp, G_comp) == (C_m, F_m, G_m)
        b2, e2, _, _ = octave_transform(b_m, e_m)
        transform_ok = (b2, e2) == (b_f, e_f)
        C_f_comp, F_f_comp, G_f_comp = triple(b2, e2)
        female_ok = (C_f_comp, F_f_comp, G_f_comp) == (C_f, F_f, G_f)
        ok = male_ok and transform_ok and female_ok
        if not ok:
            c4_pass = False
            results["ok"] = False
            results["failures"].append(
                f"C4: {fam} male=({b_m},{e_m}): male_ok={male_ok} transform_ok={transform_ok} female_ok={female_ok}"
            )
        c4_detail.append({"family": fam, "male": [b_m,e_m], "female": [b_f,e_f], "ok": ok})
    results["detail"]["C4"] = {"rows": len(PAPER_TABLE), "detail": c4_detail, "pass": c4_pass}

    # C5: Exhaustive b,e in {1..20}²
    c5_pass = True
    prims_checked = 0
    prims_gcd2 = 0
    identity_failures = 0
    for b in range(1, 21):
        for e in range(1, 21):
            _, _, d, a = beda(b, e)
            # b² + a² = 2(e² + d²) (G' = 2G identity)
            lhs = b * b + a * a
            rhs = 2 * (e * e + d * d)
            if lhs != rhs:
                c5_pass = False
                identity_failures += 1
                results["ok"] = False
                results["failures"].append(f"C5: G' identity b={b},e={e}: b²+a²={lhs}≠2(e²+d²)={rhs}")
            # Primitive males
            C, F, G = triple(b, e)
            if is_primitive(C, F, G):
                prims_checked += 1
                b2, e2, _, _ = octave_transform(b, e)
                Cf, Ff, Gf = triple(b2, e2)
                if triple_gcd(Cf, Ff, Gf) == 2:
                    prims_gcd2 += 1
                else:
                    c5_pass = False
                    results["ok"] = False
                    results["failures"].append(f"C5: primitive b={b},e={e} female gcd≠2")
    results["detail"]["C5"] = {
        "range": "b,e in {1,...,20}",
        "total_pairs": 400,
        "G_prime_eq_2G_failures": identity_failures,
        "primitives_checked": prims_checked,
        "primitives_female_gcd_eq_2": prims_gcd2,
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
