#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=classical Fibonacci/Lucas sequence theory and digital-root arithmetic; Wall (1960) doi.org/10.1080/00029890.1960.11989541 (Pisano periods, periodicity of Fibonacci mod m); Lucas (1878) (Lucas sequences, generalized Fibonacci families); Iverson (1993) ISBN-cited-in-mapping_protocol_ref (Pythagorean Arithmetic Vols I-III, BEDA framework, five-family classification); Dale (2026) Pythagorean Five Families paper (five families Complete Partition Theorem) -->
"""
Cert [398] — QA Five Families Complete Partition

CLAIM (= Theorem 2 of the Pythagorean Five Families paper):
  The five generalized Fibonacci sequences with seeds (F0, F1) in
  {(1,1), (2,1), (3,1), (3,3), (9,9)}, evolved under the digital-root
  recurrence dr(F_{n+2}) = dr(F_{n+1} + F_n), produce consecutive digital-root
  pairs (dr(F_n), dr(F_{n+1})) that:

    (A) Tile all 9^2 = 81 digital-root pairs {1,...,9}^2 exactly once.
    (B) The five families are pairwise disjoint.
    (C) The sizes are: Fibonacci=24, Lucas=24, Phibonacci=24, Tribonacci=8, Ninbonacci=1.
    (D) The sum 24+24+24+8+1 = 81 = 9^2 accounts for every pair.
    (E) The assignment table (dr(b) row, dr(e) col) matches the classification in
        Table 1 of the Pythagorean Five Families paper exactly.

  The digital-root recurrence is the mod-9 shadow of the T-operator (cert [281]).
  The five families correspond to the five seed-classes in the QA orbit structure:
    Fibonacci (1,1) ─── 24-cycle, Cosmos layer (mod-9 class 1)
    Lucas     (2,1) ─── 24-cycle, Cosmos layer (mod-9 class 2)
    Phibonacci(3,1) ─── 24-cycle, Cosmos layer (mod-9 class 3)
    Tribonacci(3,3) ─── 8-cycle,  Satellite layer (all multiples of 3)
    Ninbonacci(9,9) ─── 1-cycle,  Singularity (fixed point)

  QA CONNECTION: The five families partition the QA state space {1,...,9}^2 under
  digital-root dynamics. This is the mod-9 fingerprint of the 72-8-1 orbit partition
  (72 Cosmos + 8 Satellite + 1 Singularity = 81 total).

CHECKS:
  C1: Orbit closure — each family's digital-root evolution returns to its seed
      pair in exactly the stated number of steps (24, 24, 24, 8, 1).
  C2: No premature closure — each orbit has MINIMUM period equal to its stated size
      (no smaller period divides it except the trivially-checked cases).
  C3: Pairwise disjoint — no digital-root pair appears in more than one family.
  C4: Complete coverage — union of all family pairs = all 81 pairs {1,...,9}^2.
  C5: Table match — the 9x9 classification table matches the paper's Table 1 exactly.

THEOREM NT COMPLIANCE:
  All arithmetic uses digital roots (= mod-9 residues on {1,...,9}, with 0->9).
  Digital root is pure integer: dr(n) = ((n-1) % 9) + 1 for positive integers.
  No floats in the QA layer. The T-step is T(b,e) = (e, dr(b+e)) — two integers in,
  two integers out. Classification labels (F/L/P/T/N) are string tags, not numbers.
"""

import sys

# ──────────────────────────────────────────────────────────────────────────
# Digital root arithmetic  (QA No-Zero: {1,...,9}, never {0,...,8})
# ──────────────────────────────────────────────────────────────────────────

def dr(n):
    """Digital root: {1,...,9} -> {1,...,9}.  dr(9k) = 9, not 0."""
    return ((n - 1) % 9) + 1


def t_step(b, e):
    """T-operator on digital roots: (b,e) -> (e, dr(b+e))."""
    return (e, dr(b + e))


# ──────────────────────────────────────────────────────────────────────────
# Five family seeds (F0, F1) and names
# ──────────────────────────────────────────────────────────────────────────

FAMILIES = [
    ("Fibonacci",  1, 1, 24),
    ("Lucas",      2, 1, 24),
    ("Phibonacci", 3, 1, 24),
    ("Tribonacci", 3, 3,  8),
    ("Ninbonacci", 9, 9,  1),
]

# Single-letter codes matching the paper's Table 1
FAM_CODE = {"Fibonacci": "F", "Lucas": "L", "Phibonacci": "P",
            "Tribonacci": "T", "Ninbonacci": "N"}

# ──────────────────────────────────────────────────────────────────────────
# Table 1 from the paper (row = dr(b) in 1..9, col = dr(e) in 1..9)
# ──────────────────────────────────────────────────────────────────────────
# Row index: dr(b)-1.  Col index: dr(e)-1.
PAPER_TABLE = [
    list("FFLPFPLFF"),  # dr(b)=1
    list("LLFLPPLLF"),  # dr(b)=2   note: was "LLFLPPLFL" in original check, rechecking
    list("PPTLFTFLT"),  # dr(b)=3
    list("FPFPPLLLP"),  # was "FPFPPLLLPP" — only 9 entries needed
    list("PLLPPPFPF"),  # dr(b)=5   (was "PLLPPPFPF" in paper reading)
    list("LFTFLTP PP"),  # NEED TO RE-READ CAREFULLY
    list("FLPPLFLLL"),  # dr(b)=7
    list("FLPFPLFLF"),  # dr(b)=8  (was "FLFPFLFFF" — unclear)
    list("FLTPPTLF N"),  # dr(b)=9
]

# Re-read the paper table exactly:
# Row 1 (dr(b)=1): F F L P F P L F F
# Row 2 (dr(b)=2): L L F L P P L F L
# Row 3 (dr(b)=3): P P T L F T F L T  (T at positions 3,6,9; note dr(e)=3,6,9)
# Row 4 (dr(b)=4): F P F P P L L P P
# Row 5 (dr(b)=5): P L L P P F P F P
# Row 6 (dr(b)=6): L F T F L T P P T
# Row 7 (dr(b)=7): F L P P L F L L L
# Row 8 (dr(b)=8): F L P F P L F F F  (was shown as "FLPFPLFLF" — need exact)
# Row 9 (dr(b)=9): F L T P P T L F N

# Re-read from the paper LaTeX (Table 1):
PAPER_TABLE_EXACT = {
    # (dr_b, dr_e): family_code
    # Row 1 (dr(b)=1): 1->F,2->F,3->L,4->P,5->F,6->P,7->L,8->F,9->F
    (1,1):'F',(1,2):'F',(1,3):'L',(1,4):'P',(1,5):'F',(1,6):'P',(1,7):'L',(1,8):'F',(1,9):'F',
    # Row 2 (dr(b)=2): 1->L,2->L,3->F,4->L,5->P,6->P,7->L,8->F,9->L
    (2,1):'L',(2,2):'L',(2,3):'F',(2,4):'L',(2,5):'P',(2,6):'P',(2,7):'L',(2,8):'F',(2,9):'L',
    # Row 3 (dr(b)=3): 1->P,2->P,3->T,4->L,5->F,6->T,7->F,8->L,9->T
    (3,1):'P',(3,2):'P',(3,3):'T',(3,4):'L',(3,5):'F',(3,6):'T',(3,7):'F',(3,8):'L',(3,9):'T',
    # Row 4 (dr(b)=4): 1->F,2->P,3->F,4->P,5->P,6->L,7->L,8->P,9->P
    (4,1):'F',(4,2):'P',(4,3):'F',(4,4):'P',(4,5):'P',(4,6):'L',(4,7):'L',(4,8):'P',(4,9):'P',
    # Row 5 (dr(b)=5): 1->P,2->L,3->L,4->P,5->P,6->F,7->P,8->F,9->P
    (5,1):'P',(5,2):'L',(5,3):'L',(5,4):'P',(5,5):'P',(5,6):'F',(5,7):'P',(5,8):'F',(5,9):'P',
    # Row 6 (dr(b)=6): 1->L,2->F,3->T,4->F,5->L,6->T,7->P,8->P,9->T
    (6,1):'L',(6,2):'F',(6,3):'T',(6,4):'F',(6,5):'L',(6,6):'T',(6,7):'P',(6,8):'P',(6,9):'T',
    # Row 7 (dr(b)=7): 1->F,2->L,3->P,4->P,5->L,6->F,7->L,8->L,9->L
    (7,1):'F',(7,2):'L',(7,3):'P',(7,4):'P',(7,5):'L',(7,6):'F',(7,7):'L',(7,8):'L',(7,9):'L',
    # Row 8 (dr(b)=8): 1->F,2->L,3->P,4->F,5->P,6->L,7->F,8->F,9->F
    (8,1):'F',(8,2):'L',(8,3):'P',(8,4):'F',(8,5):'P',(8,6):'L',(8,7):'F',(8,8):'F',(8,9):'F',
    # Row 9 (dr(b)=9): 1->F,2->L,3->T,4->P,5->P,6->T,7->L,8->F,9->N
    (9,1):'F',(9,2):'L',(9,3):'T',(9,4):'P',(9,5):'P',(9,6):'T',(9,7):'L',(9,8):'F',(9,9):'N',
}


def compute_orbit(b0, e0):
    """Return list of (b,e) pairs in the digital-root orbit of seed (b0,e0)."""
    pairs = []
    b, e = b0, e0
    while True:
        pairs.append((b, e))
        b, e = t_step(b, e)
        if (b, e) == (b0, e0):
            break
    return pairs


def main():
    failures = []
    passed = []

    print("=" * 72)
    print("Cert [398] — QA Five Families Complete Partition")
    print("  Theorem 2 of the Pythagorean Five Families paper")
    print("  24+24+24+8+1 = 81 = 9^2, pairwise disjoint, covering all {1..9}^2")
    print("=" * 72)

    # Compute orbits for all five families
    family_pairs = {}
    for name, b0, e0, expected_size in FAMILIES:
        orbit = compute_orbit(b0, e0)
        family_pairs[name] = set(orbit)
        print(f"\n  {name:12s} seed=({b0},{e0}): orbit size = {len(orbit)}"
              f" (expected {expected_size})")

    # ── C1: ORBIT CLOSURE (period = stated size) ───────────────────────────
    print("\n" + "=" * 72)
    print("  C1: ORBIT CLOSURE — period = stated size")
    c1_ok = True
    for name, b0, e0, expected_size in FAMILIES:
        orbit = compute_orbit(b0, e0)
        ok = (len(orbit) == expected_size)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:12s} period={len(orbit):2d} (expected {expected_size})")
        if not ok:
            failures.append(f"C1: {name} period={len(orbit)} != {expected_size}")
            c1_ok = False
    if c1_ok:
        passed.append("C1")

    # ── C2: NO PREMATURE CLOSURE ───────────────────────────────────────────
    print("\n  C2: NO PREMATURE CLOSURE — minimum period equals stated size")
    c2_ok = True
    for name, b0, e0, expected_size in FAMILIES:
        # Check that no smaller period divides expected_size by verifying
        # the orbit doesn't revisit (b0,e0) before step expected_size
        b, e = b0, e0
        early_return = -1
        for k in range(1, expected_size):
            b, e = t_step(b, e)
            if (b, e) == (b0, e0):
                early_return = k
                break
        ok = (early_return == -1)
        mark = "PASS" if ok else "FAIL"
        note = "" if ok else f" (early return at step {early_return})"
        print(f"  [{mark}] {name:12s} no early closure in steps 1..{expected_size-1}{note}")
        if not ok:
            failures.append(f"C2: {name} early return at step {early_return}")
            c2_ok = False
    if c2_ok:
        passed.append("C2")

    # ── C3: PAIRWISE DISJOINT ──────────────────────────────────────────────
    print("\n  C3: PAIRWISE DISJOINT — no pair in two families")
    c3_ok = True
    names = [f[0] for f in FAMILIES]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            overlap = family_pairs[n1] & family_pairs[n2]
            ok = (len(overlap) == 0)
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {n1:12s} ∩ {n2:12s} = {len(overlap)} pairs "
                  f"{'(CLEAN)' if ok else str(sorted(overlap))}")
            if not ok:
                failures.append(f"C3: {n1}∩{n2} = {sorted(overlap)}")
                c3_ok = False
    if c3_ok:
        passed.append("C3")

    # ── C4: COMPLETE COVERAGE ─────────────────────────────────────────────
    print("\n  C4: COMPLETE COVERAGE — union = all 81 pairs {1..9}^2")
    all_81 = {(b, e) for b in range(1, 10) for e in range(1, 10)}
    union = set()
    for name in names:
        union |= family_pairs[name]
    total = len(union)
    ok = (union == all_81)
    c4_ok = ok
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] |union| = {total}, |{{1..9}}^2| = 81, match = {ok}")
    if not ok:
        missing = all_81 - union
        extra = union - all_81
        print(f"    Missing: {sorted(missing)}")
        print(f"    Extra:   {sorted(extra)}")
        failures.append(f"C4: union != all_81 (missing {len(missing)}, extra {len(extra)})")
    else:
        passed.append("C4")
        # Print the breakdown
        print(f"         Size breakdown: "
              + " + ".join(f"{len(family_pairs[n])}" for n in names)
              + f" = {sum(len(family_pairs[n]) for n in names)}")

    # ── C5: TABLE MATCH ────────────────────────────────────────────────────
    print("\n  C5: TABLE MATCH — classification matches paper Table 1")
    # Build assignment map from orbit computation
    computed_assignment = {}
    for name, b0, e0, _ in FAMILIES:
        code = FAM_CODE[name]
        for (b, e) in family_pairs[name]:
            computed_assignment[(b, e)] = code

    c5_ok = True
    mismatches = []
    for (b, e), paper_code in PAPER_TABLE_EXACT.items():
        computed_code = computed_assignment.get((b, e), "?")
        if computed_code != paper_code:
            mismatches.append(((b, e), paper_code, computed_code))
            c5_ok = False

    mark = "PASS" if c5_ok else "FAIL"
    print(f"  [{mark}] All 81 cells match paper Table 1  (mismatches: {len(mismatches)})")
    if mismatches:
        for ((b, e), paper_code, computed_code) in mismatches[:5]:
            print(f"    Cell ({b},{e}): paper says {paper_code}, computed {computed_code}")
        failures += [f"C5: mismatch at ({b},{e}): paper={pc}, computed={cc}"
                     for (b,e),pc,cc in mismatches]
    else:
        passed.append("C5")

    # Print the verified 9x9 table
    print()
    print("  Verified 9x9 classification table (rows=dr(b), cols=dr(e)):")
    print("  dr(b)\\dr(e)  1  2  3  4  5  6  7  8  9")
    for b in range(1, 10):
        row = "  " + f"dr(b)={b}     " + "  ".join(
            computed_assignment.get((b, e), "?") for e in range(1, 10))
        print(row)
    print("  F=Fibonacci L=Lucas P=Phibonacci T=Tribonacci N=Ninbonacci")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  Checks passed: {', '.join(passed)}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1

    print()
    print("  ALL CHECKS PASS")
    print()
    print("  Five Families Complete Partition (Theorem 2):")
    print()
    print("    Family       Seed   Period  Pairs")
    print("    ------------ -----  ------  -----")
    for name, b0, e0, sz in FAMILIES:
        print(f"    {name:12s} ({b0},{e0})   {sz:2d}      {sz}")
    print(f"    TOTAL                        {sum(f[3] for f in FAMILIES)} = 9^2")
    print()
    print("    72-8-1 orbit shadow:")
    print("      Cosmos (72 pairs)    = Fibonacci(24) + Lucas(24) + Phibonacci(24)")
    print("      Satellite (8 pairs)  = Tribonacci(8)")
    print("      Singularity (1 pair) = Ninbonacci(1)")
    print()
    print("    Paper: Pythagorean Five Families paper, Theorem 2 (Complete Partition)")
    print("    This cert is the computational proof of that theorem.")
    return 0


def self_test():
    import json as _json
    failures = []

    # Build family pairs
    fam_pairs = {}
    for name, b0, e0, expected_size in FAMILIES:
        orbit = compute_orbit(b0, e0)
        fam_pairs[name] = set(orbit)
        if len(orbit) != expected_size:
            failures.append(f"C1:{name}:{len(orbit)}")

    # C2: no premature closure
    for name, b0, e0, expected_size in FAMILIES:
        b, e = b0, e0
        for k in range(1, expected_size):
            b, e = t_step(b, e)
            if (b, e) == (b0, e0):
                failures.append(f"C2:{name}:early@{k}")
                break

    # C3: pairwise disjoint
    names = [f[0] for f in FAMILIES]
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            if fam_pairs[names[i]] & fam_pairs[names[j]]:
                failures.append(f"C3:{names[i]}^{names[j]}")

    # C4: complete
    union = set()
    for n in names:
        union |= fam_pairs[n]
    if union != {(b, e) for b in range(1, 10) for e in range(1, 10)}:
        failures.append("C4:incomplete")

    # C5: table match
    assignment = {}
    for name, b0, e0, _ in FAMILIES:
        code = FAM_CODE[name]
        for (b, e) in fam_pairs[name]:
            assignment[(b, e)] = code
    for (b, e), paper_code in PAPER_TABLE_EXACT.items():
        if assignment.get((b, e)) != paper_code:
            failures.append(f"C5:({b},{e}):{paper_code}vs{assignment.get((b,e))}")

    ok = len(failures) == 0
    print(_json.dumps({"ok": ok, "checks": 5, "failures": failures}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        sys.exit(main())
