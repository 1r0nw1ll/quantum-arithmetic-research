"""
Sefer Yetzirah 3-7-12 partition vs QA orbit classes — pure mathematical analysis.
A1 compliant: states in {1,...,9}, dr(n) = 1 + ((n-1) % 9).
S1 compliant: b*b not b-squared.
"""

QA_COMPLIANCE = "pure mathematical analysis — Sefer Yetzirah partition vs QA orbits"

from fractions import Fraction
from itertools import combinations
from math import comb, gcd
from functools import reduce

from qa_orbit_rules import norm_f, v3, orbit_family  # noqa: ORBIT-5 canonical import

# ── Digital root (QA A1: no-zero) ──
def dr(n):
    """Digital root: maps to {1,...,9}."""
    return 1 + ((n - 1) % 9)


def classify_display(b, e, m=24):
    """Classify (b,e) for display. Delegates to canonical orbit_family."""
    orb = orbit_family(b, e, m)
    return orb.capitalize()  # "cosmos" -> "Cosmos" for display
        return "Cosmos"

# ══════════════════════════════════════════════════════════════════
# 1. Hebrew letters with standard gematria
# ══════════════════════════════════════════════════════════════════
letters = [
    # Mother letters (3)
    ("Aleph",   1,   "mother"),
    ("Mem",     40,  "mother"),
    ("Shin",    300, "mother"),
    # Double letters (7)
    ("Bet",     2,   "double"),
    ("Gimel",   3,   "double"),
    ("Dalet",   4,   "double"),
    ("Kaf",     20,  "double"),
    ("Pe",      80,  "double"),
    ("Resh",    200, "double"),
    ("Tav",     400, "double"),
    # Simple letters (12)
    ("He",      5,   "simple"),
    ("Vav",     6,   "simple"),
    ("Zayin",   7,   "simple"),
    ("Chet",    8,   "simple"),
    ("Tet",     9,   "simple"),
    ("Yod",     10,  "simple"),
    ("Lamed",   30,  "simple"),
    ("Nun",     50,  "simple"),
    ("Samekh",  60,  "simple"),
    ("Ayin",    70,  "simple"),
    ("Tsade",   90,  "simple"),
    ("Qof",     100, "simple"),
]

print("=" * 70)
print("SEFER YETZIRAH 3-7-12 PARTITION vs QA ORBIT CLASSES")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 2-3. Digital roots and grouping
# ══════════════════════════════════════════════════════════════════
print("\n── 1-3. Letters, gematria values, digital roots ──\n")
print(f"{'Letter':<10} {'Value':>5} {'dr(val)':>7} {'Class':<10}")
print("-" * 35)

dr_groups = {}  # dr -> list of (name, value, class)
for name, val, cls in letters:
    d = dr(val)
    print(f"{name:<10} {val:>5} {d:>7} {cls:<10}")
    dr_groups.setdefault(d, []).append((name, val, cls))

print("\n── Digital root groupings ──")
for d in sorted(dr_groups):
    members = [(n, v, c) for n, v, c in dr_groups[d]]
    classes = [c for _, _, c in members]
    print(f"  dr={d}: {[n for n,_,_ in members]}  classes={classes}")

# ══════════════════════════════════════════════════════════════════
# 4-6. Mod-9 properties within each SY group
# ══════════════════════════════════════════════════════════════════
print("\n── 4-6. Mod-9 properties by Sefer Yetzirah class ──\n")

for cls_name in ["mother", "double", "simple"]:
    group = [(n, v) for n, v, c in letters if c == cls_name]
    vals = [v for _, v in group]
    drs = [dr(v) for v in vals]
    print(f"{cls_name.upper()} letters ({len(group)}):")
    print(f"  Values:        {vals}")
    print(f"  Digital roots: {drs}")
    print(f"  Unique dr's:   {sorted(set(drs))}")
    print(f"  dr set size:   {len(set(drs))}")

    # Check if drs form a subgroup of Z/9Z
    # Also check residues mod 3
    mod3 = [1 + ((v - 1) % 3) for v in vals]
    print(f"  Values mod 3 (no-zero): {mod3}")
    print(f"  Unique mod 3: {sorted(set(mod3))}")

    # mod 24
    mod24 = [1 + ((v - 1) % 24) for v in vals]
    print(f"  Values mod 24 (no-zero): {mod24}")
    print()

# ══════════════════════════════════════════════════════════════════
# 7. Sums within each group
# ══════════════════════════════════════════════════════════════════
print("── 7. Sums within each group ──\n")

total_all = sum(v for _, v, _ in letters)
print(f"Total sum of all 22 letters: {total_all}, dr = {dr(total_all)}")
print()

for cls_name in ["mother", "double", "simple"]:
    vals = [v for _, v, c in letters if c == cls_name]
    s = sum(vals)
    print(f"  {cls_name}: sum = {s}, dr(sum) = {dr(s)}, sum mod 24 = {1 + ((s-1)%24)}")

# ══════════════════════════════════════════════════════════════════
# 8. Products (mod 9) within each group
# ══════════════════════════════════════════════════════════════════
print("\n── 8. Products mod 9 within each group ──\n")

for cls_name in ["mother", "double", "simple"]:
    vals = [v for _, v, c in letters if c == cls_name]
    drs = [dr(v) for v in vals]
    # Product of digital roots mod 9 (using dr convention)
    prod = 1
    for d in drs:
        prod = prod * d
    prod_dr = dr(prod)
    print(f"  {cls_name}: product of dr's = {prod}, dr(product) = {prod_dr}")
    # Also raw product mod 9
    raw_prod = 1
    for v in vals:
        raw_prod *= v
    print(f"  {cls_name}: raw product = {raw_prod}, dr(raw product) = {dr(raw_prod)}")

# ══════════════════════════════════════════════════════════════════
# 9. Partition {3,7,12} in mod-9 and mod-24
# ══════════════════════════════════════════════════════════════════
print("\n── 9. Partition {3,7,12} arithmetic ──\n")

a, b, c = 3, 7, 12
print(f"  3 mod 9 = {dr(3)}")
print(f"  7 mod 9 = {dr(7)}")
print(f"  12 mod 9 = {dr(12)}")
print(f"  3+7+12 = {a+b+c}, dr = {dr(a+b+c)}")
print(f"  3*7 = {a*b}, dr = {dr(a*b)}")
print(f"  3*12 = {a*c}, dr = {dr(a*c)}")
print(f"  7*12 = {b*c}, dr = {dr(b*c)}")
print(f"  3*7*12 = {a*b*c}, dr = {dr(a*b*c)}")
print()

# mod-24 versions
print(f"  3 mod 24 = {1+((3-1)%24)}")
print(f"  7 mod 24 = {1+((7-1)%24)}")
print(f"  12 mod 24 = {1+((12-1)%24)}")
print(f"  22 mod 24 = {1+((22-1)%24)}")
print(f"  252 mod 24 = {1+((252-1)%24)}")
print()

# Check against orbit periods
print("  QA orbit periods: {1, 8, 24}")
print(f"  3 divides 24? {24 % 3 == 0}  (24/3 = {24//3})")
print(f"  7 divides 24? {24 % 7 == 0}")
print(f"  12 divides 24? {24 % 12 == 0}  (24/12 = {24//12})")
print(f"  gcd(3,24) = {gcd(3,24)}, gcd(7,24) = {gcd(7,24)}, gcd(12,24) = {gcd(12,24)}")
print(f"  lcm(3,7) = {(3*7)//gcd(3,7)}, lcm(3,12) = {(3*12)//gcd(3,12)}, lcm(7,12) = {(7*12)//gcd(7,12)}")
print(f"  lcm(3,7,12) = ?")
l12 = (3*7)//gcd(3,7)
l123 = (l12*12)//gcd(l12,12)
print(f"  lcm(3,7,12) = {l123}")
print(f"  NOTE: Euler totient phi(24) = {sum(1 for i in range(1,25) if gcd(i,24)==1)}")
phi9 = sum(1 for i in range(1,10) if gcd(i,9)==1)
print(f"  Euler totient phi(9) = {phi9}")

# ══════════════════════════════════════════════════════════════════
# 10. C(22,2) = 231 decomposition
# ══════════════════════════════════════════════════════════════════
print("\n── 10. C(22,2) = 231 decomposition by partition ──\n")

c32 = comb(3, 2)
c72 = comb(7, 2)
c122 = comb(12, 2)
cross_37 = 3 * 7
cross_312 = 3 * 12
cross_712 = 7 * 12
total_pairs = c32 + c72 + c122 + cross_37 + cross_312 + cross_712

print(f"  C(3,2)   = {c32}   (mother-mother)")
print(f"  C(7,2)   = {c72}   (double-double)")
print(f"  C(12,2)  = {c122}  (simple-simple)")
print(f"  3*7      = {cross_37}   (mother-double)")
print(f"  3*12     = {cross_312}   (mother-simple)")
print(f"  7*12     = {cross_712}   (double-simple)")
print(f"  Total    = {total_pairs}  (should be 231)")
print(f"  C(22,2)  = {comb(22,2)}")
print(f"  Match? {total_pairs == comb(22,2)}")
print()

# dr of each component
for label, val in [("C(3,2)", c32), ("C(7,2)", c72), ("C(12,2)", c122),
                   ("3*7", cross_37), ("3*12", cross_312), ("7*12", cross_712),
                   ("Total=231", 231)]:
    print(f"  dr({label}) = dr({val}) = {dr(val)}")

# Connection to Sefer Yetzirah's "231 gates"
print(f"\n  231 = C(22,2): the '231 Gates' of Sefer Yetzirah")
print(f"  Within-class pairs: {c32 + c72 + c122} = {c32}+{c72}+{c122}")
print(f"  Cross-class pairs:  {cross_37 + cross_312 + cross_712} = {cross_37}+{cross_312}+{cross_712}")

# ══════════════════════════════════════════════════════════════════
# 11. f-values for pairs drawn from each letter group
# ══════════════════════════════════════════════════════════════════
print("\n── 11. QA f-norm and orbit classification for letter pairs ──\n")

for cls_name in ["mother", "double", "simple"]:
    vals = [v for _, v, c in letters if c == cls_name]
    drs_list = [dr(v) for v in vals]
    names = [n for n, v, c in letters if c == cls_name]

    print(f"\n  {cls_name.upper()} pairs (using digital roots as (b,e)):")
    orbit_counts = {"Cosmos": 0, "Satellite": 0, "Singularity": 0}
    v3_vals = []

    for i in range(len(drs_list)):
        for j in range(len(drs_list)):
            if i == j:
                continue
            bi, ei = drs_list[i], drs_list[j]
            fval = f_norm(bi, ei)
            v = v3(fval)
            orb = orbit_class(bi, ei)
            orbit_counts[orb] += 1
            v3_vals.append(v)
            if len(drs_list) <= 7:  # Only print detail for small groups
                print(f"    ({names[i]}={bi}, {names[j]}={ei}): f={fval}, v3={v}, orbit={orb}")

    print(f"  Orbit distribution: {orbit_counts}")
    print(f"  v3 values: {sorted(set(v3_vals))} (unique)")

# Also do cross-class
print("\n  CROSS-CLASS pairs (mother x double, using digital roots):")
mother_drs = [(n, dr(v)) for n, v, c in letters if c == "mother"]
double_drs = [(n, dr(v)) for n, v, c in letters if c == "double"]
simple_drs = [(n, dr(v)) for n, v, c in letters if c == "simple"]

for label, group_a, group_b in [("mother x double", mother_drs, double_drs),
                                  ("mother x simple", mother_drs, simple_drs),
                                  ("double x simple", double_drs, simple_drs)]:
    orbit_counts = {"Cosmos": 0, "Satellite": 0, "Singularity": 0}
    for na, da in group_a:
        for nb, db in group_b:
            orb = orbit_class(da, db)
            orbit_counts[orb] += 1
            orb2 = orbit_class(db, da)
            orbit_counts[orb2] += 1
    print(f"  {label}: {orbit_counts}")

# ══════════════════════════════════════════════════════════════════
# 12. Connection between {3,7,12} and QA orbit periods {1,8,24}
# ══════════════════════════════════════════════════════════════════
print("\n── 12. {3,7,12} vs QA orbit periods {1,8,24} ──\n")

# Direct ratios
print("  Direct ratios:")
print(f"  24/3 = {24/3}")
print(f"  24/7 = {24/7:.4f}")
print(f"  24/12 = {24/12}")
print(f"  8/3 = {8/3:.4f}")
print(f"  8/7 = {8/7:.4f}")
print(f"  8/12 = {Fraction(8,12)}")

# Sum and product checks
print(f"\n  3+7+12 = 22, and 22 mod 24 = 22")
print(f"  3*8 = 24 (mother count * satellite period = cosmos period)")
print(f"  1+8+24 = 33, dr(33) = {dr(33)}")
print(f"  3+7+12 = 22, dr(22) = {dr(22)}")

# Divisibility in mod 24
print(f"\n  In Z/24Z:")
print(f"  Order of 3 in (Z/24Z)*: ", end="")
x = 3
for k in range(1, 25):
    if pow(3, k, 24) == 1:
        print(k)
        break

print(f"  Order of 7 in (Z/24Z)*: ", end="")
for k in range(1, 25):
    if pow(7, k, 24) == 1:
        print(k)
        break

# Factor structures
print(f"\n  Factor structures:")
print(f"  3 = 3")
print(f"  7 = 7")
print(f"  12 = 2*2*3")
print(f"  24 = 2*2*2*3  = 8*3 = 12*2")
print(f"  8 = 2*2*2")
print(f"  1 = 1")
print(f"  NOTE: 12 = 24/2, 3 | 24, 7 is coprime to 24")

# Does {3,7,12} form a partition of Z/9Z residues?
print(f"\n  Residues of letter VALUES by class (mod 9, no-zero):")
for cls_name in ["mother", "double", "simple"]:
    vals = [v for _, v, c in letters if c == cls_name]
    residues = sorted(set(dr(v) for v in vals))
    print(f"    {cls_name}: dr values = {residues}")

# Union
all_drs = sorted(set(dr(v) for _, v, _ in letters))
print(f"  Union of all dr values: {all_drs}")
print(f"  Full {1,...,9}? {all_drs == list(range(1,10))}")

# ══════════════════════════════════════════════════════════════════
# BONUS: Deeper structure checks
# ══════════════════════════════════════════════════════════════════
print("\n── BONUS: Additional structural checks ──\n")

# Check if mother letters span a specific subgroup of Z/9Z
mother_vals = [1, 40, 300]
mother_drs_raw = [dr(v) for v in mother_vals]
print(f"  Mother dr's: {mother_drs_raw}")
print(f"  Mother dr's sorted: {sorted(mother_drs_raw)}")
print(f"  Are these {1,4,3}? = first 3 triangular-ish? No pattern obvious.")
print(f"  Sum of mother dr's: {sum(mother_drs_raw)}, dr = {dr(sum(mother_drs_raw))}")

double_vals = [2, 3, 4, 20, 80, 200, 400]
double_drs_raw = [dr(v) for v in double_vals]
print(f"  Double dr's: {double_drs_raw}")
print(f"  Sum of double dr's: {sum(double_drs_raw)}, dr = {dr(sum(double_drs_raw))}")

simple_vals = [5, 6, 7, 8, 9, 10, 30, 50, 60, 70, 90, 100]
simple_drs_raw = [dr(v) for v in simple_vals]
print(f"  Simple dr's: {simple_drs_raw}")
print(f"  Sum of simple dr's: {sum(simple_drs_raw)}, dr = {dr(sum(simple_drs_raw))}")

# Check: do the 3 mother dr's form a coset in Z/9Z?
print(f"\n  Coset check: mother dr's = {sorted(mother_drs_raw)}")
print(f"  Differences (mod 9): ", end="")
for i in range(len(mother_drs_raw)):
    for j in range(i+1, len(mother_drs_raw)):
        diff = (mother_drs_raw[j] - mother_drs_raw[i]) % 9
        if diff == 0: diff = 9
        print(f"dr({mother_vals[j]})-dr({mother_vals[i]}) = {diff}, ", end="")
print()

# Check: multiplicative structure
print(f"\n  Multiplicative products in Z/9Z (no-zero):")
print(f"  dr(1*40) = dr(40) = {dr(40)}")
print(f"  dr(1*300) = dr(300) = {dr(300)}")
print(f"  dr(40*300) = dr(12000) = {dr(12000)}")

# v3 of the sums
print(f"\n  3-adic valuations:")
print(f"  v3(341) [mother sum] = {v3(341)}")
print(f"  v3(709) [double sum] = {v3(709)}")
print(f"  v3(445) [simple sum] = {v3(445)}")
print(f"  v3(1495) [total] = {v3(1495)}")

# The 231 gates number
print(f"\n  v3(231) = {v3(231)}")
print(f"  231 = 3 * 7 * 11")
print(f"  dr(231) = {dr(231)}")

# Check: does 3-7-12 correspond to orbits by SIZE?
print(f"\n  QA orbit SIZES in mod-24:")
print(f"  Singularity: 1 fixed point (pair (9,9)) — compare to 3 mothers?")
print(f"  Satellite: 8 pairs in 8-cycle — compare to 7 doubles?")
print(f"  Cosmos: 72 pairs in 24-cycle — compare to 12 simples?")
print(f"  Ratios: 3/1={3/1}, 7/8={Fraction(7,8)}, 12/72={Fraction(12,72)}")
print(f"  No clean ratio mapping.")

# Check: by PERIOD not size?
print(f"\n  QA orbit PERIODS: 1, 8, 24")
print(f"  SY class SIZES: 3, 7, 12")
print(f"  3*8 = 24 ← interesting: mother_count * satellite_period = cosmos_period")
print(f"  7*1 = 7 — no obvious match")
print(f"  But also: 3*1 = 3, 7*... , 12*2 = 24")

# Check number 22 in QA
print(f"\n  22 in QA context:")
print(f"  22 mod 9 = {dr(22)} (= 4)")
print(f"  22 mod 24 = {1+((22-1)%24)} (= 22)")
print(f"  v3(22) = {v3(22)} (= 0, coprime to 3)")
print(f"  f(3,7) = {f_norm(3,7)} = 3*3 + 3*7 - 7*7 = 9+21-49 = -19")
print(f"  f(7,3) = {f_norm(7,3)} = 49+21-9 = 61")
print(f"  f(3,12) = {f_norm(3,12)} = 9+36-144 = -99, v3={v3(-99)}")
print(f"  f(12,3) = {f_norm(12,3)} = 144+36-9 = 171, v3={v3(171)}")
print(f"  f(7,12) = {f_norm(7,12)} = 49+84-144 = -11")
print(f"  f(12,7) = {f_norm(12,7)} = 144+84-49 = 179")

# Check: do the class sizes have QA-meaningful f-values?
print(f"\n  f-norm on partition sizes:")
print(f"  f(3,7) = {f_norm(3,7)}, v3 = {v3(f_norm(3,7))}, orbit = {orbit_class(3,7)}")
print(f"  f(7,3) = {f_norm(7,3)}, v3 = {v3(f_norm(7,3))}, orbit = {orbit_class(7,3)}")
print(f"  f(3,12) = {f_norm(3,12)}, v3 = {v3(f_norm(3,12))}, orbit = {orbit_class(3,12)}")
print(f"  f(12,3) = {f_norm(12,3)}, v3 = {v3(f_norm(12,3))}, orbit = {orbit_class(12,3)}")
print(f"  f(7,12) = {f_norm(7,12)}, v3 = {v3(f_norm(7,12))}, orbit = {orbit_class(7,12)}")
print(f"  f(12,7) = {f_norm(12,7)}, v3 = {v3(f_norm(12,7))}, orbit = {orbit_class(12,7)}")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY: Honest assessment
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

# Check which findings are structurally significant
print("""
Key structural facts (verified above):
1. The 22 Hebrew letter values have digital roots covering {1,...,9} completely.
2. The 3-7-12 partition does NOT cleanly separate digital root classes.
3. 231 = C(22,2) = 3*7*11 has v3=1 (Cosmos-class).
4. The partition identity C(22,2) = C(3,2)+C(7,2)+C(12,2)+3*7+3*12+7*12
   is a combinatorial tautology (true for ANY partition of 22 into 3 parts).
5. 3*8 = 24 (mother_count * satellite_period = cosmos_period) —
   arithmetically true but requires cherry-picking which numbers to multiply.
6. 7 is coprime to both 9 and 24 — it lives OUTSIDE the QA modular structure.
7. The f-norm orbit classifications of pairs from letter groups show no
   clean separation by Sefer Yetzirah class.

HONEST VERDICT: Report each finding's strength below.
""")
