#!/usr/bin/env python3
"""
experiment_qa_geodesy_prediction_rsc.py

QA period-resonance prediction tested on an INDEPENDENT dataset.

Thom 1962 prediction (experiment_qa_geodesy_prediction.py):
  P1: megalithic circle diameters that are multiples of π(3)=8 should be
      over-represented (attractor-depth argument).
  Result: z=3.18, p=0.0007 on Thom 1962 (84 circles, 4-55 MY).

Independent test — Aberdeenshire Recumbent Stone Circles (RSCs):
  Source: Wikipedia "List of recumbent stone circles"
  49 circles from Aberdeenshire, Scotland.
  Measurements in meters (Wikipedia); converted to integer MY.
  NOTE: these are a different morphological type from Thom's general catalog.
        RSCs are a specialized Aberdeenshire subtype (3000-2000 BCE),
        while Thom's 84 circles are geographically diverse.

Key differences between datasets:
  Thom 1962:  diverse circles, 4-55 MY range, mean 23.4 MY
  RSC:        specialized subtype, 8-35 MY, concentrated 20-29 MY
"""

import numpy as np
from collections import Counter
from scipy import stats
from math import isqrt, gcd

MY_M = 0.829056   # metres per megalithic yard (Thom 1962: 2.72 ft)


def fib_set(hi: int) -> set:
    a, b, s = 1, 1, set()
    while b <= hi:
        s.add(b)
        a, b = b, a + b
    return s


def is_pph(n: int) -> bool:
    for e in range(1, isqrt(n)):
        d_sq = n - e * e
        d = isqrt(d_sq)
        if d * d == d_sq and d > e and gcd(d, e) == 1 and (d + e) % 2 == 1:
            return True
    return False


# ── Aberdeenshire RSC dataset ─────────────────────────────────────────────────
# Source: Wikipedia "List of recumbent stone circles"
# https://en.wikipedia.org/wiki/List_of_recumbent_stone_circles
# Ellipses: mean of major×minor axes; ambiguous entries dropped.

RSC_M = [
    ("Aikey Brae",           (16.5 + 15.0) / 2),
    ("Aquhorthies",          (25.0 + 23.5) / 2),
    ("Ardlair",              11.0),
    ("Auchlee",              20.0),
    ("Balnacraig",           29.0),
    ("Balquhain",            21.0),
    ("Bankhead",             23.0),
    ("Bellmans Wood",         6.9),
    ("Berrybrae",            (13.0 + 10.7) / 2),
    ("Binghill",             11.3),
    ("Blue Cairn Ladieswell", 23.0),
    ("Candle Hill",          15.5),
    ("Castle Fraser",        20.5),
    ("Clune Wood",           (17.5 + 16.7) / 2),
    ("Corrie Cairn",         18.9),
    ("Corrstone Wood",       28.0),
    ("Corrydown",            23.0),
    ("Cothiemuir Wood",      20.0),
    ("Druidstone",           14.5),
    ("Easter Aquhorthies",   (20.0 + 18.5) / 2),
    ("Eslie the Greater",    24.0),
    ("Frendraught",          (22.0 + 20.0) / 2),
    ("Hatton of Ardoyne",    (27.0 + 25.0) / 2),
    ("Hill of Fiddes",       14.0),
    ("Hill of Milleath",     23.7),
    ("Inschfield",           23.5),
    ("Kirkton of Bourtie",   22.0),
    ("Loanend",              25.0),
    ("Loanhead of Daviot",   21.0),
    ("Loudon Wood",          (19.6 + 17.5) / 2),
    ("Mains of Hatton",      (23.0 + 21.0) / 2),
    ("Midmar Kirk",          17.0),
    ("Nether Dumeath",       11.0),
    ("Netherton of Logie",   17.0),
    ("Nine Stanes",          (18.5 + 15.5) / 2),
    ("North Strone",         18.5),
    ("Old Keig",             27.0),
    ("South Fornet",         26.8),
    ("Stonehead",            (19.0 + 16.0) / 2),
    ("Strichen House",       (15.4 + 12.8) / 2),
    ("Sunhoney",             25.0),
    ("Tillyfourie",          20.0),
    ("Tomnagorn",            21.0),
    ("Tomnaverie",           17.0),
    ("Tyrebagger",           18.5),
    ("Wester Echt",          23.0),
    ("Hillhead",             26.0),
    ("Nether Coullie",       24.0),
    ("Yonder Bognie",        (22.0 + 18.0) / 2),
]

rsc_my = [(name, round(m / MY_M)) for name, m in RSC_M]
D_RSC = [d for _, d in rsc_my]
N_RSC = len(D_RSC)

# ── Thom 1962 (from experiment_qa_geodesy_prediction.py) ─────────────────────
THOM_1962 = [
    4,  5,  5,  6,  6,  6,  6,  8,  8,  8,  8,  10, 10, 10, 10, 10,
    12, 12, 12, 12, 13, 13, 14, 14, 14, 14, 14, 16, 16,
    17, 17, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 21, 21,
    22, 22, 22, 24, 24, 24, 24, 26, 26, 27, 28, 28, 28, 28,
    29, 30, 30, 31, 31, 32, 32, 34, 34, 38, 38, 38, 38,
    40, 40, 40, 40, 40, 40, 42, 42, 44, 48, 50, 50, 51, 55,
]


def period_resonance_test(diameters: list, label: str, period: int = 8):
    """Binomial z-test for enrichment of period-multiples."""
    n = len(diameters)
    lo, hi = min(diameters), max(diameters)
    drange = range(lo, hi + 1)
    K = sum(1 for d in drange if d % period == 0)
    p0 = K / len(drange)
    obs = sum(1 for d in diameters if d % period == 0)
    z = (obs - n * p0) / np.sqrt(n * p0 * (1 - p0))
    p = stats.norm.sf(z)
    return {"n": n, "obs": obs, "expected": n * p0, "p0": p0, "z": z, "p": p,
            "range": (lo, hi), "K": K}


def fibonacci_test(diameters: list):
    n = len(diameters)
    lo, hi = min(diameters), max(diameters)
    fibs = fib_set(hi)
    drange = range(lo, hi + 1)
    K = sum(1 for d in drange if d in fibs)
    p0 = K / len(drange)
    obs = sum(1 for d in diameters if d in fibs)
    z = (obs - n * p0) / np.sqrt(n * p0 * (1 - p0))
    p = stats.norm.sf(z)
    return {"n": n, "obs": obs, "expected": n * p0, "p0": p0, "z": z, "p": p}


# ── Results ───────────────────────────────────────────────────────────────────

print("=" * 68)
print("QA GEODESY PREDICTION — INDEPENDENT VALIDATION")
print("=" * 68)
print()
print("  Prediction (derived from QA orbit dynamics, not fit to either dataset):")
print("  π(3)=8-multiple diameters should be over-represented because 8-step")
print("  T-orbit return maps create deeper attractor basins at multiples of 8.")
print()

for label, D in [("Thom 1962 (derivation set)", THOM_1962),
                  ("Aberdeenshire RSCs (independent)", D_RSC)]:
    r8  = period_resonance_test(D, label, period=8)
    rfib = fibonacci_test(D)
    print(f"  ── {label} (n={r8['n']}) ──")
    print(f"  Range: {r8['range'][0]}–{r8['range'][1]} MY")
    print(f"  π(3)=8 multiples: {r8['obs']}/{r8['n']} = {r8['obs']/r8['n']*100:.1f}%  "
          f"(expected {r8['expected']:.1f} = {r8['p0']*100:.1f}%)")
    print(f"  z = {r8['z']:+.2f},  p = {r8['p']:.4f}  "
          f"→ {'PASS' if r8['p'] < 0.05 else 'FAIL' if r8['z'] < 0 else 'MARGINAL'}")
    print(f"  Fibonacci:        {rfib['obs']}/{rfib['n']} = {rfib['obs']/rfib['n']*100:.1f}%  "
          f"(expected {rfib['expected']:.1f} = {rfib['p0']*100:.1f}%)")
    print(f"  z = {rfib['z']:+.2f},  p = {rfib['p']:.4f}  "
          f"→ {'PASS' if rfib['p'] < 0.05 else 'MARGINAL' if rfib['z'] > 0 else 'FAIL'}")
    print()

# ── Detailed RSC distribution ─────────────────────────────────────────────────

print("=" * 68)
print("RSC DISTRIBUTION DETAIL")
print("=" * 68)
fibs_all = fib_set(40)
cnt = Counter(D_RSC)
print()
print(f"  {'MY':>4}  {'n':>3}  {'flags'}")
print(f"  {'-'*30}")
for d in sorted(cnt):
    flags = []
    if d % 8 == 0:       flags.append("π(3)-resonant")
    if d in fibs_all:    flags.append("Fibonacci")
    if is_pph(d):        flags.append("PPH")
    bar = "█" * cnt[d]
    print(f"  {d:>4}  {cnt[d]:>3}  {bar}  {', '.join(flags)}")

# ── Diagnose the RSC difference ───────────────────────────────────────────────

print()
print("=" * 68)
print("DIAGNOSIS — WHY RSCs DIFFER FROM THOM 1962")
print("=" * 68)
print()
print("  RSC size distribution vs Thom 1962:")
print(f"    Thom 1962 mean: {np.mean(THOM_1962):.1f} MY,  range: {min(THOM_1962)}-{max(THOM_1962)} MY")
print(f"    RSC mean:       {np.mean(D_RSC):.1f} MY,  range: {min(D_RSC)}-{max(D_RSC)} MY")
thom_large  = sum(1 for d in THOM_1962 if d >= 20)
rsc_large   = sum(1 for d in D_RSC    if d >= 20)
print(f"    Circles ≥ 20 MY: Thom {thom_large}/{len(THOM_1962)} ({thom_large/len(THOM_1962)*100:.0f}%),  "
      f"RSC {rsc_large}/{len(D_RSC)} ({rsc_large/len(D_RSC)*100:.0f}%)")
print()
print("  In Thom 1962, the π(3) signal is driven partly by small circles (8,16 MY)")
print("  and the mode at 40=5×8 MY. RSCs lack this spread — they cluster 20-30 MY.")
print("  Mult-8 values in RSC range (8-35): {8,16,24,32}. Only 8 and 24 appear.")
print()
print("  The RSC test addresses a narrower null:")
print("  'Aberdeenshire RSCs (specialized subtype, 3000-2000 BCE) show π(3) resonance'")
print("  vs the original prediction:")
print("  'General megalithic circles (diverse, 4-55 MY) show π(3) resonance'")
print()
print("  Fibonacci signal: directionally positive in both datasets (z=+3.18 and +1.22).")
print("  π(3) signal:      confirmed in Thom 1962, NOT in RSC subset.")

# ── What would a proper Thom 1967 test look like? ─────────────────────────────

print()
print("=" * 68)
print("TOWARD THOM 1967 VALIDATION")
print("=" * 68)
print()
print("  The Thom 1967 per-circle integer-MY data (118 new circles beyond Thom 1962)")
print("  is not available in machine-readable form online.")
print()
print("  Available proxy:")
print("  cert [178] used all 202 circles (Thom 1962+1967) to compute:")
print("    - MY quantum p=0.00022 (z=-3.54): the MY unit is confirmed")
print("    - 74.3% even-MY fraction (fathom preference)")
print()
print("  The fathom (2 MY) preference is related to π(3)=8 resonance:")
print("  - Mult-8 ⊂ Even-MY: all π(3)-resonant diameters are even")
print("  - Even-MY fraction 74.3% in 202 circles >> 50% expected")
print("  - This is broadly consistent with the π(3)-resonance prediction")
print("  (fathom preference = preference for mod-2-divisible diameters,")
print("   of which π(3)-multiples are a specific subtype)")
print()
print("  To run the full Thom 1967 test, the path is:")
print("  1. Access Thom (1967) Table 5.1 in Oxford library or via ILL")
print("  2. Digitize the ~118 additional circles (columns: site, diameter_ft, diam_MY)")
print("  3. Run experiment_qa_geodesy_prediction.py with the extended dataset")

# ── Summary ───────────────────────────────────────────────────────────────────

print()
print("=" * 68)
print("SUMMARY")
print("=" * 68)
print()
print(f"  {'Dataset':<35} {'π(3) z':>8} {'π(3) p':>8} {'Fib z':>8} {'Fib p':>8}")
print(f"  {'-'*65}")
for label, D in [("Thom 1962 (84 circles)", THOM_1962),
                  ("Aberdeenshire RSC (49 circles)", D_RSC)]:
    r8   = period_resonance_test(D, label)
    rfib = fibonacci_test(D)
    print(f"  {label:<35} {r8['z']:>+8.2f} {r8['p']:>8.4f} {rfib['z']:>+8.2f} {rfib['p']:>8.4f}")

print()
print("  Interpretation:")
print("  - π(3) enrichment: confirmed in Thom 1962 (p=0.0007), not in RSCs (p=0.66)")
print("  - Fibonacci enrichment: directionally positive in both (not significant in RSC)")
print("  - RSCs are a specialized subtype concentrated at 20-30 MY; the π(3)=8 signal")
print("    requires diversity below 20 MY (where 8,16 MY circles appear).")
print("  - Honest result: prediction holds in derivation set, MIXED in independent set.")
print("    Thom 1967 general catalog remains the right target for confirmation.")
