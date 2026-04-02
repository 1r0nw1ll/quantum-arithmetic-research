#!/usr/bin/env python3
"""
qa_solar_system_deep.py — Deep QA analysis of the solar system.

For each solar-system body we find the best-matching QA quantum number
(b, e, d, a) where d = b + e, a = b + 2*e, gcd(b, e) = 1, and the
QA eccentricity e/d approximates the body's observed orbital eccentricity.

We then build a prime-harmonic network (Law of Harmonics), verify
mean-motion resonances in QN structure, and test the Titius-Bode law.

QA axioms enforced:
  A1  States in {1,...,N}          (no zero)
  A2  d = b+e, a = b+2e           (derived coords)
  S1  b*b not b*b with pow        (no libm pow)
  S2  b, e are int                 (no float state)

Author: Will Dale (analysis), Claude (code)
"""

QA_COMPLIANCE = "observer=eccentricity_matching, state_alphabet=primitive_QN_tuple"

import math
import itertools
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── seed ────────────────────────────────────────────────────────────────
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# 1. COMPREHENSIVE BODY CATALOGUE
# ═══════════════════════════════════════════════════════════════════════
# Format: (name, eccentricity, category, parent, semi_major_au_or_km)
# semi_major: AU for planets/dwarfs/comets/asteroids, km for moons
# parent: None for heliocentric, planet name for moons

BODIES = [
    # ── Planets (heliocentric) ───────────────────────────────────────
    ("Mercury",    0.20563,  "planet",   None,      0.387),
    ("Venus",      0.00677,  "planet",   None,      0.723),
    ("Earth",      0.01671,  "planet",   None,      1.000),
    ("Mars",       0.09339,  "planet",   None,      1.524),
    ("Jupiter",    0.04839,  "planet",   None,      5.203),
    ("Saturn",     0.05415,  "planet",   None,      9.537),
    ("Uranus",     0.04717,  "planet",   None,     19.191),
    ("Neptune",    0.00859,  "planet",   None,     30.069),

    # ── Earth's Moon ─────────────────────────────────────────────────
    ("Moon",       0.0549,   "moon",     "Earth",   384400),

    # ── Mars moons ───────────────────────────────────────────────────
    ("Phobos",     0.0151,   "moon",     "Mars",    9376),
    ("Deimos",     0.0002,   "moon",     "Mars",    23463),

    # ── Jupiter moons ────────────────────────────────────────────────
    ("Io",         0.0041,   "moon",     "Jupiter", 421700),
    ("Europa",     0.0094,   "moon",     "Jupiter", 671034),
    ("Ganymede",   0.0011,   "moon",     "Jupiter", 1070412),
    ("Callisto",   0.0074,   "moon",     "Jupiter", 1882709),
    ("Amalthea",   0.0032,   "moon",     "Jupiter", 181366),

    # ── Saturn moons ─────────────────────────────────────────────────
    ("Mimas",      0.0196,   "moon",     "Saturn",  185539),
    ("Enceladus",  0.0047,   "moon",     "Saturn",  238042),
    ("Tethys",     0.0001,   "moon",     "Saturn",  294619),
    ("Dione",      0.0022,   "moon",     "Saturn",  377396),
    ("Rhea",       0.0013,   "moon",     "Saturn",  527108),
    ("Titan",      0.0288,   "moon",     "Saturn",  1221870),
    ("Iapetus",    0.0276,   "moon",     "Saturn",  3560820),

    # ── Uranus moons ─────────────────────────────────────────────────
    ("Miranda",    0.0013,   "moon",     "Uranus",  129390),
    ("Ariel",      0.0012,   "moon",     "Uranus",  190900),
    ("Umbriel",    0.0039,   "moon",     "Uranus",  266300),
    ("Titania",    0.0011,   "moon",     "Uranus",  436300),
    ("Oberon",     0.0014,   "moon",     "Uranus",  583500),

    # ── Neptune moons ────────────────────────────────────────────────
    ("Triton",     0.000016, "moon",     "Neptune", 354759),
    ("Nereid",     0.7507,   "moon",     "Neptune", 5513818),

    # ── Pluto system ─────────────────────────────────────────────────
    ("Charon",     0.0002,   "moon",     "Pluto",   17536),

    # ── Dwarf planets (heliocentric) ─────────────────────────────────
    ("Pluto",      0.2488,   "dwarf",    None,     39.482),
    ("Ceres",      0.0758,   "dwarf",    None,      2.767),
    ("Eris",       0.4407,   "dwarf",    None,     67.668),
    ("Haumea",     0.1912,   "dwarf",    None,     43.335),
    ("Makemake",   0.1559,   "dwarf",    None,     45.792),
    ("Sedna",      0.8496,   "dwarf",    None,    506.8),

    # ── Comets ───────────────────────────────────────────────────────
    ("Halley",             0.96714, "comet", None, 17.834),
    ("Hale-Bopp",          0.99507, "comet", None, 186.0),
    ("Encke",              0.8483,  "comet", None, 2.215),
    ("Hyakutake",          0.99990, "comet", None, 1700.0),
    ("67P/C-G",            0.6405,  "comet", None, 3.463),

    # ── Asteroids ────────────────────────────────────────────────────
    ("Vesta",      0.0887,   "asteroid", None,  2.362),
    ("Pallas",     0.2313,   "asteroid", None,  2.773),
    ("Eros",       0.2229,   "asteroid", None,  1.458),
    ("Bennu",      0.2037,   "asteroid", None,  1.126),
    ("Itokawa",    0.2802,   "asteroid", None,  1.324),
]


# ═══════════════════════════════════════════════════════════════════════
# 2. QN FINDER — eccentricity = e / d  where d = b + e
# ═══════════════════════════════════════════════════════════════════════

def _best_rational_approx(target, max_denom):
    """Stern-Brocot / continued-fraction best rational approximation e/d <= max_denom.

    Returns list of (e_val, d_val) candidates (best convergents + semi-convergents).
    target must be in (0, 1).
    """
    # Collect continued-fraction convergents
    candidates = []
    # Mediant / Stern-Brocot walk
    lo_n, lo_d = 0, 1   # 0/1
    hi_n, hi_d = 1, 1   # 1/1
    candidates.append((lo_n, lo_d))
    candidates.append((hi_n, hi_d))

    for _ in range(200):
        med_n = lo_n + hi_n
        med_d = lo_d + hi_d
        if med_d > max_denom:
            break
        candidates.append((med_n, med_d))
        med_val = med_n / med_d
        if med_val < target:
            lo_n, lo_d = med_n, med_d
        elif med_val > target:
            hi_n, hi_d = med_n, med_d
        else:
            break

    # Also add continued-fraction convergents
    a_list = []
    x = target
    for _ in range(30):
        a_i = int(x)
        a_list.append(a_i)
        frac = x - a_i
        if frac < 1e-14:
            break
        x = 1.0 / frac
        if x > 1e12:
            break

    # Build convergents from CF
    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0
    for a_i in a_list:
        p_next = a_i * p_curr + p_prev
        q_next = a_i * q_curr + q_prev
        if q_next > max_denom:
            # Try partial: largest k s.t. k*q_curr + q_prev <= max_denom
            if q_curr > 0:
                k_max = (max_denom - q_prev) // q_curr
                for k in range(max(1, k_max - 1), k_max + 2):
                    q_t = k * q_curr + q_prev
                    p_t = k * p_curr + p_prev
                    if 0 < q_t <= max_denom and p_t > 0:
                        candidates.append((p_t, q_t))
            break
        if q_next > 0 and p_next > 0:
            candidates.append((p_next, q_next))
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next

    return candidates


def find_qn(ecc_target, max_d=500):
    """Return (b, e, d, a, ecc_qa, error) for best primitive QN match.

    Uses continued-fraction rational approximation for speed.

    QA rules:
        b, e >= 1          (A1 no-zero)
        d = b + e           (A2 derived)
        a = b + 2*e         (A2 derived)
        gcd(b, e) = 1       (primitive)
        eccentricity = e / d
    """
    # For very small eccentricities, extend search range
    if ecc_target < 0.005:
        max_d = max(max_d, 5000)
    if ecc_target < 0.001:
        max_d = max(max_d, 10000)
    # For eccentricities very close to 1, also extend (b will be small, d large)
    if ecc_target > 0.995:
        max_d = max(max_d, 10000)

    # ecc = e / d  =>  e / (b + e) = ecc  =>  e / b = ecc / (1 - ecc)
    # So we want rational approx of ecc as e/d, where d = b + e, b = d - e >= 1
    # Equivalently: find e, d with e/d ~ ecc, gcd(e, d-e) = 1 (primitive), d-e >= 1

    best = None
    best_err = float("inf")

    # Strategy: get best rational approximations of ecc as p/q
    # Then e = p, d = q  =>  b = q - p >= 1  =>  need q > p
    # gcd(b, e) = gcd(q-p, p) = gcd(q, p)  =>  need gcd(p, q) = 1

    if ecc_target <= 0 or ecc_target >= 1:
        # Edge case: ecc ~ 0 or ~ 1
        if ecc_target <= 0:
            return (max_d - 1, 1, max_d, max_d + 1, 1.0 / max_d, abs(1.0 / max_d - ecc_target))
        # ecc ~ 1
        ecc_target = min(ecc_target, 0.9999999)

    candidates = _best_rational_approx(ecc_target, max_d)

    # Also do a local brute-force around the best CF approximation for refinement
    # but only for small denominators
    brute_max = min(max_d, 500)
    for enum in range(1, brute_max):
        # denom ~ enum / ecc_target
        if ecc_target > 0:
            denom_approx = round(enum / ecc_target)
            for denom in range(max(enum + 1, denom_approx - 2), min(brute_max + 1, denom_approx + 3)):
                if denom > brute_max:
                    break
                candidates.append((enum, denom))

    for enum, denom in candidates:
        if enum < 1 or denom < 2:
            continue
        b_val = denom - enum        # A2: d = b + e  =>  b = d - e
        if b_val < 1:
            continue                # A1: b >= 1
        if denom > max_d:
            continue
        e_val = enum
        d_val = b_val + e_val       # A2: d = b + e (derived)
        if math.gcd(b_val, e_val) != 1:
            continue                # primitive
        a_val = b_val + 2 * e_val   # A2: a = b + 2e (derived)
        ecc_qa = e_val / d_val      # observer projection (float)
        err = abs(ecc_qa - ecc_target)
        if err < best_err:
            best_err = err
            best = (b_val, e_val, d_val, a_val, ecc_qa, err)
            if err < 1e-12:
                return best

    return best


def prime_factors(n):
    """Return set of prime factors of |n|."""
    n = abs(n)
    if n < 2:
        return set()
    factors = set()
    d = 2
    while d * d <= n:                       # S1: d*d not d**2
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def qn_product(b, e, d, a):
    """Product of QN tuple elements."""
    return b * e * d * a


def all_prime_factors_of_qn(b, e, d, a):
    """Union of prime factors of all four QN elements."""
    return prime_factors(b) | prime_factors(e) | prime_factors(d) | prime_factors(a)


# ═══════════════════════════════════════════════════════════════════════
# 3. ASSIGN QNs TO ALL BODIES
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  QA SOLAR SYSTEM DEEP ANALYSIS")
print("  Quantum Number assignment + Prime Harmonic Network")
print("=" * 78)
print()

results = {}  # name -> dict with all info

print(f"{'Body':<20} {'Ecc_obs':>10} {'(b,e,d,a)':<20} {'Ecc_QA':>10} {'Error':>12} {'Primes'}")
print("-" * 100)

for name, ecc, cat, parent, sma in BODIES:
    qn = find_qn(ecc)
    b, e, d, a, ecc_qa, err = qn
    primes = all_prime_factors_of_qn(b, e, d, a)
    prod = qn_product(b, e, d, a)

    results[name] = {
        "ecc": ecc,
        "cat": cat,
        "parent": parent,
        "sma": sma,
        "b": b, "e": e, "d": d, "a": a,
        "ecc_qa": ecc_qa,
        "err": err,
        "primes": primes,
        "product": prod,
    }

    primes_str = ",".join(str(p) for p in sorted(primes))
    print(f"{name:<20} {ecc:>10.6f} ({b},{e},{d},{a}){'':<{max(0, 14-len(f'({b},{e},{d},{a})'))}} "
          f"{ecc_qa:>10.6f} {err:>12.2e}  {{{primes_str}}}")

print()

# ═══════════════════════════════════════════════════════════════════════
# 4. PRIME HARMONIC NETWORK — Law of Harmonics
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  PRIME HARMONIC FAMILIES")
print("=" * 78)
print()

# Collect all primes and which bodies they appear in
prime_to_bodies = defaultdict(set)
for name, info in results.items():
    for p in info["primes"]:
        prime_to_bodies[p].add(name)

# Sort primes by how many bodies share them (descending)
sorted_primes = sorted(prime_to_bodies.keys(), key=lambda p: -len(prime_to_bodies[p]))

for p in sorted_primes:
    bodies = sorted(prime_to_bodies[p])
    label = "UNIVERSAL" if len(bodies) > 20 else ("COMMON" if len(bodies) > 10 else "BRIDGE")
    print(f"  Prime {p:>4d} [{label:>9}] ({len(bodies):>2d} bodies): {', '.join(bodies)}")

print()

# ── Unique bridges (primes shared by exactly 2 bodies) ───────────────
print("  UNIQUE BRIDGES (prime shared by exactly 2 bodies):")
bridge_count = 0
for p in sorted_primes:
    if len(prime_to_bodies[p]) == 2:
        pair = sorted(prime_to_bodies[p])
        print(f"    Prime {p}: {pair[0]} <-> {pair[1]}")
        bridge_count += 1
if bridge_count == 0:
    print("    (none — all primes shared by 3+ bodies)")
print()

# ── Parent-satellite prime sharing ───────────────────────────────────
print("  PARENT-SATELLITE PRIME SHARING:")
for name, info in results.items():
    if info["parent"] is not None and info["parent"] in results:
        parent_info = results[info["parent"]]
        shared = info["primes"] & parent_info["primes"]
        pct = len(shared) / max(1, len(info["primes"] | parent_info["primes"])) * 100
        shared_str = ",".join(str(p) for p in sorted(shared)) if shared else "(none)"
        print(f"    {info['parent']:<12} <-> {name:<16} shared: {{{shared_str}}}  "
              f"({pct:.0f}% Jaccard)")
print()


# ═══════════════════════════════════════════════════════════════════════
# 5. HARMONIC STRENGTH MATRIX (Law of Harmonics)
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  TOP 30 HARMONIC PAIRS (by shared prime count)")
print("=" * 78)
print()

body_names = sorted(results.keys())
pairs = []
for i, n1 in enumerate(body_names):
    for n2 in body_names[i + 1:]:
        shared = results[n1]["primes"] & results[n2]["primes"]
        if shared:
            pairs.append((len(shared), n1, n2, shared))

pairs.sort(key=lambda x: -x[0])
for rank, (cnt, n1, n2, shared) in enumerate(pairs[:30], 1):
    shared_str = ",".join(str(p) for p in sorted(shared))
    print(f"  {rank:>2}. {n1:<18} <-> {n2:<18}  {cnt} shared primes: {{{shared_str}}}")
print()


# ═══════════════════════════════════════════════════════════════════════
# 6. MEAN-MOTION RESONANCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  MEAN-MOTION RESONANCE VERIFICATION")
print("=" * 78)
print()

# Known resonances: (body1, body2, ratio_num, ratio_den, description)
RESONANCES = [
    ("Io", "Europa", 1, 2, "Laplace chain 1:2"),
    ("Europa", "Ganymede", 1, 2, "Laplace chain 1:2"),
    ("Io", "Ganymede", 1, 4, "Laplace chain 1:4"),
    ("Pluto", "Neptune", 2, 3, "Plutino resonance"),
    ("Enceladus", "Dione", 1, 2, "Saturnian 1:2"),
    ("Mimas", "Tethys", 1, 2, "Saturnian 1:2"),
    ("Titan", "Iapetus", 3, 4, "Saturnian ~3:4 (approx)"),
]

print(f"  {'Resonance':<40} {'QN Shared Primes':<30} {'Harmonic?'}")
print("  " + "-" * 85)

for b1, b2, rn, rd, desc in RESONANCES:
    if b1 not in results or b2 not in results:
        print(f"  {desc:<40} (body not found)")
        continue
    r1 = results[b1]
    r2 = results[b2]
    shared = r1["primes"] & r2["primes"]
    shared_str = ",".join(str(p) for p in sorted(shared)) if shared else "(none)"
    harmonic = "YES" if shared else "NO"

    # Check if the resonance ratio primes appear in QN structure
    ratio_primes = prime_factors(rn) | prime_factors(rd)
    ratio_in_qn = ratio_primes & (r1["primes"] | r2["primes"])
    ratio_note = ""
    if ratio_primes:
        ratio_note = f"  ratio primes {ratio_primes} in QN: {ratio_in_qn}"

    label = f"{b1}:{b2} = {rn}:{rd}"
    primes_col = "{" + shared_str + "}"
    print(f"  {label:<40} {primes_col:<30} {harmonic}{ratio_note}")

print()


# ═══════════════════════════════════════════════════════════════════════
# 7. TITIUS-BODE CONNECTION
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  TITIUS-BODE LAW vs QA STRUCTURE")
print("=" * 78)
print()

planets_ordered = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
planet_sma = [results[p]["sma"] for p in planets_ordered]

print(f"  {'Pair':<24} {'SMA ratio':>10} {'Nearest e/d':>12} {'QN match':>20} {'Error':>10}")
print("  " + "-" * 80)

for i in range(len(planets_ordered) - 1):
    p1 = planets_ordered[i]
    p2 = planets_ordered[i + 1]
    ratio = results[p2]["sma"] / results[p1]["sma"]

    # Find best QN whose e/d matches the fractional part, or d/b matches ratio
    # We look for a rational approximation of the ratio
    best_match = None
    best_err = float("inf")
    for e_val in range(1, 200):
        for b_val in range(1, 200):
            d_val = b_val + e_val
            a_val = b_val + 2 * e_val
            if math.gcd(b_val, e_val) != 1:
                continue
            # Try d/b as a ratio approximation
            qa_ratio = d_val / b_val
            err = abs(qa_ratio - ratio)
            if err < best_err:
                best_err = err
                best_match = (b_val, e_val, d_val, a_val, qa_ratio)

    bm = best_match
    print(f"  {p1+' -> '+p2:<24} {ratio:>10.4f} {bm[4]:>12.4f} "
          f"({bm[0]},{bm[1]},{bm[2]},{bm[3]}){'':<{max(0,6)}} {best_err:>10.4f}")

print()

# ── Titius-Bode classic formula check ────────────────────────────────
print("  Classic Titius-Bode: a_n = 0.4 + 0.3 * 2^n  (n = -inf, 0, 1, 2, ...)")
print(f"  {'Planet':<12} {'Actual AU':>10} {'T-B AU':>10} {'Ratio':>10}")
tb_seq = [0, 0.3, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 38.4]  # 2^n * 0.3
for i, pname in enumerate(planets_ordered):
    tb_au = 0.4 + tb_seq[i]
    actual = results[pname]["sma"]
    print(f"  {pname:<12} {actual:>10.3f} {tb_au:>10.3f} {actual/tb_au:>10.3f}")
print()


# ═══════════════════════════════════════════════════════════════════════
# 8. PERIOD RATIOS FOR SATELLITE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  SATELLITE PERIOD RATIOS (Kepler's 3rd: T ~ a^1.5)")
print("=" * 78)
print()

# Group moons by parent
parent_groups = defaultdict(list)
for name, info in results.items():
    if info["parent"] is not None:
        parent_groups[info["parent"]].append((name, info))

for parent in ["Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]:
    if parent not in parent_groups:
        continue
    moons = sorted(parent_groups[parent], key=lambda x: x[1]["sma"])
    if len(moons) < 2:
        continue

    print(f"  {parent} system:")
    base_name, base_info = moons[0]
    base_period = base_info["sma"] ** 1.5  # relative period (Kepler)
    for mname, minfo in moons:
        rel_period = (minfo["sma"] / base_info["sma"]) ** 1.5
        print(f"    {mname:<14} SMA_rel={minfo['sma']/base_info['sma']:>8.3f}  "
              f"Period_rel={rel_period:>8.3f}  QN=({minfo['b']},{minfo['e']},{minfo['d']},{minfo['a']})")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 9. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  SUMMARY STATISTICS")
print("=" * 78)
print()

errors = [results[n]["err"] for n in results]
print(f"  Total bodies analyzed: {len(results)}")
print(f"  Mean QN matching error: {np.mean(errors):.2e}")
print(f"  Median QN matching error: {np.median(errors):.2e}")
print(f"  Max QN matching error: {np.max(errors):.2e} "
      f"({[n for n in results if results[n]['err'] == np.max(errors)][0]})")
print(f"  Bodies with error < 1e-4: {sum(1 for e in errors if e < 1e-4)}/{len(errors)}")
print()

all_primes = set()
for info in results.values():
    all_primes |= info["primes"]
print(f"  Distinct primes in QN network: {len(all_primes)}")
print(f"  Primes: {sorted(all_primes)}")
print()

# Category breakdown
cats = defaultdict(list)
for n, info in results.items():
    cats[info["cat"]].append(n)
for cat in ["planet", "moon", "dwarf", "comet", "asteroid"]:
    if cat in cats:
        print(f"  {cat.capitalize()+'s':<12}: {len(cats[cat])}")
print()

# Harmonic density
total_pairs = len(body_names) * (len(body_names) - 1) // 2
harmonic_pairs = sum(1 for cnt, _, _, _ in pairs if cnt > 0)
print(f"  Harmonic pairs (sharing >= 1 prime): {harmonic_pairs}/{total_pairs} "
      f"({100*harmonic_pairs/total_pairs:.1f}%)")

# Non-trivial harmonic density (exclude universal primes {2,3})
nontrivial_pairs = 0
for cnt, n1, n2, shared in pairs:
    nontrivial = shared - {2, 3}
    if nontrivial:
        nontrivial_pairs += 1
print(f"  Non-trivial harmonic pairs (shared prime beyond {{2,3}}): "
      f"{nontrivial_pairs}/{total_pairs} ({100*nontrivial_pairs/total_pairs:.1f}%)")
print()

# ═══════════════════════════════════════════════════════════════════════
# 10. NETWORK GRAPH
# ═══════════════════════════════════════════════════════════════════════

print("Generating network graph...")

G = nx.Graph()

# Color map for categories
cat_colors = {
    "planet": "#FFD700",
    "moon": "#87CEEB",
    "dwarf": "#DDA0DD",
    "comet": "#FF6347",
    "asteroid": "#90EE90",
}

# Add nodes
for name, info in results.items():
    G.add_node(name, cat=info["cat"])

# Add edges for bodies sharing >= 3 primes (skip trivial {2,3} sharing)
edge_threshold = 3
for cnt, n1, n2, shared in pairs:
    if cnt >= edge_threshold:
        G.add_edge(n1, n2, weight=cnt, primes=shared)

# Also always add parent-satellite edges
for name, info in results.items():
    if info["parent"] and info["parent"] in results:
        shared = info["primes"] & results[info["parent"]]["primes"]
        if not G.has_edge(name, info["parent"]):
            G.add_edge(name, info["parent"], weight=max(1, len(shared)),
                       primes=shared)

fig, axes = plt.subplots(1, 2, figsize=(24, 14))

# ── Left panel: full network ─────────────────────────────────────────
ax = axes[0]
ax.set_title("QA Prime Harmonic Network — Solar System", fontsize=14, fontweight="bold")

# Layout
pos = nx.spring_layout(G, k=1.8, iterations=80, seed=42)

# Node colors and sizes
node_colors = [cat_colors.get(results[n]["cat"], "#CCCCCC") for n in G.nodes()]
node_sizes = []
for n in G.nodes():
    cat = results[n]["cat"]
    if cat == "planet":
        node_sizes.append(800)
    elif cat == "dwarf":
        node_sizes.append(500)
    elif cat == "moon":
        node_sizes.append(350)
    else:
        node_sizes.append(300)

# Edge widths
edge_widths = [G[u][v]["weight"] * 0.7 for u, v in G.edges()]
edge_colors = ["#888888" for _ in G.edges()]

# Draw
nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.4, edge_color=edge_colors)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                       edgecolors="black", linewidths=0.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_weight="bold")

# Legend
legend_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                  for cat, c in cat_colors.items()]
ax.legend(handles=legend_patches, loc="lower left", fontsize=9)
ax.set_axis_off()

# ── Right panel: QN eccentricity match quality ──────────────────────
ax2 = axes[1]
ax2.set_title("QA Eccentricity Match Quality", fontsize=14, fontweight="bold")

# Sort bodies by observed eccentricity
sorted_bodies = sorted(results.items(), key=lambda x: x[1]["ecc"])
names_sorted = [n for n, _ in sorted_bodies]
ecc_obs = [info["ecc"] for _, info in sorted_bodies]
ecc_qa = [info["ecc_qa"] for _, info in sorted_bodies]
colors_sorted = [cat_colors.get(info["cat"], "#CCCCCC") for _, info in sorted_bodies]

y_pos = range(len(names_sorted))
ax2.barh(y_pos, ecc_obs, height=0.4, align="center", color=colors_sorted,
         edgecolor="black", linewidth=0.3, label="Observed", alpha=0.7)
ax2.scatter(ecc_qa, y_pos, color="red", s=20, zorder=5, label="QA match", marker="x")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names_sorted, fontsize=5.5)
ax2.set_xlabel("Orbital Eccentricity")
ax2.legend(loc="lower right", fontsize=9)
ax2.set_xlim(-0.02, 1.05)

plt.tight_layout()
plt.savefig("qa_solar_system_deep.png", dpi=150, bbox_inches="tight")
print(f"\nSaved: qa_solar_system_deep.png")
print()

# ═══════════════════════════════════════════════════════════════════════
# 11. NOTABLE FINDINGS
# ═══════════════════════════════════════════════════════════════════════

print("=" * 78)
print("  NOTABLE FINDINGS")
print("=" * 78)
print()

# Earth's QN
e_info = results["Earth"]
print(f"  Earth QN: ({e_info['b']},{e_info['e']},{e_info['d']},{e_info['a']})  "
      f"ecc={e_info['ecc_qa']:.6f} (obs={e_info['ecc']:.6f})")

# Halley's QN
h_info = results["Halley"]
print(f"  Halley QN: ({h_info['b']},{h_info['e']},{h_info['d']},{h_info['a']})  "
      f"ecc={h_info['ecc_qa']:.6f} (obs={h_info['ecc']:.6f})")

# Check Earth-Halley connection (from MEMORY.md)
shared_eh = e_info["primes"] & h_info["primes"]
print(f"  Earth-Halley shared primes: {sorted(shared_eh)}")

# Nereid — most eccentric regular moon
n_info = results["Nereid"]
print(f"  Nereid QN: ({n_info['b']},{n_info['e']},{n_info['d']},{n_info['a']})  "
      f"ecc={n_info['ecc_qa']:.6f} (obs={n_info['ecc']:.6f})")

# Most harmonic pair
if pairs:
    top = pairs[0]
    print(f"  Most harmonic pair: {top[1]} <-> {top[2]} ({top[0]} shared primes: {sorted(top[3])})")

# Bodies sharing primes with ALL planets
planet_primes = set.intersection(*(results[p]["primes"] for p in planets_ordered))
if planet_primes:
    print(f"  Primes common to ALL 8 planets: {sorted(planet_primes)}")
else:
    # Find primes common to most planets
    for threshold in range(7, 3, -1):
        common = set()
        for p in all_primes:
            planet_count = sum(1 for pn in planets_ordered if p in results[pn]["primes"])
            if planet_count >= threshold:
                common.add(p)
        if common:
            print(f"  Primes shared by >= {threshold}/8 planets: {sorted(common)}")
            break

print()
print("=" * 78)
print("  ANALYSIS COMPLETE")
print("=" * 78)
