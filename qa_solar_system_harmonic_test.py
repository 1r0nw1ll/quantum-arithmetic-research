#!/usr/bin/env python3
"""
qa_solar_system_harmonic_test.py — Quantitative test of QA harmonic structure
in the solar system.

QUESTION A: Do gravitationally coupled solar system bodies share more QN prime
factors than random pairs with the same eccentricity distribution?

QUESTION B: Do real solar-system eccentricities preferentially sit near
low-complexity QA attractor packets rather than merely being arbitrary rational
fits?

TEST DESIGN:
1. Find best QN for each real solar system body using eccentricity e_orbit ≈ e/d.
2. Compute prime factor sharing (Jaccard similarity) for all pairs.
3. Compare COUPLED pairs (parent-satellite, resonance partners) vs ALL pairs.
4. NULL MODEL: random permutations of QN assignments.
5. LOW-ORDER ATTRACTOR TEST: compare exact/high-resolution QN fits to nearest
   low-order primitive packets, especially e_QA ∈ {1,2,3} and d ≤ 120.
6. CONTINUED-FRACTION BASELINE: report whether the selected QN is just the
   minimal-denominator rational fit implied by the measured decimal precision.
7. PREDICTION TEST: withhold bodies and predict QN from parent/neighborhood
   prime sharing.

Tier logic:
- Prime-sharing significance tests relational harmonic structure.
- Low-order diagnostics test recurrence of special QA packet families.
- Continued-fraction diagnostics prevent mistaking rational approximation for
  evidence of QA structure.

Author: Will Dale (design), Claude/ChatGPT (code extensions)
"""

QA_COMPLIANCE = (
    "observer=prime_sharing_and_low_order_attractor_test, "
    "state_alphabet=primitive_QN_tuple, eccentricity_map=e_orbit≈e/d"
)

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

np.random.seed(42)


@dataclass(frozen=True)
class Body:
    name: str
    ecc_text: str
    category: str
    parent: Optional[str]

    @property
    def ecc(self) -> float:
        return float(self.ecc_text)

    @property
    def decimal_tolerance(self) -> float:
        """Half-unit tolerance of the published decimal value."""
        if "." not in self.ecc_text:
            return 0.5
        digits = len(self.ecc_text.split(".", 1)[1])
        return 0.5 * 10 ** (-digits)


@dataclass(frozen=True)
class QNFit:
    b: int
    e: int
    d: int
    a: int
    ecc_qa: float
    err: float

    @property
    def tuple(self) -> Tuple[int, int, int, int]:
        return (self.b, self.e, self.d, self.a)

    @property
    def complexity_d(self) -> int:
        return self.d

    @property
    def complexity_l1(self) -> int:
        return self.b + self.e + self.d + self.a

    @property
    def family(self) -> str:
        if self.e == 1:
            return "unit-e attractor"
        if self.e <= 3:
            return "small-e attractor"
        if self.e <= 9:
            return "low-e harmonic"
        return "high-resolution fit"


# ═══════════════════════════════════════════════════════════════════════
# BODY CATALOGUE (from qa_solar_system_deep.py; eccentricities stored as
# decimal strings so printed precision can be treated as observer tolerance)
# ═══════════════════════════════════════════════════════════════════════

BODIES: List[Body] = [
    Body("Mercury",    "0.20563",  "planet",   None),
    Body("Venus",      "0.00677",  "planet",   None),
    Body("Earth",      "0.01671",  "planet",   None),
    Body("Mars",       "0.09339",  "planet",   None),
    Body("Jupiter",    "0.04839",  "planet",   None),
    Body("Saturn",     "0.05415",  "planet",   None),
    Body("Uranus",     "0.04717",  "planet",   None),
    Body("Neptune",    "0.00859",  "planet",   None),
    Body("Moon",       "0.0549",   "moon",     "Earth"),
    Body("Phobos",     "0.0151",   "moon",     "Mars"),
    Body("Deimos",     "0.0002",   "moon",     "Mars"),
    Body("Io",         "0.0041",   "moon",     "Jupiter"),
    Body("Europa",     "0.0094",   "moon",     "Jupiter"),
    Body("Ganymede",   "0.0011",   "moon",     "Jupiter"),
    Body("Callisto",   "0.0074",   "moon",     "Jupiter"),
    Body("Amalthea",   "0.0032",   "moon",     "Jupiter"),
    Body("Mimas",      "0.0196",   "moon",     "Saturn"),
    Body("Enceladus",  "0.0047",   "moon",     "Saturn"),
    Body("Tethys",     "0.0001",   "moon",     "Saturn"),
    Body("Dione",      "0.0022",   "moon",     "Saturn"),
    Body("Rhea",       "0.0013",   "moon",     "Saturn"),
    Body("Titan",      "0.0288",   "moon",     "Saturn"),
    Body("Iapetus",    "0.0276",   "moon",     "Saturn"),
    Body("Miranda",    "0.0013",   "moon",     "Uranus"),
    Body("Ariel",      "0.0012",   "moon",     "Uranus"),
    Body("Umbriel",    "0.0039",   "moon",     "Uranus"),
    Body("Titania",    "0.0011",   "moon",     "Uranus"),
    Body("Oberon",     "0.0014",   "moon",     "Uranus"),
    Body("Triton",     "0.000016", "moon",     "Neptune"),
    Body("Nereid",     "0.7507",   "moon",     "Neptune"),
    Body("Charon",     "0.0002",   "moon",     "Pluto"),
    Body("Pluto",      "0.2488",   "dwarf",    None),
    Body("Ceres",      "0.0758",   "dwarf",    None),
    Body("Eris",       "0.4407",   "dwarf",    None),
    Body("Haumea",     "0.1912",   "dwarf",    None),
    Body("Makemake",   "0.1559",   "dwarf",    None),
    Body("Sedna",      "0.8496",   "dwarf",    None),
    Body("Halley",     "0.96714",  "comet",    None),
    Body("Hale-Bopp",  "0.99507",  "comet",    None),
    Body("Encke",      "0.8483",   "comet",    None),
    Body("Hyakutake",  "0.99990",  "comet",    None),
    Body("67P/C-G",    "0.6405",   "comet",    None),
    Body("Vesta",      "0.0887",   "asteroid", None),
    Body("Pallas",     "0.2313",   "asteroid", None),
    Body("Eros",       "0.2229",   "asteroid", None),
    Body("Bennu",      "0.2037",   "asteroid", None),
    Body("Itokawa",    "0.2802",   "asteroid", None),
]

BODY_BY_NAME: Dict[str, Body] = {body.name: body for body in BODIES}

COUPLED_PAIRS: List[Tuple[str, str]] = [
    (body.parent, body.name) for body in BODIES if body.parent is not None
]

RESONANCE_PAIRS: List[Tuple[str, str]] = [
    ("Io", "Europa"),           # 2:1
    ("Europa", "Ganymede"),     # 2:1
    ("Mimas", "Tethys"),        # 2:1
    ("Enceladus", "Dione"),     # 2:1
    ("Titan", "Iapetus"),       # 5:1 approx
    ("Pluto", "Neptune"),       # 3:2
    ("Jupiter", "Saturn"),      # 5:2 Great Inequality
]

LOW_ORDER_MAX_D = 120
LOW_ORDER_MAX_E = 3
N_PERM = 10_000


# ═══════════════════════════════════════════════════════════════════════
# QN FINDERS AND QA DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

def qn_from_e_d(e_val: int, d_val: int) -> Optional[QNFit]:
    if d_val <= 0 or e_val <= 0 or e_val >= d_val:
        return None
    b_val = d_val - e_val
    if gcd(b_val, e_val) != 1:
        return None
    a_val = b_val + 2 * e_val
    ecc_qa = e_val / d_val
    return QNFit(b_val, e_val, d_val, a_val, ecc_qa, 0.0)


def find_qn(ecc_target: float, max_d: int = 500) -> Optional[QNFit]:
    """Find best primitive QN (b,e,d,a) where orbit eccentricity ≈ e/d."""
    if ecc_target <= 0:
        return QNFit(1, 0, 1, 1, 0.0, abs(ecc_target))
    if ecc_target >= 1:
        ecc_qa = max_d / (max_d + 1)
        return QNFit(1, max_d, max_d + 1, max_d + 2, ecc_qa, abs(ecc_target - ecc_qa))

    if ecc_target < 0.005:
        max_d = max(max_d, 5_000)
    if ecc_target < 0.001:
        max_d = max(max_d, 10_000)
    if ecc_target < 0.0001:
        max_d = max(max_d, 100_000)
    if ecc_target > 0.99:
        max_d = max(max_d, 10_000)

    best: Optional[QNFit] = None
    best_err = float("inf")

    for d_val in range(2, max_d + 1):
        e_center = round(ecc_target * d_val)
        for e_try in range(max(1, e_center - 1), min(d_val, e_center + 2)):
            qn = qn_from_e_d(e_try, d_val)
            if qn is None:
                continue
            err = abs(qn.ecc_qa - ecc_target)
            if err < best_err:
                best_err = err
                best = QNFit(qn.b, qn.e, qn.d, qn.a, qn.ecc_qa, err)
                if err < 1e-12:
                    return best

    return best


def nearest_low_order_qn(
    ecc_target: float,
    max_d: int = LOW_ORDER_MAX_D,
    max_e: int = LOW_ORDER_MAX_E,
) -> Optional[QNFit]:
    """Nearest primitive low-complexity attractor packet."""
    best: Optional[QNFit] = None
    best_key = (float("inf"), float("inf"), float("inf"))
    for d_val in range(2, max_d + 1):
        for e_val in range(1, min(max_e, d_val - 1) + 1):
            qn = qn_from_e_d(e_val, d_val)
            if qn is None:
                continue
            err = abs(qn.ecc_qa - ecc_target)
            rel = err / max(ecc_target, 1e-15)
            key = (rel, err, d_val)
            if key < best_key:
                best_key = key
                best = QNFit(qn.b, qn.e, qn.d, qn.a, qn.ecc_qa, err)
    return best


def minimal_d_within_tolerance(ecc_target: float, tolerance: float, max_d: int = 200_000) -> Optional[QNFit]:
    """Smallest d primitive QN that fits the observer decimal tolerance."""
    for d_val in range(2, max_d + 1):
        e_center = round(ecc_target * d_val)
        for e_try in range(max(1, e_center - 1), min(d_val, e_center + 2)):
            qn = qn_from_e_d(e_try, d_val)
            if qn is None:
                continue
            err = abs(qn.ecc_qa - ecc_target)
            if err <= tolerance:
                return QNFit(qn.b, qn.e, qn.d, qn.a, qn.ecc_qa, err)
    return None


def continued_fraction_baseline(ecc_target: float, max_denominator: int) -> QNFit:
    """Best rational approximation baseline using Python's continued fractions."""
    frac = Fraction(ecc_target).limit_denominator(max_denominator)
    qn = qn_from_e_d(frac.numerator, frac.denominator)
    if qn is None:
        # Fall back to exhaustive primitive search because limit_denominator may
        # return a non-QA-legal edge case for extreme values.
        fallback = find_qn(ecc_target, max_denominator)
        if fallback is None:
            raise RuntimeError(f"no CF baseline for eccentricity {ecc_target}")
        return fallback
    return QNFit(qn.b, qn.e, qn.d, qn.a, qn.ecc_qa, abs(qn.ecc_qa - ecc_target))


def qa_mod_class(qn: QNFit) -> Tuple[int, int, int]:
    """Compact modular signature for recurrence diagnostics."""
    return (qn.e % 9, qn.d % 24, qn.a % 24)


def prime_factors(n: int) -> Set[int]:
    if n <= 1:
        return set()
    factors: Set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def qn_primes(qn: QNFit) -> Set[int]:
    return prime_factors(qn.b * max(qn.e, 1) * qn.d * qn.a)


def jaccard(set1: Set[int], set2: Set[int]) -> float:
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def mean_jaccard_for_pairs(prime_assignment: Sequence[Set[int]], pair_indices: Sequence[Tuple[int, int]]) -> float:
    if not pair_indices:
        return 0.0
    return sum(jaccard(prime_assignment[i], prime_assignment[j]) for i, j in pair_indices) / len(pair_indices)


def format_qn(qn: QNFit) -> str:
    return f"({qn.b},{qn.e},{qn.d},{qn.a})"


# ═══════════════════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 80)
    print("QA SOLAR SYSTEM HARMONIC TEST — Phase 2 Geodesy Roadmap")
    print("=" * 80)

    print("\n── STEP 1: High-Resolution QN Assignment ──")
    body_qns: Dict[str, QNFit] = {}
    body_primes: Dict[str, Set[int]] = {}
    names: List[str] = []

    for body in BODIES:
        qn = find_qn(body.ecc)
        if qn is None:
            continue
        body_qns[body.name] = qn
        body_primes[body.name] = qn_primes(qn)
        names.append(body.name)

    print(f"  {len(names)} bodies assigned QNs")
    for name in ["Mercury", "Earth", "Moon", "Jupiter", "Io", "Halley"]:
        if name in body_qns:
            body = BODY_BY_NAME[name]
            qn = body_qns[name]
            print(
                f"  {name:12s}: {format_qn(qn):22s} "
                f"ecc={qn.ecc_qa:.8f} target={body.ecc_text:>8s} "
                f"err={qn.err:.2e} family={qn.family:22s} primes={sorted(body_primes[name])}"
            )

    print("\n── STEP 2: Low-Order Attractor Diagnostics ──")
    low_order: Dict[str, QNFit] = {}
    tolerance_hits = 0
    unit_e_hits = 0
    small_e_hits = 0

    rows = []
    for body in BODIES:
        qn_low = nearest_low_order_qn(body.ecc)
        if qn_low is None:
            continue
        low_order[body.name] = qn_low
        within_tolerance = qn_low.err <= body.decimal_tolerance
        tolerance_hits += int(within_tolerance)
        unit_e_hits += int(qn_low.e == 1)
        small_e_hits += int(qn_low.e <= LOW_ORDER_MAX_E)
        rows.append((qn_low.err / max(body.ecc, 1e-15), body, qn_low, within_tolerance))

    rows.sort(key=lambda item: item[0])
    print(
        f"  Low-order search: primitive QN with e≤{LOW_ORDER_MAX_E}, d≤{LOW_ORDER_MAX_D}"
    )
    print(f"  Decimal-tolerance hits: {tolerance_hits}/{len(BODIES)}")
    print(f"  Nearest low-order packets with e=1: {unit_e_hits}/{len(BODIES)}")
    print(f"  Nearest low-order packets with e≤{LOW_ORDER_MAX_E}: {small_e_hits}/{len(BODIES)}")
    print("\n  Best low-order hits by relative error:")
    for rel_err, body, qn_low, within_tolerance in rows[:12]:
        print(
            f"    {body.name:12s}: low={format_qn(qn_low):18s} "
            f"e/d={qn_low.ecc_qa:.8f} target={body.ecc_text:>8s} "
            f"rel_err={rel_err:8.3%} mod(e,d,a)={qa_mod_class(qn_low)} "
            f"{'WITHIN_DECIMAL_TOL' if within_tolerance else ''}"
        )

    print("\n  Earth/Moon canonical low-order geodesy packets:")
    for name in ["Earth", "Moon"]:
        body = BODY_BY_NAME[name]
        qn_low = low_order[name]
        qn_hi = body_qns[name]
        print(
            f"    {name:5s}: low={format_qn(qn_low):18s} err={qn_low.err:.3e}; "
            f"hi={format_qn(qn_hi):22s} err={qn_hi.err:.3e}; "
            f"complexity_ratio d_hi/d_low={qn_hi.d / qn_low.d:.2f}"
        )

    print("\n── STEP 3: Continued-Fraction / Decimal-Precision Baseline ──")
    cf_matches = 0
    min_tol_hits = 0
    baseline_rows = []
    for body in BODIES:
        qn_hi = body_qns[body.name]
        cf = continued_fraction_baseline(body.ecc, qn_hi.d)
        min_tol = minimal_d_within_tolerance(body.ecc, body.decimal_tolerance)
        cf_matches += int(cf.tuple == qn_hi.tuple)
        min_tol_hits += int(min_tol is not None)
        baseline_rows.append((body.name, body.ecc_text, qn_hi, cf, min_tol))

    print(f"  High-resolution QN equals CF best within same denominator: {cf_matches}/{len(BODIES)}")
    print(f"  Minimal primitive QN within published decimal tolerance found: {min_tol_hits}/{len(BODIES)}")
    print("\n  Representative baseline rows:")
    for name, ecc_text, qn_hi, cf, min_tol in baseline_rows[:10]:
        min_txt = format_qn(min_tol) if min_tol else "none"
        min_d = min_tol.d if min_tol else -1
        print(
            f"    {name:12s}: target={ecc_text:>8s} hi={format_qn(qn_hi):22s} "
            f"CF={format_qn(cf):22s} min_tol={min_txt:22s} min_d={min_d}"
        )

    print("\n── STEP 4: Prime Sharing — Coupled vs All Pairs ──")
    all_jaccards = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            all_jaccards.append(jaccard(body_primes[n1], body_primes[n2]))

    coupled_jaccards = []
    coupled_labels = []
    for parent, child in COUPLED_PAIRS:
        if parent in body_primes and child in body_primes:
            coupled_jaccards.append(jaccard(body_primes[parent], body_primes[child]))
            coupled_labels.append(f"{parent}-{child}")

    resonance_jaccards = []
    resonance_labels = []
    for n1, n2 in RESONANCE_PAIRS:
        if n1 in body_primes and n2 in body_primes:
            resonance_jaccards.append(jaccard(body_primes[n1], body_primes[n2]))
            resonance_labels.append(f"{n1}-{n2}")

    print(f"  All pairs:       mean Jaccard = {np.mean(all_jaccards):.4f}  (n={len(all_jaccards)})")
    print(f"  Coupled pairs:   mean Jaccard = {np.mean(coupled_jaccards):.4f}  (n={len(coupled_jaccards)})")
    print(f"  Resonance pairs: mean Jaccard = {np.mean(resonance_jaccards):.4f}  (n={len(resonance_jaccards)})")

    print("\n  Top coupled pairs by prime sharing:")
    for j, label in sorted(zip(coupled_jaccards, coupled_labels), reverse=True)[:10]:
        p, c = label.split("-")
        print(f"    {label:25s} J={j:.3f}  primes_shared={sorted(body_primes[p] & body_primes[c])}")

    print("\n  Resonance pairs:")
    for j, label in zip(resonance_jaccards, resonance_labels):
        n1, n2 = label.split("-")
        print(f"    {label:25s} J={j:.3f}  primes_shared={sorted(body_primes[n1] & body_primes[n2])}")

    print(f"\n── STEP 5: Permutation Null Model ({N_PERM} shuffles) ──")
    prime_list = [body_primes[n] for n in names]
    name_to_idx = {n: i for i, n in enumerate(names)}
    coupled_idx = [(name_to_idx[p], name_to_idx[c]) for p, c in COUPLED_PAIRS if p in name_to_idx and c in name_to_idx]
    resonance_idx = [(name_to_idx[n1], name_to_idx[n2]) for n1, n2 in RESONANCE_PAIRS if n1 in name_to_idx and n2 in name_to_idx]

    real_coupled_mean = mean_jaccard_for_pairs(prime_list, coupled_idx)
    real_resonance_mean = mean_jaccard_for_pairs(prime_list, resonance_idx)

    null_coupled = []
    null_resonance = []
    for _ in range(N_PERM):
        perm = np.random.permutation(len(names))
        shuffled_primes = [prime_list[p] for p in perm]
        null_coupled.append(mean_jaccard_for_pairs(shuffled_primes, coupled_idx))
        null_resonance.append(mean_jaccard_for_pairs(shuffled_primes, resonance_idx))

    null_coupled = np.array(null_coupled)
    null_resonance = np.array(null_resonance)
    p_coupled = float(np.mean(null_coupled >= real_coupled_mean))
    p_resonance = float(np.mean(null_resonance >= real_resonance_mean))
    z_coupled = float((real_coupled_mean - np.mean(null_coupled)) / (np.std(null_coupled) + 1e-10))
    z_resonance = float((real_resonance_mean - np.mean(null_resonance)) / (np.std(null_resonance) + 1e-10))

    print("  COUPLED PAIRS:")
    print(f"    Real mean Jaccard:  {real_coupled_mean:.4f}")
    print(f"    Null mean:          {np.mean(null_coupled):.4f} ± {np.std(null_coupled):.4f}")
    print(f"    z-score:            {z_coupled:+.2f}")
    print(f"    p-value (one-tail): {p_coupled:.4f}")
    print(f"    Verdict:            {'SIGNIFICANT (p<0.05)' if p_coupled < 0.05 else 'NOT SIGNIFICANT'}")

    print("\n  RESONANCE PAIRS:")
    print(f"    Real mean Jaccard:  {real_resonance_mean:.4f}")
    print(f"    Null mean:          {np.mean(null_resonance):.4f} ± {np.std(null_resonance):.4f}")
    print(f"    z-score:            {z_resonance:+.2f}")
    print(f"    p-value (one-tail): {p_resonance:.4f}")
    print(f"    Verdict:            {'SIGNIFICANT (p<0.05)' if p_resonance < 0.05 else 'NOT SIGNIFICANT'}")

    print("\n── STEP 6: Withheld Body Prediction Test ──")
    print("  (Withhold 5 moons, predict their QN from parent's prime network)")
    withheld = ["Europa", "Titan", "Oberon", "Phobos", "Triton"]
    print(f"  Withheld bodies: {withheld}")

    correct = 0
    total = 0
    for wname in withheld:
        body = BODY_BY_NAME[wname]
        if body.parent is None or body.parent not in body_primes:
            continue
        parent_primes = body_primes[body.parent]
        real_qn = body_qns[wname]

        candidates = []
        for d_val in range(2, 200):
            for e_val in range(1, d_val):
                qn = qn_from_e_d(e_val, d_val)
                if qn is None:
                    continue
                if abs(qn.ecc_qa - body.ecc) / max(body.ecc, 1e-10) < 0.20:
                    cand_primes = qn_primes(qn)
                    j_parent = jaccard(cand_primes, parent_primes)
                    candidates.append((j_parent, -qn.d, qn, cand_primes))

        if not candidates:
            print(f"  {wname}: no candidates found")
            continue

        candidates.sort(reverse=True)
        best_cand = candidates[0]
        predicted_qn = best_cand[2]
        match = predicted_qn.tuple == real_qn.tuple
        correct += int(match)
        total += 1
        print(
            f"  {wname:12s}: real={format_qn(real_qn):22s} "
            f"predicted={format_qn(predicted_qn):22s} "
            f"{'MATCH' if match else 'MISS'} "
            f"(J_parent={best_cand[0]:.3f}, n_candidates={len(candidates)})"
        )

    print(f"\n  Prediction accuracy: {correct}/{total} = {correct / max(total, 1) * 100:.0f}%")
    print("  Chance level: ~1/n_candidates (varies per body)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Coupled pairs test:   z={z_coupled:+.2f}, p={p_coupled:.4f} "
          f"{'→ Tier 3 CONFIRMED' if p_coupled < 0.05 else '→ Tier 2 (not significant)'}")
    print(f"  Resonance pairs test: z={z_resonance:+.2f}, p={p_resonance:.4f} "
          f"{'→ Tier 3 CONFIRMED' if p_resonance < 0.05 else '→ Tier 2 (not significant)'}")
    print(f"  Prediction test:      {correct}/{total} matches")
    print(f"  Low-order decimal-tolerance hits: {tolerance_hits}/{len(BODIES)}")
    print(f"  CF-baseline identity hits:        {cf_matches}/{len(BODIES)}")
    print()

    if p_coupled < 0.05 or p_resonance < 0.05:
        print("  AT LEAST ONE RELATIONAL TEST SIGNIFICANT — QA harmonic structure is")
        print("  non-random in gravitationally coupled or resonant systems under this")
        print("  observer test.")
    else:
        print("  RELATIONAL TESTS NOT SIGNIFICANT — prime sharing does not exceed")
        print("  chance-level assignment under this null model.")

    print()
    print("  INTERPRETATION DISCIPLINE:")
    print("  • High-resolution QNs mainly test rational approximation, not QA physics.")
    print("  • Low-order packets test recurrence of special QA attractor families.")
    print("  • CF/min-denominator diagnostics separate generic rational approximation")
    print("    from nontrivial QA structure.")
    print("  • A positive low-order result should next be tested against a synthetic")
    print("    eccentricity distribution with the same category/range profile.")


if __name__ == "__main__":
    main()
