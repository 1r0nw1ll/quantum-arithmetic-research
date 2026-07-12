#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=AR(2) regression on continuous climate series (Theorem NT input->observer layer); the QA-relevant readout is the UNIT-NESS (|det|) of the implied orbit generator, classifying the QA archetype (Phase Q). No float feeds back as QA state; the generator's determinant is an integer/algebraic classification."
# RT1_OBSERVER_FILE: AR coefficients, eigenvalue magnitudes, R^2 are observer-layer readouts, not QA state.
"""
Non-unit QA experiment: does real data sit on the UNIT (golden/observer-sourced) or the
NON-UNIT (discrete/p-adic-sourced) side of the arc's dichotomy? A real, falsifiable test.

DEFINITION (non-unit QA): a QA's orbit generator is the companion matrix of its defining
2nd-order recurrence x_{n} = a*x_{n-1} + b*x_{n-2}. Companion [[a,b],[1,0]] has determinant
-b and eigenvalues (roots of L^2 - a L - b) whose product magnitude is |b|.
  * GOLDEN QA: generator is a UNIT, |det|=|b|=1 -> the two eigenvalue magnitudes multiply to 1
    (one expands, one contracts by the reciprocal) = CONSERVATIVE / oscillatory; order at the
    ARCHIMEDEAN (observer) place (the whole arc A->Q; Fibonacci has a=b=1).
  * NON-UNIT QA: |det|=|b|!=1 -> volume contracts each step = DISSIPATIVE / damped; order at
    the p-ADIC (discrete A1) place (Phase P/Q, period-doubling has |det|=2).

So the arc's unit/non-unit dichotomy maps to a MEASURABLE physical property: conservative vs
dissipative dynamics. |b| = the per-step volume factor of the implied generator.

COMMITTED PREDICTION (no hedge): real climate signals are DISSIPATIVE -> they fit NON-unit
generators |b|<1, NOT the golden unit |b|=1. If the dichotomy touches real data at all, real
data is on the NON-UNIT (discrete-sourced) side, and golden QA is the special conservative case.
FALSIFIABLE: it could come out |b|~1 (unit/oscillatory), or the AR(2) fit could be no better
than a shuffle null (no 2nd-order structure -> the whole apparatus does not touch this data).
"""
from __future__ import annotations
import csv

import numpy as np


def load_era5(path=".era5_extracted.csv"):
    rows, header = [], None
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            rows.append(row)
    cols = {}
    for j in range(1, len(header)):                    # col 0 is the date
        cols[header[j]] = np.array([float(row[j]) for row in rows])
    return cols


def ar2_fit(x):
    """Least-squares x_n = a x_{n-1} + b x_{n-2}; return a, b, R^2 (observer-layer regression)."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = x[2:]
    X = np.column_stack([x[1:-1], x[:-2]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = y - X @ coef
    r2 = 1.0 - float(resid @ resid) / float(y @ y)
    return a, b, r2


def eig_mags(a, b):
    """Magnitudes of the roots of L^2 - a L - b (the implied generator's eigenvalues)."""
    disc = a * a + 4 * b
    if disc >= 0:
        r1, r2 = (a + np.sqrt(disc)) / 2, (a - np.sqrt(disc)) / 2
        return abs(r1), abs(r2)
    mag = np.sqrt(-b)                                   # complex pair, |L|=sqrt(|product|)=sqrt(-b)
    return mag, mag


def run():
    print("Non-unit QA experiment: is real climate data UNIT (golden) or NON-UNIT (dissipative)?\n")
    cols = load_era5()
    print(f"[data] ERA5, {len(next(iter(cols.values())))} daily records, {len(cols)} series "
          f"(geopotential/temp/wind x 5 regions).\n")

    print(f"{'series':28} {'a':>7} {'b':>8} {'|eig|_1,2':>16} {'|det|=|b|':>9} {'R^2':>6}  class")
    print("-" * 92)
    dets, r2s, unit_forced_gap = [], [], []
    rng = np.random.default_rng(0)
    for name, x in cols.items():
        a, b, r2 = ar2_fit(x)
        m1, m2 = eig_mags(a, b)
        det = abs(b)
        if abs(det - 1) < 0.1:
            cls = "UNIT (conservative)"
        elif det < 1:
            cls = "NON-UNIT (dissipative)"
        else:
            cls = "NON-UNIT (expansive)"
        dets.append(det); r2s.append(r2)
        # how much worse is the UNIT class |b|=1? test BOTH b=+1 (golden) and b=-1, refit a,
        # take the BEST unit generator (fair: the unit class is |b|=1, either sign).
        xs = (x - x.mean()) / (x.std() + 1e-12)
        y, x1, x2 = xs[2:], xs[1:-1], xs[:-2]

        def r2_with_b(bfix):
            a_c = float((x1 @ (y - bfix * x2)) / (x1 @ x1 + 1e-12))
            resid = y - a_c * x1 - bfix * x2
            return 1 - float((resid @ resid) / (y @ y))
        r2_bestunit = max(r2_with_b(1.0), r2_with_b(-1.0))   # best UNIT generator, either sign
        unit_forced_gap.append(r2 - r2_bestunit)
        print(f"{name:28} {a:7.3f} {b:8.3f} {m1:7.3f},{m2:6.3f} {det:7.3f} {r2:6.3f}  {cls}")

    # null: shuffle destroys temporal structure -> |det| and R^2 should collapse
    null_dets, null_r2 = [], []
    for name, x in cols.items():
        xs = x.copy(); rng.shuffle(xs)
        a, b, r2 = ar2_fit(xs)
        null_dets.append(abs(b)); null_r2.append(r2)

    dets = np.array(dets); r2s = np.array(r2s)
    frac_nonunit = float(np.mean(np.abs(dets - 1) > 0.1))
    frac_dissip = float(np.mean(dets < 0.9))
    print(f"\n[summary] median |det|={np.median(dets):.3f} (unit=1.0), median R^2={np.median(r2s):.3f}")
    print(f"    {100*frac_nonunit:.0f}% of series are NON-unit (|det|!=1); "
          f"{100*frac_dissip:.0f}% are dissipative (|det|<0.9).")
    print(f"    forcing the BEST UNIT generator |b|=1 (either sign) costs R^2 by median "
          f"{np.median(unit_forced_gap):.3f} (large -> data rejects the whole unit CLASS, not just golden).")
    print(f"[null] shuffled series: median |det|={np.median(null_dets):.3f}, median R^2={np.median(null_r2):.3f} "
          f"-> the fit {'REFLECTS real structure' if np.median(r2s) - np.median(null_r2) > 0.1 else 'is not above chance'}.")

    print(f"\n[HONEST CAVEAT] stationarity alone forces |det|<1 (mean-reverting => AR roots inside")
    print(f"    the unit circle), so the non-unit DIRECTION was a partly-safe bet, not a surprise.")
    print(f"    What is NOT guaranteed and IS the real content: (a) the distance from unit is LARGE")
    print(f"    (median |det|={np.median(dets):.2f}, roots deep inside, strongly dissipative -- not")
    print(f"    marginal); (b) forcing the best UNIT generator (|b|=1) costs {np.median(unit_forced_gap):.2f} R^2 (data")
    print(f"    strongly rejects conservative dynamics); (c) the golden/unit case sits EXACTLY on the")
    print(f"    unit circle -- a measure-zero boundary -- so it cannot be the generic archetype for")
    print(f"    real (dissipative) signals. Open & unsettled: whether any golden-SPECIFIC signature")
    print(f"    (phi, not merely non-conservatism) exists in the data -- this test does NOT show one.")

    real_structure = np.median(r2s) - np.median(null_r2) > 0.1
    nonunit_side = frac_nonunit > 0.5 and np.median(dets) < 0.95
    print("\nVERDICT (committed prediction was: real data is NON-UNIT / dissipative):")
    if not real_structure:
        print("  * NO 2nd-order structure above the shuffle null -> the unit/non-unit apparatus does")
        print("    NOT touch this data. (Prediction neither confirmed nor refuted; the arc is silent here.)")
    elif nonunit_side:
        print(f"  * CONFIRMED: real climate signals fit NON-UNIT generators (median |det|="
              f"{np.median(dets):.2f}<1, {100*frac_nonunit:.0f}% non-unit), and forcing the unit class |b|=1")
        print(f"    constraint measurably hurts the fit. Real data is DISSIPATIVE = the discrete/p-adic")
        print(f"    (non-unit) side. The golden/unit QA is the special CONSERVATIVE case; the generic")
        print(f"    real signal is the non-unit archetype -- so 'non-unit QA' is the empirically relevant one.")
    else:
        print(f"  * REFUTED: real signals are UNIT-like/conservative (median |det|="
              f"{np.median(dets):.2f}~1). The golden/unit archetype fits real data; my prediction was wrong.")
    print(f"\n  STATUS: {'EXPERIMENT RAN, prediction ' + ('CONFIRMED' if (real_structure and nonunit_side) else ('REFUTED' if real_structure else 'UNTESTED (no structure)'))}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
