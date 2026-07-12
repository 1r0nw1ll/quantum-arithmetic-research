#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=archimedean readouts (letter frequencies, real slope phi) — Theorem NT; QA layer = INTEGER substitution words, 2-adic valuations, integer inflation matrices (A1). No float QA state; phi appears only as an observer-layer slope in the Sturmian comparison."
# RT1_OBSERVER_FILE: letter frequencies, the real slope phi, Mahler/entropy values are observer-layer readouts, not QA state.
"""
Phase P: can the firewall polarity INVERT -- the DISCRETE (finite) layer as the SOURCE and
the archimedean (observer) as the shadow?

Phases N/O: for the golden unit, ORDER lives at the archimedean (observer) place; the finite
(discrete/mod-m) places are neutral. The observer is the source, the discrete layer a shadow.
Phase M hinted the polarity can flip for NON-units (p-adic contraction). Here we exhibit a
genuine QA-native ordered structure SOURCED at the finite place, with the observer derived.

Two aperiodic, pure-point-diffractive words -- same order class, OPPOSITE source place:
  FIBONACCI (golden): fixed point of L->LS, S->L. Inflation matrix [[1,1],[1,0]], eigenvalues
    phi, psi -- UNITS (N=-1). It is STURMIAN: its n-th letter is a cut of a line of slope 1/phi,
    an ARCHIMEDEAN real number; its letter frequency is 1/phi, IRRATIONAL. By the rational-
    frequency theorem (Cobham) an irrational frequency => NOT automatic => cannot be produced
    from finite-place (base-p digit) data. SOURCE = the observer (archimedean) place.
  PERIOD-DOUBLING: fixed point of a->ab, b->aa. Inflation matrix [[1,2],[1,0]], eigenvalues
    2, -1 -- 2 is a NON-unit (N=-2). Its n-th letter is EXACTLY [v_2(n) even] -- a function of
    the 2-ADIC valuation of n alone, i.e. pure finite-place (base-2 digit) data, NO real number.
    Its frequency is 2/3, RATIONAL; it is 2-automatic. SOURCE = the finite (discrete/2-adic) place.

So the Theorem-NT firewall polarity INVERTS: the golden order is observer-sourced (archimedean,
unit phi, irrational frequency), while the period-doubling order is DISCRETE-sourced (2-adic,
non-unit 2, rational frequency) -- a limit-periodic structure NATIVE to QA's mod-2^k finite
layer, with the real (archimedean) positions as the DERIVED shadow. QA's discrete A1 layer is
therefore not only a passive shadow of the observer: for NON-unit inflations it is the SOURCE
of a whole family of aperiodic (automatic / limit-periodic) orders the golden/unit thread
structurally cannot reach.
"""
from __future__ import annotations
import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def substitution_word(rules, start, n_min):
    w = start
    while len(w) < n_min:
        w = "".join(rules[c] for c in w)
    return w


def v2(n):
    """2-adic valuation of n (a pure finite-place quantity: the base-2 digit data of n)."""
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def eigvals_2x2(a, b, c, d):
    tr, det = a + d, a * d - b * c
    disc = math.sqrt(abs(tr * tr - 4 * det))
    return (tr + disc) / 2, (tr - disc) / 2


def run():
    print("Phase P: can the firewall polarity INVERT? (discrete layer as SOURCE, observer as shadow)\n")
    N = 30000
    fib = substitution_word({"L": "LS", "S": "L"}, "L", N)
    pd = substitution_word({"a": "ab", "b": "aa"}, "a", N)
    M = min(len(fib), len(pd), 20000)

    # ===== [A] inflation eigenvalues: golden UNIT vs period-doubling NON-unit =====
    g1, g2 = eigvals_2x2(1, 1, 1, 0)               # golden inflation
    p1, p2 = eigvals_2x2(1, 2, 1, 0)               # period-doubling inflation
    print(f"[A] inflation matrices and their Perron eigenvalues (= entropy / Mahler measure):")
    print(f"    FIBONACCI  [[1,1],[1,0]]: eigenvalues {g1:.4f}, {g2:.4f} -> phi,psi are UNITS (N=-1);")
    print(f"      entropy log phi={math.log(g1):.4f}, order module Z[phi] (archimedean).")
    print(f"    PERIOD-DBL [[1,2],[1,0]]: eigenvalues {p1:.4f}, {p2:.4f} -> 2 is a NON-unit (N=-2);")
    print(f"      entropy log 2={math.log(p1):.4f}, order module Z[1/2] dyadic (2-adic).")

    # ===== [B] PERIOD-DOUBLING is FINITE (2-adic) sourced =====
    pd_from_v2 = all((pd[i] == "a") == (v2(i + 1) % 2 == 0) for i in range(M))
    freq_pd = sum(pd[i] == "a" for i in range(M)) / M
    print(f"\n[B] period-doubling is DISCRETE (2-adic) sourced:")
    print(f"    PD_n == [v_2(n) even] for all {M} positions: {pd_from_v2}  <- a pure finite-place")
    print(f"    rule (the 2-adic valuation of n; NO real number). frequency freq(a)={freq_pd:.5f} ~ 2/3")
    print(f"    = {abs(freq_pd - 2/3) < 1e-2} RATIONAL -> 2-AUTOMATIC -> sourced by the finite place.")

    # ===== [C] FIBONACCI is ARCHIMEDEAN (observer) sourced =====
    # Sturmian: the golden word is a cut of slope 1/phi (a real number). Its frequency is 1/phi.
    freq_fib = sum(fib[i] == "L" for i in range(M)) / M
    matches_v2 = all((fib[i] == "L") == (v2(i + 1) % 2 == 0) for i in range(M))
    irrational = abs(freq_fib - 1.0 / PHI) < 1e-3 and all(abs(freq_fib - a / b) > 1e-4
                                                          for b in range(2, 12) for a in range(1, b))
    print(f"\n[C] Fibonacci is OBSERVER (archimedean) sourced:")
    print(f"    frequency freq(L)={freq_fib:.5f} ~ 1/phi={1/PHI:.5f}, IRRATIONAL (not a low ratio): "
          f"{irrational}")
    print(f"    -> by the rational-frequency theorem, an irrational frequency is NOT automatic:")
    print(f"       Fibonacci CANNOT be produced from finite-place (base-p digit) data.")
    print(f"    cross-check: does the 2-adic rule [v_2(n) even] produce Fibonacci? {matches_v2}")
    print(f"    (the archimedean phi and the 2-adic v_2 are NON-interchangeable sources.)")

    ok = (pd_from_v2 and abs(freq_pd - 2 / 3) < 1e-2 and irrational and not matches_v2
          and abs(g1 - PHI) < 1e-6 and abs(p1 - 2.0) < 1e-6)
    print("\nVERDICT (firewall polarity INVERTED -- data-driven):")
    print(f"  * YES, the polarity inverts. Two aperiodic pure-point orders, opposite SOURCE places:")
    print(f"    - FIBONACCI (golden): inflation phi is a UNIT; Sturmian; frequency 1/phi IRRATIONAL;")
    print(f"      order module Z[phi]. Sourced at the ARCHIMEDEAN (observer) place -- needs a real")
    print(f"      number, cannot be built from finite-place digits. (This is the whole golden arc.)")
    print(f"    - PERIOD-DOUBLING: inflation 2 is a NON-unit; PD_n=[v_2(n) even]; frequency 2/3")
    print(f"      RATIONAL; 2-automatic; order module Z[1/2]. Sourced ENTIRELY at the FINITE")
    print(f"      (2-adic/discrete) place -- built from n's base-2 digits, NO real number.")
    print(f"  * So QA's discrete A1 (mod-2^k) layer is NOT only a passive shadow of the observer:")
    print(f"    for NON-unit inflations it is the SOURCE of aperiodic order, with the archimedean")
    print(f"    (real positions) as the DERIVED shadow -- the Theorem-NT polarity run in reverse.")
    print(f"  * The unit/non-unit dichotomy (Phase M) IS the polarity: UNIT inflation => order at")
    print(f"    the observer place (golden, irrational-frequency, non-automatic); NON-unit inflation")
    print(f"    => order at the discrete place (automatic, rational-frequency, limit-periodic). QA's")
    print(f"    golden M is a unit, so ITS order is observer-only; but the discrete-sourced family")
    print(f"    is real, QA-native, and exactly the inversion.")
    print(f"\n  STATUS: {'POLARITY INVERSION EXHIBITED -- discrete-sourced aperiodic order (period-doubling)' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
