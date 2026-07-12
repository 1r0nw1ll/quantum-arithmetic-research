#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=archimedean readouts (circle position {n*phi}, frequencies, entropy) — Theorem NT; QA layer = INTEGER substitution words, 2-adic valuations, integer inflation matrices (A1). No float QA state; phi appears only as an observer-layer circle coordinate."
# RT1_OBSERVER_FILE: circle positions {n*phi}, letter frequencies, entropy/Mahler values are observer-layer readouts, not QA state.
"""
Phase Q: the end-to-end NON-unit QA -- two QA archetypes, and which COMPLETION each order needs.

Phase P showed the firewall polarity inverts (period-doubling is discrete/finite-sourced). This
phase builds the discrete-sourced QA end-to-end and pins down the precise structural fact: the
two orders live in COMPLEMENTARY COMPLETIONS of the global field, and 'which place is the source'
is 'which completion the order is local to'.

  ARCHETYPE 1 -- GOLDEN QA (unit generator): M=[[0,1],[1,1]], Perron phi, a UNIT. Its word is
    STURMIAN: the n-th letter is [ {n/phi} >= 1-1/phi ], a function of the ARCHIMEDEAN circle
    position {n*phi} (a real number). It is archimedean-LOCAL and NOT 2-adic-local. Its order
    is the ARCHIMEDEAN completion (real precision of phi = the OBSERVER place). Entropy log phi;
    module Z[phi]. This is the whole arc A->P.

  ARCHETYPE 2 -- DISCRETE-SOURCED QA (non-unit generator): PD=[[1,2],[1,0]], Perron 2, a NON-unit.
    Its word is 2-AUTOMATIC: the n-th letter is [ v_2(n) even ], a function of the 2-ADIC data of
    n (base-2 digits). It is 2-adic-LOCAL and NOT archimedean-local. Its order is the p-ADIC
    completion (the mod-2^k tower = QA's A1 finite-alphabet layer taken to its natural limit).
    Entropy log 2; module Z[1/2]. NO observer projection is needed to generate this order.

THE PRECISE FACT (verified 2x2): golden order is archimedean-local (Sturmian) and NOT finite-
local; period-doubling order is finite-local (automatic) and NOT archimedean. So the choice of
generator's UNIT-NESS (Phase M) selects WHICH COMPLETION hosts the order:
    UNIT generator     -> order at the ARCHIMEDEAN (observer) completion  (golden, quasicrystal)
    NON-UNIT generator -> order at the p-ADIC (discrete A1) completion     (automatic, limit-periodic)

Honest consequence for the QA framework: Theorem NT holds the discrete layer to be primary and
continuous functions to be observer readouts. For the GOLDEN QA that is in tension -- its
characteristic ORDER (the quasicrystal) is observer-sited (archimedean), which is exactly why
mod-m destroys it (Phase L). A NON-UNIT QA resolves the tension: its order IS native to the
discrete mod-p^k layer, the observer (real positions) being an optional derived readout. Both
archetypes are shadows of the one global adelic object; the generator picks the source place.
"""
from __future__ import annotations
import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0
ALPHA = PHI - 1.0                                  # 1/phi


def substitution_word(rules, start, n_min):
    w = start
    while len(w) < n_min:
        w = "".join(rules[c] for c in w)
    return w


def v2(n):
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def run():
    print("Phase Q: two QA archetypes -- which COMPLETION each order is local to\n")
    N = 40000
    fib = substitution_word({"L": "LS", "S": "L"}, "L", N)
    pd = substitution_word({"a": "ab", "b": "aa"}, "a", N)
    M = 20000

    # ===== [A] the two generators: unit vs non-unit =====
    print("[A] two QAs, by generator unit-ness:")
    print(f"    GOLDEN     M=[[0,1],[1,1]]  Perron phi={PHI:.4f} UNIT (N=-1); entropy log phi="
          f"{math.log(PHI):.4f}; module Z[phi]; order type: quasicrystal (Sturmian).")
    print(f"    PERIOD-DBL [[1,2],[1,0]]  Perron 2.0000 NON-unit (N=-2); entropy log 2="
          f"{math.log(2):.4f}; module Z[1/2]; order type: limit-periodic (automatic).")

    # ===== [B] the 2x2 locality proof: complementary completions =====
    # GOLDEN: archimedean-local (Sturmian) ?  and  2-adic-local ?
    fib_arch = sum((fib[n - 1] == "L") != (((n * ALPHA) % 1.0) >= 1 - ALPHA) for n in range(1, M))
    K = 10
    fib_padic_counterex = any(fib[n - 1] != fib[n - 1 + (1 << K)] for n in range(1, 3000))
    # PERIOD-DBL: 2-adic-local ([v_2 even]) ?  and  archimedean-local ?
    pd_padic = all((pd[n - 1] == "a") == (v2(n) % 2 == 0) for n in range(1, M))
    pd_arch_minmiss = min(sum((pd[n - 1] == "a") != (((n * ALPHA) % 1.0) < t) for n in range(1, M))
                          for t in (i / 200 for i in range(201)))

    print(f"\n[B] the 2x2 locality proof (which completion the order lives in):")
    print(f"    GOLDEN  archimedean-local?  fib == Sturmian [{{n/phi}}>=1-1/phi]: {fib_arch} mismatches "
          f"-> {'YES' if fib_arch == 0 else 'no'}")
    print(f"            2-adic-local?        exists n with fib_n != fib_(n+2^{K}): {fib_padic_counterex} "
          f"-> {'NO (not finite-local)' if fib_padic_counterex else 'finite-local'}")
    print(f"    PERIOD-DBL 2-adic-local?     pd == [v_2(n) even]: {pd_padic} -> {'YES' if pd_padic else 'no'}")
    print(f"            archimedean-local?   best threshold on {{n/phi}} still misses "
          f"{pd_arch_minmiss}/{M} ({100*pd_arch_minmiss//M}%) -> {'NO (not archimedean)' if pd_arch_minmiss > M*0.1 else 'archimedean'}")

    # ===== [C] the completion each order needs =====
    print(f"\n[C] the completion each order is local to:")
    print(f"    GOLDEN order   = the ARCHIMEDEAN completion: to know letter n you need the real")
    print(f"      circle position {{n*phi}} (real precision of phi) -- the OBSERVER place. mod-m")
    print(f"      data does NOT determine it (Phase L: mod-m destroys the quasicrystal).")
    print(f"    PERIOD-DBL order = the p-ADIC completion: to know letter n you need only v_2(n)")
    print(f"      (base-2 digits) -- the mod-2^k tower = QA's A1 finite layer at its natural limit.")
    print(f"      NO real number, NO observer projection is needed to generate it.")

    ok = (fib_arch == 0 and fib_padic_counterex and pd_padic and pd_arch_minmiss > M * 0.1)
    print("\nVERDICT (the end-to-end non-unit QA -- data-driven):")
    print(f"  * There are TWO QA archetypes, selected by the generator's UNIT-NESS (Phase M):")
    print(f"    - UNIT generator (golden phi): order is ARCHIMEDEAN-local (Sturmian, verified 0")
    print(f"      mismatches) and NOT finite-local. Its order lives at the OBSERVER completion --")
    print(f"      it needs a real number. This is the whole golden arc, and it is exactly why")
    print(f"      mod-m destroys the quasicrystal.")
    print(f"    - NON-UNIT generator (period-doubling 2): order is p-ADIC-local (automatic,")
    print(f"      [v_2 even] exact) and NOT archimedean. Its order lives at the DISCRETE (A1")
    print(f"      mod-2^k) completion -- built from base-2 digits, NO observer projection.")
    print(f"  * So the discrete-sourced QA is real and end-to-end: for a non-unit generator the")
    print(f"    QA's own finite arithmetic IS the source of a genuine aperiodic order, with the")
    print(f"    archimedean readout an optional derived shadow -- the exact reverse of the golden QA.")
    print(f"  * HONEST consequence: Theorem NT holds the discrete layer primary, but the GOLDEN QA's")
    print(f"    characteristic order is observer-sited (archimedean) -- a precise structural fact,")
    print(f"    not a flaw. A NON-UNIT QA is the archetype in which the order is genuinely native to")
    print(f"    the discrete layer. Both are shadows of the one global adelic object; the generator's")
    print(f"    unit-ness picks which completion (observer vs discrete) hosts the order.")
    print(f"\n  STATUS: {'TWO ARCHETYPES CHARACTERIZED -- end-to-end discrete-sourced QA exhibited' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
