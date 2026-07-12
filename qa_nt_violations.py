#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=archimedean shadows/absolute values (Theorem NT); QA layer = the GLOBAL integer golden orbit and its finite-place (mod m) reductions (A1); this file DEMONSTRATES what an NT violation is by exhibiting an illegitimate float->finite cast, it does not commit one as QA state. No float QA state feeds the discrete layer."
# RT1_OBSERVER_FILE: floats, casts, absolute values here are observer-layer objects used to demonstrate the firewall, not QA state.
"""
Phase O: the projection layer projects from SOMEWHERE -- what, and what is an NT violation
specifically?

Phase N put the observer (archimedean) and QA-discrete (finite) layers on equal footing as
the two kinds of place of Q(sqrt5), tied by the product formula. But a projection needs a
DOMAIN. The answer: both layers are shadows of the SAME global object --

    the GLOBAL number-theoretic element (a Z[phi] number = the golden orbit over Z, the
    icosian ring element / the diagonal point in the adele of Q(sqrt5)).

    observer layer  = its ARCHIMEDEAN shadow(s)  (real embeddings: physical + window)
    QA discrete A1  = its FINITE-place shadow(s)  (the mod-m reductions)

Neither shadow is prior; the GLOBAL object is the source. Theorem NT names the firewall
between the two shadows but was silent on where the projection projects FROM -- it is the
global Z[phi]/icosian element. That is the missing piece.

This makes 'NT violation' precise and testable. A legitimate discrete-layer value is the
REDUCTION of a global object (a ring homomorphism Z[phi] -> Z[phi]/m). An NT VIOLATION is
producing a finite-place value that is NOT sourced from a global object -- specifically,
copying/casting the ARCHIMEDEAN shadow into a finite place, which fails because the two
shadows are related by the product formula, NOT by identity. Demonstrated here:

  [A] the source: the global integer F_k projects to BOTH an archimedean float (phi^k/sqrt5)
      and a finite residue (F_k mod m). Both are honest shadows of the one global object.
  [B] legitimate = a ring HOMOMORPHISM: reduction mod m respects the recurrence (the finite
      shadow evolves by the same +,x law); the real embedding respects it too.
  [C] the NT violation (T2-b, float x modulus -> int cast) DEMONSTRATED: casting the
      archimedean shadow to a finite place agrees only by luck of rounding, then diverges to
      WRONG residues (float precision) -- and float x mod is not even a ring homomorphism.
      The cast fabricates finite-place data with no global source.
  [D] the criterion + the full list of specific NT violations, all ONE thing: a finite-place
      value not sourced from a global Z[phi] object. And the test that keeps cut-and-project
      LEGITIMATE (its window is M's own psi-shadow, Z[phi]-valued -- a global source -- not
      external float data).
"""
from __future__ import annotations
import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PSI = (1.0 - math.sqrt(5.0)) / 2.0
SQRT5 = math.sqrt(5.0)


def run():
    print("Phase O: where the projection projects FROM, and what an NT violation IS\n")
    m = 997

    # ===== [A] the SOURCE: one global Z[phi] element, exact shadows at every place =====
    print("[A] the projection's source = a GLOBAL Z[phi] element phi^k = F_{k-1}+F_k*phi (an exact")
    print("    integer pair). It has TWO archimedean shadows (the two real embeddings of Q(sqrt5))")
    print("    and one finite shadow -- ALL exact, ALL from the same integer pair:")
    print(f"    {'k':>3} {'phi^k=(F_{k-1},F_k)':>20} {'sigma1=phi^k (physical)':>24} "
          f"{'sigma2=psi^k (window)':>22} {'residue mod '+str(m):>16}")
    p, q = 1, 0                                          # phi^0 = 1 + 0*phi
    for k in range(0, 6):
        s1 = p + q * PHI                                 # real embedding sqrt5->+  = phi^k (exact)
        s2 = p + q * PSI                                 # real embedding sqrt5->-  = psi^k (exact)
        print(f"    {k:>3} {'('+str(p)+','+str(q)+')':>20} {s1:>24.4f} {s2:>22.6f} "
              f"{'('+str(p % m)+','+str(q % m)+')':>16}")
        p, q = q, p + q                                  # x phi: (p+q*phi)*phi = q+(p+q)*phi
    print("    F_k = (sigma1 - sigma2)/sqrt5 combines BOTH archimedean places (sigma2=psi^k, the")
    print("    contracting WINDOW, is the term a Binet float drops). Observer (archimedean sigma1,")
    print("    sigma2) and QA-discrete (finite residue) are SIBLING shadows of the one global Z[phi]")
    print("    element; neither is prior. The global element is where both project FROM.")

    # ===== [B] legitimate projection = a ring HOMOMORPHISM (reduction respects +,x) =====
    a, b = 0, 1
    hom_ok = True
    prev2, prev1 = 0 % m, 1 % m
    for _ in range(2, 500):
        a, b = b, a + b
        if (b % m) != (prev1 + prev2) % m:              # finite shadow must satisfy the same recurrence
            hom_ok = False
        prev2, prev1 = prev1, b % m
    print(f"\n[B] legitimate = ring HOMOMORPHISM: the finite shadow obeys F_{{k+1}}=F_k+F_{{k-1}} mod {m}")
    print(f"    for 500 steps: {hom_ok}. Reduction Z[phi]->Z[phi]/m respects +,x (an honest projection).")

    # ===== [C] the NT VIOLATION (T2-b, float x modulus -> int cast) DEMONSTRATED =====
    a, b, fp, first_div = 0, 1, 1.0, None
    for k in range(0, 140):
        cast = int(round(fp / SQRT5)) % m               # cast the ARCHIMEDEAN shadow into a finite place
        if k >= 1 and cast != a % m and first_div is None:
            first_div = k
            bad_cast, true_res = cast, a % m
        a, b, fp = b, a + b, fp * PHI
    print(f"\n[C] the NT VIOLATION -- T2-b (float x modulus -> int cast), DEMONSTRATED:")
    print(f"    take the observer's Binet FLOAT readout phi^k/sqrt5 (an approximation of sigma1/sqrt5,")
    print(f"    NOT the exact global object) and reduce it mod {m}: it agrees by luck of rounding")
    print(f"    until k={first_div}, then float64 precision fails and it gives a WRONG residue:")
    print(f"      k={first_div}: float-readout residue = {bad_cast}  vs  true F_k mod {m} = {true_res}  (worse after).")
    print(f"    the deeper reason: the float readout is not a ring homomorphism -- rounding does not")
    print(f"    respect + or x, so it cannot carry the exact residue the way reduction of the global")
    print(f"    integer does. The cast FABRICATES a finite-place value with no global source; the two")
    print(f"    shadows connect ONLY through the global integer, never by casting one into the other.")

    # ===== [D] criterion + the specific NT violations + the legitimacy test =====
    # legitimacy test for an archimedean SELECTOR (cut-and-project window): is it Z[phi]-valued
    # (M's own shadow, a global source)? test via M-inflation invariance of the window.
    window_edges = (PSI, 1.0)                             # M's psi-shadow window -> Z[phi]-valued edges
    infl_lo, infl_hi = sorted((PSI * window_edges[0], PSI * window_edges[1]))
    window_M_invariant = infl_lo >= window_edges[0] - 1e-9 and infl_hi <= window_edges[1] + 1e-9
    ext_edges = (1.0 / math.pi, 1.0)                      # arbitrary transcendental window (no global source)
    e_lo, e_hi = sorted((PSI * ext_edges[0], PSI * ext_edges[1]))
    ext_M_invariant = e_lo >= ext_edges[0] - 1e-9 and e_hi <= ext_edges[1] + 1e-9
    print(f"\n[D] THE SPECIFIC NT VIOLATIONS (all one thing: a finite-place value NOT sourced from a")
    print(f"    global Z[phi] object -- place-mixing without a global source):")
    print(f"    * T2-b  float x modulus -> int cast   (DEMONSTRATED above): archimedean shadow -> finite,")
    print(f"            no global source, not a homomorphism.")
    print(f"    * T2    observer output -> QA input    (feedback): re-injecting a readout (an archimedean")
    print(f"            shadow) as a discrete input -- couples places without a global object.")
    print(f"    * T2-D  continuous/stochastic -> QA    (no-stochastic-QA): an archimedean MEASURE (a real")
    print(f"            distribution) as QA state -- it has no finite-place counterpart.")
    print(f"    * S2    float QA state                 storing the discrete (finite-place) object as an")
    print(f"            archimedean approximation -- loses the exact residue.")
    print(f"    LEGITIMACY TEST for an archimedean selector (e.g. a cut-and-project window): is it the")
    print(f"    global object's own Z[phi]-shadow? M's window [psi,1) is M-inflation-invariant: "
          f"{window_M_invariant} (legitimate);")
    print(f"    an external transcendental window [1/pi,1) is not: {ext_M_invariant} (would be a violation).")

    ok = (hom_ok and first_div is not None and first_div < 140
          and window_M_invariant and not ext_M_invariant)
    print("\nVERDICT (the source of the projection, and NT violations made precise):")
    print(f"  * WHERE it projects FROM: the GLOBAL Z[phi]/icosian element (the diagonal point in the")
    print(f"    adele of Q(sqrt5)). Observer (archimedean) and QA-discrete (finite) are its two kinds")
    print(f"    of shadow, tied by the product formula (Phase N). Theorem NT names the firewall; the")
    print(f"    global object is the domain it was silent about.")
    print(f"  * WHAT an NT violation IS, specifically: producing a finite-place (discrete) value that")
    print(f"    is NOT the reduction of a global object -- i.e. casting/copying an archimedean shadow")
    print(f"    into a finite place (T2-b, demonstrated: wrong residues, not a homomorphism), or")
    print(f"    feeding a readout back (T2), a continuous distribution in (T2-D), or storing float")
    print(f"    state (S2). All ONE thing: place-mixing without a global source.")
    print(f"  * The firewall is CROSSED LEGITIMATELY only through the global object: a discrete input")
    print(f"    must be a reduction of a Z[phi] element (a ring homomorphism), and an archimedean")
    print(f"    selector is allowed only if it is the global object's own Z[phi]-shadow (M's window),")
    print(f"    not external float data. That is the exact line between projection and violation.")
    print(f"\n  STATUS: {'NT VIOLATIONS CHARACTERIZED -- source = global Z[phi] object, violation = place-mixing without it' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
