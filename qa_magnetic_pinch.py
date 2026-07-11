#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=virial-scaling exponents (Theorem NT); QA layer = integer golden-orbit currents (directed flows on A1 {1..m}); coupling energies + equilibrium radii are observer-layer readouts. No float QA state."
# RT1_OBSERVER_FILE: pairwise sums, virial equilibria, power-law fits are observer-layer readouts, not QA state.
"""
Phase G (corrected): does an EM long-range attraction give the sub-extensive M~R that
Phase E needs -- correcting Phase F's bias toward mass-gravity, AND correcting my own
first attempt (Codex-caught: I used the 1/r POINT potential and mislabeled it the
"magnetic pinch"; parallel LINE currents have force ~1/r but a LOGARITHMIC potential).

Three couplings, done honestly:
  (N) QA-NATIVE resonance similarity (Phase F): non-decaying, R-independent -> extensive
      / super-extensive, no binding scale (recapped).
  (C) 1/r POINT attraction, U = -k/r (force ~1/r^2). This is the COULOMB (electrostatic)
      OR gravitational form -- NOT the magnetic one. Self-bound virial 2K=|U|, |U|~N^2/R,
      K=(3/2)NT -> R_eq ~ N -> M ~ R (SUB-extensive). Coulomb is electromagnetic, so a 1/r
      point EM attraction DOES give M~R -- Phase F's "needs gravity" was wrong. CAVEAT:
      electrostatic 1/r is Debye-SCREENED in a quasineutral plasma (which is exactly why
      plasma cosmology invokes the magnetic force instead).
  (M) MAGNETIC parallel-current pinch, U = +k*ln(r) (force ~1/r inward, attractive). The
      virial of a log potential is r*dU/dr = k = CONSTANT per pair -> 2K = k*N_pairs,
      R-INDEPENDENT: the Bennett pinch equilibrium is radius-independent (mu0 I^2 =
      8 pi N_L kT), scale-free -- NOT a clean M~R. So the specific magnetic pinch does
      not give M~R the way the 1/r point attraction does.

Honest verdict this reaches: the sub-extensive M~R holography needs (Phase E) is reachable
via a 1/r POINT attraction, which is EM (Coulomb) as much as gravity -- so Phase F's
gravity-only framing was a bias (the user's point). But (Codex's point) the specific
MAGNETIC pinch is the Bennett scale-invariant case, not M~R. And QA's NATIVE coupling
(similarity) is neither -- it is extensive. So the sub-extensive scaling requires IMPOSING
a 1/r point coupling; QA's own dynamics does not produce it. The coupling FORM is the crux.
"""
from __future__ import annotations
import numpy as np

M_MOD = 10007


def golden_orbit_positions(n, m=M_MOD):
    """n QA golden-orbit tuples as current elements; positions (b,e,d)/m in 3-D."""
    def qm(x):
        return ((x - 1) % m) + 1
    b, e = 1, 1
    pos = np.empty((n, 3))
    for k in range(n):
        pos[k] = (b / m, e / m, qm(b + e) / m)
        b, e = e, qm(b + e)
    pos = pos - pos.mean(axis=0)
    return pos / (np.linalg.norm(pos, axis=1).max() + 1e-12)     # unit ball


def pair_sum(pos, kernel, soft=1e-3):
    """sum_{i<j} kernel(r_ij)."""
    total = 0.0
    for i in range(len(pos) - 1):
        d = pos[i + 1:] - pos[i]
        r = np.sqrt((d * d).sum(axis=1) + soft * soft)
        total += kernel(r).sum()
    return float(total)


def run():
    print("Phase G (corrected): EM long-range attraction and the sub-extensive M~R\n")
    Ns = np.array([50, 100, 200, 400, 800, 1600, 3200], dtype=float)
    T = 1.0

    # (C) 1/r POINT (Coulomb/gravity) potential U=-k/r: |U(R)| = k*W_invr/R, virial R_eq
    # (M) MAGNETIC log potential U=+k ln r: virial term r*dU/dr = k (const) -> R-independent
    Req_coulomb, W_logvir = [], []
    for N in Ns:
        pos = golden_orbit_positions(int(N))
        W_invr = pair_sum(pos, lambda r: 1.0 / r)          # ~ N^2, gives U ~ 1/R
        Req_coulomb.append(W_invr / (3 * N * T))           # 2K=|U|: 3NT = k W/R
        # log-potential virial contribution sum_{i<j} r*dU/dr = sum k = k * n_pairs (R-free)
        W_logvir.append(pair_sum(pos, lambda r: np.ones_like(r)))   # = n_pairs (R-independent)
    Req_coulomb = np.array(Req_coulomb)

    mr_full = float(np.polyfit(np.log(Req_coulomb), np.log(Ns), 1)[0])
    mr_asy = float(np.polyfit(np.log(Req_coulomb[-4:]), np.log(Ns[-4:]), 1)[0])
    print("(C) 1/r POINT attraction  U=-k/r  [Coulomb (EM) OR gravity -- same form]:")
    print(f"    mass-size exponent M~R^{mr_full:.2f} (large-N asymptote {mr_asy:.2f})  "
          f"-> {'SUB-extensive (near BH M~R)' if mr_full < 2.5 else 'not sub-ext'}")
    print("    Coulomb is electromagnetic, so a 1/r POINT EM attraction gives M~R: Phase F's")
    print("    'needs GRAVITY' was a bias. CAVEAT: electrostatic 1/r is Debye-SCREENED in a")
    print("    quasineutral plasma (the reason plasma cosmology uses magnetic forces instead).")

    # (M) magnetic pinch: virial term is R-independent -> scale-free (Bennett)
    logvir_exp = float(np.polyfit(np.log(Ns), np.log(np.array(W_logvir)), 1)[0])
    print(f"\n(M) MAGNETIC parallel-current pinch  U=+k*ln(r)  [the actual pinch]:")
    print(f"    virial term sum r*dU/dr ~ N^{logvir_exp:.2f} and is R-INDEPENDENT -> the Bennett")
    print(f"    equilibrium is radius-INDEPENDENT (mu0 I^2 = 8 pi N_L kT), scale-free. So the")
    print(f"    magnetic pinch does NOT give a clean M~R the way the 1/r point attraction does.")

    print("\n(N) QA-NATIVE resonance similarity (Phase F): non-decaying, R-independent -> "
          "extensive/super-extensive (alpha>=0.95), no binding scale.")

    print("\nVERDICT (both corrections honored):")
    print(f"  - The sub-extensive M~R that holography needs IS reachable by a 1/r POINT")
    print(f"    attraction (M~R^{mr_full:.2f}, asymptote {mr_asy:.2f}) -- and 1/r is COULOMB")
    print("    (electromagnetic) as much as gravity. So Phase F's 'needs emergent gravity' was")
    print("    a bias: an EM point attraction gives M~R too. (Will's correction stands.)")
    print("  - BUT the specific MAGNETIC pinch is the log-potential Bennett case -- scale-free,")
    print("    radius-independent -- NOT a clean M~R. (Codex's correction stands; my first")
    print("    labeling of the 1/r kernel as 'magnetic' was wrong.)")
    print("  - And QA's NATIVE coupling (similarity) is NEITHER -- it is extensive. The")
    print("    sub-extensive scaling requires IMPOSING a 1/r point coupling; QA's own dynamics")
    print("    does not produce it. The coupling FORM is the crux, and it is put in by hand.")
    print("  Net: the honest missing ingredient is 'a QA-native reason for a 1/r (or otherwise")
    print("  binding) coupling FORM', not specifically 'gravity' and not specifically 'the pinch'.")
    print("  Plasma cosmology's viability at cosmic scales (Debye screening) remains a separate,")
    print("  genuinely open question this does not settle -- not dismissed, not proven.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
