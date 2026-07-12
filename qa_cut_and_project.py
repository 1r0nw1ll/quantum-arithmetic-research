#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=cut-and-project physical/internal projections (Theorem NT); QA layer = the INTEGER golden orbit / integer lattice Z^2 under the golden element M (A1); physical x_par and internal x_perp coordinates are observer-layer readouts. No float QA state."
# RT1_OBSERVER_FILE: eigenvectors, physical/internal projections, tile lengths, windows are observer-layer readouts, not QA state.
"""
Phase L: does QA DYNAMICS generate the quasicrystal, or only carry the algebra? Bridging the
arc's frontier (algebra-yes, dynamics-not-yet), and explaining Phase H's mod-m null.

The golden element M=[[0,1],[1,1]] is BOTH the QA orbit generator ((b,e)->(e,b+e)) AND the
Fibonacci INFLATION matrix. Its two eigenvalues are the two spaces of a cut-and-project:
    phi (expanding)  -> PHYSICAL space E_par  (where the quasicrystal lives)
    psi=-1/phi (contracting) -> INTERNAL space E_perp = the acceptance WINDOW.
Crucially E_perp is the GALOIS CONJUGATE coordinate (phi->psi), which is exactly the QA
OBSERVER PROJECTION (Theorem NT): the contracting/internal direction is what QA calls the
observer/boundary. So M supplies EVERYTHING -- lattice action, physical space, and window --
from its own eigenstructure. This makes the quasicrystal a DYNAMICAL object of QA, not just
a shared algebra:
  [1] cut-and-project Z^2 through M's eigenspaces -> the Fibonacci chain (2 tiles, ratio phi).
  [2] M acts as the INFLATION: x_par -> phi*x_par, x_perp -> psi*x_perp; the model set is
      M-invariant (a self-similar FIXED POINT of the QA golden dynamics).
  [3] the QA golden orbit's internal coordinate = -psi^k*(2+psi)/sqrt5 -> 0: the orbit is
      CONFINED to the window by the contraction -> the QA orbit IS a model-set point sequence.
  [4] mod-m reduction (the QA finite-alphabet A1 layer) DESTROYS this: phi,psi survive only
      as finite-field RESIDUES -- they lose their archimedean magnitude, so |psi|<1 (the
      contraction) is meaningless in F_m -> no contraction -> the internal coordinate reaches
      the far side of the torus, the window is lost, the orbit is merely Pisano-periodic.
      THIS is why Phase H (mod P) saw generic Weil behavior, not golden order.

So: QA dynamics DOES generate the quasicrystal -- in the UNREDUCED (observer-layer, Z[phi])
inflation; the acceptance window is the Theorem-NT observer projection (the psi/Galois
direction); and reducing mod m is precisely the "measurement" that collapses it. Bridges
algebra->dynamics and unifies Theorem NT with the golden arc.
"""
from __future__ import annotations
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
PSI = (1.0 - np.sqrt(5.0)) / 2.0                  # = -1/phi, the contracting eigenvalue


def star_coords(n, m):
    """Physical and internal (Galois-conjugate) projections of a Z^2 point under M."""
    return n + PHI * m, n + PSI * m               # x_par, x_perp (= Galois conjugate of x_par)


def cut_and_project(rng_n, window):
    """Accept Z^2 points whose INTERNAL coord lies in the window; return sorted physical
    positions -> the 1D quasicrystal."""
    lo, hi = window
    pos = []
    for n in range(-rng_n, rng_n + 1):
        for m in range(-rng_n, rng_n + 1):
            xpar, xperp = star_coords(n, m)
            if lo <= xperp < hi:
                pos.append(xpar)
    return np.array(sorted(pos))


def fibonacci_word(n_iter):
    """Fibonacci substitution word L->LS, S->L (inflation matrix = M) for comparison."""
    w = "L"
    for _ in range(n_iter):
        w = "".join({"L": "LS", "S": "L"}[c] for c in w)
    return w


def run():
    print("Phase L: does QA dynamics GENERATE the quasicrystal? (cut-and-project via M)\n")

    # ===== [0] M's eigenstructure = the two cut-and-project spaces =====
    M = np.array([[0.0, 1.0], [1.0, 1.0]])
    evals, evecs = np.linalg.eig(M)
    print(f"[0] golden element M=[[0,1],[1,1]] eigenvalues: {sorted(evals, reverse=True)}")
    print(f"    phi={PHI:.4f} EXPANDING -> physical space E_par (quasicrystal lives here)")
    print(f"    psi={PSI:.4f} CONTRACTING -> internal space E_perp = acceptance WINDOW")
    print(f"    E_perp coord = n+psi*m = GALOIS CONJUGATE of the physical n+phi*m = the QA")
    print(f"    observer projection (Theorem NT): the contracting/internal direction.")

    # ===== [1] cut-and-project -> Fibonacci chain =====
    # window = projection of one unit cell onto E_perp: corners {0,1,psi,1+psi} -> [psi, 1)
    window = (PSI, 1.0)
    pos = cut_and_project(60, window)
    gaps = np.round(np.diff(pos), 4)
    gvals = np.array(sorted({g for g in gaps if g > 1e-3}))
    ratio = gvals[-1] / gvals[0] if len(gvals) >= 2 else float("nan")
    # tile word (L=long gap, S=short) vs the Fibonacci substitution word
    L = gvals[-1]
    word = "".join("L" if abs(g - L) < 1e-3 else "S" for g in gaps)
    fib = fibonacci_word(9)
    # compare on the common overlap (allow a global offset by scanning a small shift)
    match = max(sum(a == b for a, b in zip(word[s:], fib)) / max(len(fib), 1)
                for s in range(6))
    print(f"\n[1] cut-and-project Z^2 (window=[psi,1), the unit-cell projection):")
    print(f"    {len(pos)} points, {len(gvals)} distinct tile lengths {list(np.round(gvals,4))}, "
          f"ratio {ratio:.4f} = {'phi (FIBONACCI QUASICRYSTAL)' if abs(ratio-PHI)<0.02 else 'non-golden'}")
    print(f"    tile word matches the Fibonacci substitution word at {match:.0%} "
          f"(same L->LS,S->L order = same chain as Phase I)")

    # ===== [2] M is the INFLATION: x_par->phi*x_par, x_perp->psi*x_perp; model set M-invariant
    test = [(3, 5), (5, 8), (2, 3), (8, 13)]
    infl_ok = True
    for (n, m) in test:
        xpar, xperp = star_coords(n, m)
        nn, mm = m, n + m                          # M applied: (n,m)->(m,n+m)
        xpar2, xperp2 = star_coords(nn, mm)
        if not (abs(xpar2 - PHI * xpar) < 1e-6 and abs(xperp2 - PSI * xperp) < 1e-6):
            infl_ok = False
    # M-invariance of the window: psi*[psi,1) subset [psi,1) ?
    win_inv = PSI * 1.0 >= PSI and PSI * PSI < 1.0 and PSI * PSI >= PSI
    print(f"\n[2] M acts as the INFLATION (the QA dynamics): x_par->phi*x_par, x_perp->psi*x_perp")
    print(f"    verified on {test}: {infl_ok}; window psi*[psi,1) subset [psi,1): {win_inv}")
    print(f"    -> the model set is a self-similar FIXED POINT of iterating M (the golden orbit).")

    # ===== [3] the QA golden orbit is CONFINED to the window by the contraction =====
    a, b = 1, 1
    xperps = []
    for _ in range(20):
        xperps.append(a + PSI * b)                 # internal coord of orbit point (F_k, F_{k+1})
        a, b = b, a + b
    xperps = np.array(xperps)
    confined = np.all((xperps >= PSI) & (xperps < 1.0)) and abs(xperps[-1]) < 1e-3
    print(f"\n[3] QA golden orbit (F_k,F_{{k+1}}) internal coord = -psi^k*(2+psi)/sqrt5 -> 0:")
    print(f"    |x_perp| decays {abs(xperps[0]):.3f} -> {abs(xperps[-1]):.2e}; all in window [psi,1): "
          f"{confined} -> the QA orbit IS a model-set point sequence (dynamically confined).")

    # ===== [4] mod-m DESTROYS it: MEASURE that the internal coordinate no longer confines =====
    # over Z the internal coord collapses to ~2 values near 0 (a tiny window). mod m, build the
    # finite-field internal coordinate (F_k + psi_m*F_{k+1}) mod m with psi_m a root of
    # x^2-x-1 in F_m, and measure its COVERAGE of the torus over one Pisano period. If it fills
    # a large fraction, the window is gone; if it stayed confined, this would FALSIFY the claim.
    # over Z the internal coord CONTRACTS toward 0: its max |value| is a small bounded window
    # (here <1, i.e. << the physical scale phi^k). The confinement is that it never travels far.
    z_reach = float(np.max(np.abs(xperps)))                          # Z: farthest the internal coord gets
    print(f"\n[4] mod-m reduction (QA A1 finite-alphabet layer) collapses it (MEASURED):")
    print(f"    over Z the internal coord contracts: max|x_perp| = {z_reach:.3f} (bounded window, ->0).")
    destroyed = []
    for m in (11, 29, 101):
        r = next(s for s in range(m) if (s * s) % m == 5 % m)        # sqrt(5) in F_m (m = +-1 mod 5)
        inv2 = pow(2, m - 2, m)
        psi_m = ((1 - r) * inv2) % m                                 # contracting eigenvalue's F_m image
        aa, bb, period, reach = 1, 1, 0, 0
        while True:
            v = (aa + psi_m * bb) % m
            centered = v - m if v > m // 2 else v                    # nearest-0 representative
            reach = max(reach, abs(centered))
            aa, bb = bb, (aa + bb) % m
            period += 1
            if (aa, bb) == (1, 1):
                break
        reach_frac = reach / (m // 2)                                # 1 = travels to the far side
        filled = reach_frac > 0.5                                    # real, falsifiable: NO contraction
        destroyed.append(filled)
        print(f"    m={m:3d}: psi_m={psi_m} in F_{m} (finite residue, no |psi|<1); Pisano period "
              f"{period}; internal coord reaches {reach}/{m//2} = {reach_frac:.0%} to the far side "
              f"-> {'NO contraction, window DESTROYED' if filled else 'still confined (!)'}")
    print(f"    -> mod m keeps phi/psi ONLY as residues (no archimedean |psi|<1 in F_m): the")
    print(f"       contraction is gone, so the internal coordinate reaches the far side of the")
    print(f"       torus and no window exists. Exactly why Phase H (mod P) saw generic Weil order.")

    ok = (abs(ratio - PHI) < 0.02 and len(gvals) == 2 and infl_ok and win_inv
          and confined and all(destroyed))
    print("\nVERDICT (algebra->dynamics bridge, data-driven):")
    print(f"  * QA DYNAMICS generates the quasicrystal: iterating M (the golden orbit) IS the")
    print(f"    Fibonacci inflation, and M's two eigenvalues supply BOTH cut-and-project spaces --")
    print(f"    physical (phi) and the acceptance WINDOW (psi). The Fibonacci chain is the fixed")
    print(f"    point ({len(gvals)} tiles ratio {ratio:.3f}=phi, {match:.0%} match to the substitution).")
    print(f"  * The window = internal space = psi = GALOIS CONJUGATE = the Theorem-NT OBSERVER")
    print(f"    projection. The contracting eigenvalue confines the orbit (x_perp->0): the QA")
    print(f"    orbit is dynamically a model set. Observer projection and acceptance window are")
    print(f"    the SAME direction -- unifying Theorem NT with the golden/quasicrystal arc.")
    print(f"  * mod-m (the A1 finite-alphabet layer) DESTROYS it: phi/psi survive as residues but")
    print(f"    lose |psi|<1 (no contraction), so the orbit is merely Pisano-periodic and the")
    print(f"    internal coord reaches the far side -- the QA-native EXPLANATION of Phase H's null:")
    print(f"    golden order lives in the UNREDUCED observer-layer inflation, mod-m is the collapse.")
    print(f"\n  STATUS: {'BRIDGE HOLDS -- QA dynamics generates the quasicrystal' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
