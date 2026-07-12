#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=2D diffraction/structure-factor physics (Theorem NT); QA layer = integer index k of the golden orbit / integer combinations of the 5 wavevectors (A1); Vogel positions, structure factors, module radii are observer-layer readouts. No float QA state."
# RT1_OBSERVER_FILE: point positions, structure factors, radial/angular profiles, module radii are observer-layer readouts, not QA state.
"""
Phase J: 2D phyllotaxis diffraction -- the golden angle made an observer-layer field, and
the "Whittaker superposition over a circle of directions" made literal.

Two DISTINCT golden 2D objects (Phase I found the 1D Fibonacci chain = pure-point Bragg):
  (V) VOGEL SUNFLOWER phyllotaxis: points at (r_k, th_k) = (sqrt(k), k*theta_gold),
      theta_gold = 2*pi/phi^2 ~ 137.5deg. The golden angle here optimizes ANGULAR
      UNIFORMITY (why plants use it -- no gaps, uniform coverage). PREDICTION: its
      diffraction is approximately ISOTROPIC diffuse rings (low angular contrast,
      intensity ~N like a liquid, NOT ~N^2 Bragg); a rational divergence angle instead
      gives anisotropic SPOKES. Its Fibonacci signature is in REAL space: the visible
      spiral arms (parastichies) = consecutive Fibonacci numbers, because the best
      rational approximants of theta_gold/2pi = 1/phi^2 have Fibonacci denominators.
  (W) WHITTAKER n-WAVE: superpose n plane waves over a circle of directions at 2*pi*j/n.
      For n=5 (72deg) this is the DECAGONAL (Penrose) quasicrystal: its Fourier module is
      Z[phi]-valued -> radial shells scaled by phi (golden self-similar), 10-fold symmetric.
      THIS is where sharp golden Bragg + 5-fold order live -- the 2D face of the icosian/
      E8 golden structure (H3/H4 Coxeter, golden quaternions = E8).

So the honest picture refines Phase I: "phyllotaxis" (uniformity-optimal, isotropic) and
the "5-fold quasicrystal" (sharp golden Bragg) are BOTH golden but different diffraction
classes; the E8/icosahedral connection lives with the Whittaker 5-wave, not the sunflower.
Both are OBSERVER-layer (Theorem NT); the QA layer is the integer index k / integer
combinations, and the golden element M / Q(sqrt5) is the shared generator.
"""
from __future__ import annotations
import itertools

import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = 2.0 * np.pi / (PHI * PHI)          # ~137.5077 deg -- most uniform divergence


def sq_sum(phases):
    """|sum exp(i*phases)|^2 without '**' (S1)."""
    z = np.exp(1j * phases).sum()
    return z.real * z.real + z.imag * z.imag


# ---------------- (V) Vogel sunflower ----------------
def vogel_spiral(n, angle):
    """Vogel model: r_k=sqrt(k) (area-uniform), th_k=k*angle. Observer-layer positions of
    the integer-index (A1 k=1..n) golden orbit."""
    k = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(k)
    th = k * angle
    return r * np.cos(th), r * np.sin(th)


def parastichy_offsets(x, y, sample=400):
    """Index-offset to the nearest spatial neighbor, sampled ACROSS radii. Golden
    phyllotaxis -> offsets are consecutive Fibonacci numbers (convergents of 1/phi^2);
    the offset grows with radius, so a radial spread reveals the Fibonacci ladder."""
    n = len(x)
    ks = np.unique(np.geomspace(max(n // 16, 20), n - 2, sample).astype(int))
    offs = []
    for k in ks:
        dx = x - x[k]
        dy = y - y[k]
        d2 = dx * dx + dy * dy
        d2[k] = np.inf
        offs.append(abs(int(np.argmin(d2)) - k))
    vals, counts = np.unique(offs, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [(int(vals[i]), int(counts[i])) for i in order[:6]]


def poisson_disk(n, radius, seed=3):
    """Random uniform points in a disk (observer-layer null, NOT QA state): isotropic but
    NOT hyperuniform -- S(q)->1 at small q (unsuppressed density fluctuations)."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    return r * np.cos(th), r * np.sin(th)


def small_q_mean(x, y, q_band, n_ang=64):
    """Angularly-averaged S(q) over a small-q band (below the first ring). Hyperuniform ->
    suppressed (<< Poisson); random -> ~1."""
    ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    vals = []
    for q in q_band:
        ph = np.outer(q * np.cos(ang), x) + np.outer(q * np.sin(ang), y)
        z = np.exp(1j * ph).sum(axis=1)
        vals.append(np.mean((z.real * z.real + z.imag * z.imag) / len(x)))
    return float(np.mean(vals))


def diffraction_polar(x, y, q_radii, n_ang=180):
    """S(q) on a polar grid: rows=|q|, cols=angle. Returns S[radii, angles]."""
    n = len(x)
    ang = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    S = np.empty((len(q_radii), n_ang))
    for i, qr in enumerate(q_radii):
        qx = qr * np.cos(ang)
        qy = qr * np.sin(ang)
        ph = np.outer(qx, x) + np.outer(qy, y)       # (n_ang, N)
        z = np.exp(1j * ph).sum(axis=1)
        S[i] = (z.real * z.real + z.imag * z.imag) / n
    return S, ang


# ---------------- (W) Whittaker 5-wave / decagonal module ----------------
def fivefold_module(nmax=3):
    """Fourier module of 5 plane waves at 72deg (Whittaker over a circle of 5 directions):
    all integer combinations sum_j n_j * (cos,sin)(2pi j/5)."""
    gens = [(np.cos(2 * np.pi * j / 5), np.sin(2 * np.pi * j / 5)) for j in range(5)]
    vx, vy = [], []
    for combo in itertools.product(range(-nmax, nmax + 1), repeat=5):
        vx.append(sum(c * g[0] for c, g in zip(combo, gens)))
        vy.append(sum(c * g[1] for c, g in zip(combo, gens)))
    return np.array(vx), np.array(vy)


def rotational_symmetry(vx, vy, fold):
    """Fraction of module points mapped back into the module by a 2*pi/fold rotation."""
    a = 2.0 * np.pi / fold
    rx = vx * np.cos(a) - vy * np.sin(a)
    ry = vx * np.sin(a) + vy * np.cos(a)
    pts = set(zip(np.round(vx, 3), np.round(vy, 3)))
    hit = sum(((round(x, 3), round(y, 3)) in pts) for x, y in zip(rx, ry))
    return hit / len(vx)


def run():
    print("Phase J: 2D phyllotaxis diffraction (Vogel) + Whittaker 5-wave golden module\n")
    n = 4000
    xg, yg = vogel_spiral(n, GOLDEN_ANGLE)
    xr, yr = vogel_spiral(n, 2.0 * np.pi / 6.0)              # genuine RATIONAL control (60deg -> spokes)
    xp, yp = poisson_disk(n, np.sqrt(n))                     # isotropic-but-random null

    # --- (A) real-space Fibonacci parastichies (golden phyllotaxis signature) ---
    par = parastichy_offsets(xg, yg)
    fibs = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233}
    n_fib = sum(c for o, c in par if o in fibs)
    n_tot = sum(c for _, c in par)
    print("[A] Vogel real-space parastichies (nearest-neighbor index offsets, across radii):")
    print(f"    top offsets {[(o, c) for o, c in par]}")
    print(f"    -> {100 * n_fib // max(n_tot,1)}% land on Fibonacci numbers "
          f"{sorted(o for o, c in par if o in fibs)} (consecutive Fibonacci = the parastichy ladder)")

    # --- (B) 2D diffraction: golden = ISOTROPIC + HYPERUNIFORM; rational = spokes; random = neither ---
    q_radii = np.linspace(0.2, 12.0, 90)
    Sg, ang = diffraction_polar(xg, yg, q_radii)
    Sr, _ = diffraction_polar(xr, yr, q_radii)
    Sp, _ = diffraction_polar(xp, yp, q_radii)
    contrast_g = float(np.median(Sg.std(axis=1) / (Sg.mean(axis=1) + 1e-9)))
    contrast_r = float(np.median(Sr.std(axis=1) / (Sr.mean(axis=1) + 1e-9)))
    contrast_p = float(np.median(Sp.std(axis=1) / (Sp.mean(axis=1) + 1e-9)))
    q_star = q_radii[int(np.argmax(Sg.mean(axis=1)[5:]) + 5)]
    # hyperuniformity: mean S over a small-q band (below the first ring ~3.5)
    q_band = np.array([0.35, 0.5, 0.7, 0.9, 1.1])
    hu_g = small_q_mean(xg, yg, q_band)
    hu_p = small_q_mean(xp, yp, q_band)
    print(f"\n[B] 2D diffraction (structure factor):")
    print(f"    angular contrast (low = ISOTROPIC rings):  golden={contrast_g:.2f}  "
          f"60deg-rational={contrast_r:.2f}  Poisson={contrast_p:.2f}")
    print(f"    -> golden is isotropic like random, UNLIKE the rational spokes "
          f"({'confirmed' if contrast_r > contrast_g else 'not'}); brightest ring |q|={q_star:.2f}")
    print(f"    small-q S (low = HYPERUNIFORM, suppressed density fluctuations):  "
          f"golden={hu_g:.2f}  Poisson={hu_p:.2f}")
    print(f"    -> golden suppresses low-q fluctuations {hu_p/max(hu_g,1e-6):.1f}x below Poisson "
          f"({'HYPERUNIFORM hidden order' if hu_g < 0.6 * hu_p else 'not suppressed'}): the golden")
    print(f"    angle's 2D role is ISOTROPIC + HYPERUNIFORM order (not sharp Bragg) -- the")
    print(f"    stealthy uniformity that makes it the phyllotaxis (and isotropic-photonics) optimum.")

    # --- (C) Whittaker 5-wave decagonal module: golden-scaled shells + 10-fold ---
    vx, vy = fivefold_module(nmax=3)
    rad = np.hypot(vx, vy)
    shells = np.array(sorted({round(r, 3) for r in rad if r > 1e-6}))
    ratios = shells[1:] / shells[:-1]
    near_phi = shells[1:][np.abs(ratios - PHI) < 0.02]
    sym10 = rotational_symmetry(vx, vy, 10)
    sym5 = rotational_symmetry(vx, vy, 5)
    # find an explicit golden pair among shells
    gold_pairs = [(a, b) for a in shells for b in shells if a > 1e-6 and abs(b / a - PHI) < 0.01]
    print(f"\n[C] Whittaker 5-wave module (5 plane waves at 72deg over a circle of directions):")
    print(f"    {len(shells)} distinct radial shells; rotational symmetry: 10-fold={sym10:.2f}, "
          f"5-fold={sym5:.2f} -> {'DECAGONAL (10-fold)' if sym10 > 0.99 else 'lower'}")
    if gold_pairs:
        a, b = gold_pairs[len(gold_pairs) // 2]
        print(f"    radial shells are golden-scaled: e.g. {b:.3f}/{a:.3f} = {b/a:.4f} = phi "
              f"({len(gold_pairs)} shell pairs with ratio phi) -> module is Z[phi]-valued")
    print(f"    -> sharp golden Bragg + 5/10-fold order = the DECAGONAL quasicrystal, the 2D")
    print(f"       face of the icosian/E8 golden structure (H3/H4 Coxeter, golden quaternions).")

    # --- verdict ---
    isotropic = contrast_g < contrast_r
    hyperuniform = hu_g < 0.6 * hu_p
    golden_module = len(near_phi) > 0 and sym10 > 0.99
    print("\nVERDICT (2D golden diffraction, data-driven):")
    print(f"  * VOGEL phyllotaxis: real-space parastichies are Fibonacci "
          f"({sorted(o for o,c in par if o in fibs)}); its DIFFRACTION is ISOTROPIC "
          f"(contrast {contrast_g:.2f} vs rational-spoke {contrast_r:.2f}) and HYPERUNIFORM")
    print(f"    (small-q S {hu_g:.2f} vs Poisson {hu_p:.2f}). The golden angle's 2D role is stealthy")
    print(f"    isotropic uniformity, NOT sharp Bragg. {'CONFIRMED' if isotropic and hyperuniform else 'MIXED'}.")
    print(f"  * WHITTAKER 5-wave: the 'superposition over a circle of directions' at 72deg IS the")
    print(f"    decagonal quasicrystal -- {sym10:.0%} 10-fold, radial shells scaled by phi (Z[phi]).")
    print(f"    {'CONFIRMED' if golden_module else 'MIXED'}: this is the sharp golden Bragg + 5-fold")
    print(f"    order, the 2D projection of the icosian/E8 golden lattice.")
    print("  * Net: Phase I's 1D Fibonacci chain and this 2D decagonal module are the SAME golden")
    print("    module Z[phi]=O_Q(sqrt5) in 1 and 2 dimensions; the sunflower is the uniformity-")
    print("    optimal SAMPLING of the plane, a distinct (isotropic) golden object. Whittaker over")
    print("    a circle of 5 directions closes the loop to 5-fold/icosahedral/E8 -- observer-layer,")
    print("    M/Q(sqrt5)-generated, the same spine as the QA quaternion order.")

    _save_png(xg, yg, q_radii, Sg, vx, vy)
    return 0 if (isotropic and hyperuniform and golden_module) else 1


def _save_png(xg, yg, q_radii, Sg, vx, vy):
    """Optional figure: sunflower, its diffraction rings, and the 5-fold module."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # observer-layer plotting only
        print(f"\n[png] skipped ({exc})")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(xg[:1500], yg[:1500], s=3, c="goldenrod")
    ax[0].set_title("Vogel sunflower (golden angle)"); ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[1].imshow(np.log1p(Sg), aspect="auto", origin="lower", cmap="inferno",
                 extent=[0, 360, q_radii[0], q_radii[-1]])
    ax[1].set_title("diffraction S(|q|, angle): isotropic rings"); ax[1].set_xlabel("angle (deg)")
    ax[1].set_ylabel("|q|")
    ax[2].scatter(vx, vy, s=6, c="teal")
    ax[2].set_title("Whittaker 5-wave module (decagonal, Z[phi])"); ax[2].set_aspect("equal")
    ax[2].axis("off")
    fig.tight_layout()
    fig.savefig("qa_phyllotaxis_diffraction.png", dpi=110)
    print("\n[png] wrote qa_phyllotaxis_diffraction.png")


if __name__ == "__main__":
    raise SystemExit(run())
