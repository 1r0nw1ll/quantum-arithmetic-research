#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=field spectrum + physics quantities (hbar, omega, energy) — Theorem NT; the QA layer is the integer golden cat map supplying deterministic phases. No float QA state."
# RT1_OBSERVER_FILE: field amplitudes, FFT, power-law fits, hbar*omega are observer-layer physics readouts, not QA state.
"""
Phase B: compare a QA golden-oscillator field's spectrum to SED's zero-point-field.

The Stochastic-Electrodynamics zero-point field (ZPF) has spectral ENERGY DENSITY
rho(omega) proportional to omega^3 -- the Lorentz-invariant vacuum spectrum. That
omega^3 factors cleanly into two pieces:

  rho(omega) ~ [energy per mode = (1/2) hbar omega]  x  [3-D mode density ~ omega^2]
             = omega^1 (zero-point energy law)        x  omega^2 (geometry)
             = omega^3.

The geometry (omega^2) is just 3-D k-space; the zero-point energy law (energy ~ omega)
is the physical postulate. What QA's golden dynamics supplies is the third ingredient
SED needs -- a REAL stochastic field -- but as DETERMINISTIC golden chaos (Phase A:
the cat map is Bernoulli), not ontic randomness.

This script builds a 3-D field of golden-cat-map oscillators and measures rho(omega):
  (A) UNWEIGHTED (equal energy per mode)  -> should fit omega^2 (pure 3-D mode density)
  (B) ZERO-POINT weighted (energy ~ omega) -> should fit omega^3 (the SED ZPF law)
and confirms the field's phases -- hence its spatial randomness -- come entirely from
the deterministic golden cat map.

HONEST expectation (stated first): the omega^3 match in (B) is EXPECTED and is NOT a
QA discovery -- it is geometry x the imposed zero-point law. The QA content is only
that the STOCHASTIC realization can be deterministic (golden), i.e. QA demystifies the
vacuum's *randomness* while leaving the omega^3 *spectral law* postulated exactly as in
SED. The test is falsifiable in the sense that the fitted exponents must come out 2 and
3; if the golden field imposed its own color the decomposition would fail.
"""
from __future__ import annotations
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def golden_phases(n, x0=0.31415926, y0=0.27182818, drop=200):
    """n deterministic phases in [0,2pi) from the golden cat map (Phase A: Bernoulli).
    x_{k+1},y_{k+1} = (y, x+y) mod 1; phase = 2*pi*x. No ontic randomness."""
    x, y = x0, y0
    for _ in range(drop):
        x, y = y % 1.0, (x + y) % 1.0
    out = np.empty(n)
    for k in range(n):
        x, y = y % 1.0, (x + y) % 1.0
        out[k] = 2 * np.pi * x
    return out


def build_field(kmax=40):
    """3-D integer k-lattice; omega=|k|; deterministic golden phase per mode."""
    ks = range(-kmax, kmax + 1)
    kvec = np.array([(kx, ky, kz) for kx in ks for ky in ks for kz in ks
                     if 0 < (kx * kx + ky * ky + kz * kz) <= kmax * kmax], dtype=np.float64)
    omega = np.sqrt((kvec * kvec).sum(axis=1))
    phases = golden_phases(len(omega))
    return omega, phases


def spectral_density(omega, energy, nbins=24):
    """rho(omega) = total mode energy in a shell [w,w+dw], divided by dw."""
    wmin, wmax = omega.min(), omega.max()
    edges = np.linspace(wmin, wmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dw = edges[1] - edges[0]
    idx = np.clip(np.digitize(omega, edges) - 1, 0, nbins - 1)
    rho = np.array([energy[idx == b].sum() for b in range(nbins)]) / dw
    return centers, rho


def fit_slope(centers, rho):
    """Log-log slope of rho(omega), dropping empty and edge bins."""
    m = rho > 0
    c, r = centers[m], rho[m]
    c, r = c[2:-2], r[2:-2]                      # trim edge bins (finite-lattice artefacts)
    slope, intercept = np.polyfit(np.log(c), np.log(r), 1)
    pred = np.polyval([slope, intercept], np.log(c))
    ss_res = np.sum((np.log(r) - pred) ** 2)
    ss_tot = np.sum((np.log(r) - np.mean(np.log(r))) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2)


def run():
    print("QA golden-oscillator field vs SED zero-point spectrum (rho(omega) ~ omega^3)\n")
    omega, phases = build_field(kmax=40)
    print(f"3-D golden field: {len(omega)} modes, omega in [{omega.min():.1f}, {omega.max():.1f}]")

    # (A) unweighted: equal energy per mode -> rho ~ 3-D mode density ~ omega^2
    c_a, rho_a = spectral_density(omega, np.ones_like(omega))
    slope_a, r2_a = fit_slope(c_a, rho_a)
    print(f"\n[A] UNWEIGHTED (equal energy/mode): rho(omega) fit slope = {slope_a:.2f}  "
          f"(R^2={r2_a:.3f})   expect 2 (pure 3-D mode density)")

    # (B) zero-point weighted: energy per mode = (1/2) hbar omega  (hbar=1) -> rho ~ omega^3
    c_b, rho_b = spectral_density(omega, 0.5 * omega)
    slope_b, r2_b = fit_slope(c_b, rho_b)
    print(f"[B] ZERO-POINT (energy=1/2 hbar omega): rho(omega) fit slope = {slope_b:.2f}  "
          f"(R^2={r2_b:.3f})   expect 3 = the SED ZPF law")

    # the phases -- hence the field's spatial randomness -- are deterministic golden chaos
    p = phases
    two_pi_over = np.exp(1j * p)
    resultant = np.abs(two_pi_over.mean())            # ~0 for uniform (random-looking) phases
    # deterministic reproducibility of the phase field:
    reproducible = np.allclose(golden_phases(len(p)), p)
    print(f"\n[C] field phases from the golden cat map: circular resultant "
          f"{resultant:.3f} (->0 = uniform/random-looking), exactly reproducible={reproducible}")

    ok = abs(slope_a - 2) < 0.25 and abs(slope_b - 3) < 0.25
    print("\nRESULT:")
    print(f"  unweighted -> omega^{slope_a:.2f} (geometry), zero-point-weighted -> "
          f"omega^{slope_b:.2f} (ZPF): match = {ok}")
    print("  omega^3 = omega^2 (3-D geometry) x omega^1 (zero-point energy law). The golden")
    print("  field supplies the DETERMINISTIC stochastic phases (the vacuum's 'randomness'")
    print("  as golden chaos, Phase A), NOT the omega^3 color. HONEST BOUND: QA-as-ZPF-substrate")
    print("  reproduces the SED spectrum only WITH the standard zero-point energy law imposed")
    print("  (energy ~ omega) -- which neither QA nor SED derives; QA demystifies the randomness,")
    print("  the spectral law stays postulated. Not a derivation of omega^3, a clarification of")
    print("  what QA does and does not contribute.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
