#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=vacuum-energy physics (hbar, omega, mode sums) — Theorem NT; the QA content is the DISCRETE modulus (A1 finite alphabet) acting as a UV cutoff. No float QA state."
# RT1_OBSERVER_FILE: omega, energy sums, power-law integrals are observer-layer physics readouts, not QA state.
"""
Phase C: does QA's structure give a NATURAL reason for the zero-point energy law
(energy ~ omega) and, more importantly, cure its divergence?

Honest split of what "energy ~ omega" is:
  * The omega-LINEARITY is kinematic -- energy is the rate of phase advance
    (E/hbar = omega = dphi/dt). QA supplies the variable naturally: a phase mode
    advancing by omega QA-units per path-time step (T1) has angular frequency ~ omega,
    so E = hbar_QA * omega. This is the Planck-Einstein relation, NOT a QA discovery --
    it is true of any phase oscillator. QA does not derive hbar or the 1/2 ordering.

What QA's AXIOMS actually add, that continuum QED lacks:
  (1) An off-zero SHIFT (a structural rhyme with the zero-point 1/2, NOT a derivation).
      QA's arithmetic is shifted off 0 -- states in {1,...,m}, qa_step=((x-1)%m)+1 is a
      +1 offset -- structurally like the oscillator energy being shifted off 0
      (n -> n+1/2). CAVEAT (this is only an analogy): A1 constrains the STATE, not the
      phase-ADVANCE. A nonzero state can stay constant (a zero-frequency / DC mode), so
      A1 does NOT forbid omega=0 and does NOT by itself impose an energy floor. This is
      not load-bearing; the real QA contribution is (2).
  (2) UV CUTOFF <-> DISCRETENESS (the real payoff). The continuum ZPF energy density
      rho(omega) ~ omega^3 makes the total vacuum energy DIVERGE: integral of omega^3
      diverges (the vacuum catastrophe / cosmological-constant problem, ~10^120 too big).
      QA is DISCRETE: on an m-point phase circle the largest distinguishable phase rate
      is floor(m/2) (Nyquist/aliasing), so omega_max ~ m. The QA vacuum energy is then
      FINITE: sum of (1/2)omega over 3-D modes with omega <= omega_max ~ omega_max^4 ~ m^4.
      This is exactly lattice regularization (the modulus is QA's UV scale).

This script shows: the modular-rotor dispersion is E ~ omega; the A1 floor is nonzero;
and the discrete cutoff makes the vacuum energy finite (~ m^4) where the continuum
diverges. Honest claim: QA does not DERIVE the zero-point energy, but its axioms
(A1 no-zero + finite modulus) map onto the never-zero floor and a UV cutoff that
REGULARIZES the vacuum-energy divergence -- QA as a regulator, a real and testable
contribution, not a from-nothing derivation.
"""
from __future__ import annotations
import numpy as np


def rotor_dispersion(m, omegas):
    """A QA phase mode advancing by 'omega' units per path-time step on the m-point
    circle: evolution operator U|j> = exp(2i*pi*omega*j/m)|j>. Its fundamental angular
    frequency (phase advance per step) is 2*pi*omega/m ~ omega. Return that rate."""
    return np.array([2 * np.pi * (w % m) / m for w in omegas])


def nyquist_omega_max(m):
    """Largest DISTINGUISHABLE phase-advance rate on an m-point circle (aliasing)."""
    return m // 2


def vacuum_energy_discrete(m, c=1.0):
    """Total zero-point energy of a 3-D field of QA phase modes, cut off at the
    discreteness scale omega_max = floor(m/2): sum over integer k with 0<|k|<=k_max of
    (1/2) hbar omega_k, omega_k = c|k|, hbar=1. Returns (n_modes, total_energy)."""
    kmax = nyquist_omega_max(m)
    if kmax > 60:                                  # cap the explicit 3-D sum for speed
        # analytic: sum ~ integral 4*pi k^2 * (1/2) c k dk = (pi/2) c kmax^4 (shell approx)
        return None, float(0.5 * np.pi * c * kmax ** 4)
    ks = range(-kmax, kmax + 1)
    e = 0.0; n = 0
    for kx in ks:
        for ky in ks:
            for kz in ks:
                k2 = kx * kx + ky * ky + kz * kz
                if 0 < k2 <= kmax * kmax:
                    e += 0.5 * c * (k2 ** 0.5); n += 1
    return n, e


def vacuum_energy_continuum(omega_cut, c=1.0):
    """Continuum ZPF: integral_0^cut 4*pi (omega/c)^2 * (1/2) hbar omega / c domega
    ~ (pi/(2 c^3)) omega_cut^4.  Diverges as omega_cut -> infinity."""
    return float(0.5 * np.pi / c ** 3 * omega_cut ** 4)


def run():
    print("Phase C: QA discreteness as the zero-point-energy regulator\n")

    # [1] dispersion E ~ omega (kinematic, via the modular rotor)
    m = 24
    ws = [1, 2, 3, 5, 8]
    rates = rotor_dispersion(m, ws)
    slope = np.polyfit(ws, rates, 1)[0]
    print(f"[1] modular-rotor dispersion (m={m}): phase-rate vs omega is linear, "
          f"slope={slope:.4f} (= 2*pi/m) -> E = hbar_QA * omega")
    print("    (kinematic: energy is the phase-advance rate under T1 path-time; not a QA")
    print("     derivation of hbar -- true of any phase oscillator).")

    # [2] A1 off-zero shift -- a structural rhyme, NOT a floor derivation
    print("\n[2] A1 off-zero shift (structural rhyme, NOT load-bearing): QA's arithmetic is")
    print("    shifted off 0 (states {1..m}; qa_step=((x-1)%m)+1 is a +1 offset), like the")
    print("    oscillator energy shifted off 0 (n -> n+1/2). CAVEAT: A1 constrains the STATE,")
    print("    not the phase-ADVANCE -- a nonzero state can be constant (omega=0), so A1 does")
    print("    NOT forbid the zero-frequency mode nor impose an energy floor. Analogy only.")

    # [3] the payoff: discrete cutoff makes the vacuum energy FINITE; continuum diverges
    print("\n[3] UV cutoff from discreteness: omega_max = floor(m/2); vacuum energy finite ~ m^4")
    print(f"    {'m':>7} {'omega_max':>10} {'n_modes':>10} {'E_vacuum (QA, finite)':>22} {'~ m^4 scaling':>14}")
    prev = None
    for m in (24, 48, 96, 240, 2400, 24000):
        n, e = vacuum_energy_discrete(m)
        nstr = str(n) if n is not None else "(shell approx)"
        ratio = f"{e/prev:.1f}x" if prev else "-"
        print(f"    {m:>7} {nyquist_omega_max(m):>10} {nstr:>10} {e:>22.3e} {ratio:>14}")
        prev = e
    print("    (each 10x in m -> ~10^4x in E_vacuum: E ~ m^4, FINITE at every finite m.)")

    print("\n[4] continuum QED (m -> infinity): E_vacuum = (pi/(2 c^3)) omega_cut^4 -> INFINITY")
    for cut in (1e2, 1e4, 1e6, 1e12):
        print(f"    omega_cut={cut:.0e} -> E_vacuum={vacuum_energy_continuum(cut):.3e}  (diverges as cut->inf)")

    print("\nCONCLUSION:")
    print("  QA does NOT derive the zero-point energy law from nothing (the omega-linearity is")
    print("  the kinematic phase-rate; hbar and the 1/2 are not derived; and the A1 off-zero")
    print("  shift is only a structural rhyme, not an energy floor). What QA genuinely supplies")
    print("  is the one thing continuum QED lacks: a UV CUTOFF at the modulus scale (discreteness")
    print("  = lattice regularization), which renders the vacuum energy FINITE (~ m^4) where the")
    print("  continuum")
    print("  omega^3 spectrum diverges (the vacuum catastrophe). QA is a natural REGULATOR of the")
    print("  zero-point field, with the cutoff set by its own discreteness scale m -- a precise,")
    print("  honest, testable contribution, not a from-nothing derivation of E = hbar*omega.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
