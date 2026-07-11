#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=entropy/energy scaling physics — Theorem NT; the QA content is the discrete finite-alphabet voxel structure. No float QA state."
# RT1_OBSERVER_FILE: entropy counts, energy-scaling exponents, Bekenstein/BH relations are observer-layer physics readouts, not QA state.
"""
Phase E: does QA IMPLY holography (derive area-scaling), or only realize it (Phase D)?

't Hooft's decomposition of the holographic area-law: it needs TWO ingredients --
  (1) a fundamental DISCRETENESS / cutoff (finite DOF per Planck cell), and
  (2) the BLACK-HOLE entropy bound S <= A/4 (you cannot pack more entropy into a region
      than a black hole of that size, or it collapses).
Discreteness alone gives VOLUME-scaling DOF; the BH bound is what forces the reduction
to AREA.

The sharp, testable discriminator: the BH bound is really the statement that a black
hole's energy is SUB-EXTENSIVE, E ~ R (Schwarzschild M = R c^2 / 2G), so that
Bekenstein's S <= 2*pi R E / (hbar c) becomes S <= (area). A LOCAL theory whose energy
is EXTENSIVE (E ~ R^3, volume) cannot produce the area-law: its DOF and energy both
scale with volume. So: is QA's energy extensive or sub-extensive?

QA answer (Phase C): a region of size R holds modes up to a FIXED cutoff omega_max ~ 1/l
(the discreteness scale, independent of R), so its vacuum energy is
    E_QA ~ rho * R^3 ~ R^3   -- EXTENSIVE (volume), like any local discrete theory.
A black hole is E_BH ~ R -- SUB-EXTENSIVE. These are different scalings.

So QA does NOT contain the black-hole (sub-extensive, gravitational M~R) relation, and
therefore does NOT by itself imply the holographic area-law. QA supplies ingredient (1)
of 't Hooft's derivation (the discrete voxels), NOT ingredient (2) (the gravitational BH
bound). The honest answer to "does QA imply holography" is NO-alone: it is the
discreteness half. This script measures the two scalings explicitly and pinpoints the
missing gravitational ingredient (emergent M~R) as the real frontier.
"""
from __future__ import annotations
import math

import numpy as np

# ---- scales (SI) ----
c = 2.99792458e8
hbar = 1.054571817e-34
G = 6.67430e-11
l_P = math.sqrt(hbar * G / c ** 3)


def fit_exponent(Rs, Ys):
    """Power-law exponent d(log Y)/d(log R)."""
    return float(np.polyfit(np.log(Rs), np.log(Ys), 1)[0])


def run():
    print("Phase E: does QA imply holography (derive area-scaling)?\n")
    Rs = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6], dtype=float)     # region size in units of l

    # [1] QA naive entropy: N=(R/l)^3 voxels, each with m states -> S = N ln m  (VOLUME)
    m = 24
    S_qa = (Rs ** 3) * math.log(m)
    print(f"[1] QA entropy (m={m} states/voxel, N=(R/l)^3 voxels): S = N ln m")
    print(f"    exponent d(log S)/d(log R) = {fit_exponent(Rs, S_qa):.2f}  -> VOLUME-scaling (3)")

    # [2] holographic (BH) bound: S_max = A/(4 l^2) ~ (R/l)^2  (AREA)
    S_bh = math.pi * Rs ** 2                                        # ~ A/4 in l=1 units
    print(f"[2] Bekenstein-Hawking bound S_BH = A/(4 l^2) ~ (R/l)^2")
    print(f"    exponent = {fit_exponent(Rs, S_bh):.2f}  -> AREA-scaling (2)")
    print(f"    QA naive entropy EXCEEDS the BH bound by S_qa/S_BH ~ R/l "
          f"(exponent {fit_exponent(Rs, S_qa / S_bh):.2f}): the volume count over-saturates,")
    print(f"    so the true DOF must reduce to area -- the holographic argument.\n")

    # [3] THE DISCRIMINATOR: energy scaling. QA is EXTENSIVE; a black hole is SUB-EXTENSIVE
    #     QA vacuum energy in region R: fixed cutoff omega_max ~ 1/l -> E ~ rho * R^3
    E_qa = Rs ** 3                                                  # extensive (volume)
    #     black hole: E_BH = M c^2, Schwarzschild M = R c^2 / 2G -> E ~ R
    E_bh = Rs                                                       # sub-extensive (radius)
    print(f"[3] ENERGY scaling (the crux):")
    print(f"    QA vacuum energy (fixed cutoff omega_max~1/l): E ~ R^{fit_exponent(Rs, E_qa):.2f}  "
          f"-> EXTENSIVE (volume)")
    print(f"    black hole (Schwarzschild M~R):               E ~ R^{fit_exponent(Rs, E_bh):.2f}  "
          f"-> SUB-EXTENSIVE (radius)")
    print(f"    Bekenstein S <= 2*pi R E /(hbar c): with E~R (BH) -> S ~ R^2 = AREA;")
    print(f"    with E~R^3 (QA/local) -> S ~ R^4, which VIOLATES the bound -- so an extensive")
    print(f"    theory has no area-law. The area-law needs the BH's sub-extensive E~R.\n")

    print("VERDICT (honest, and it is a NEGATIVE that pinpoints the gap):")
    print("  QA does NOT imply the holographic area-law. QA is a LOCAL DISCRETE theory: its")
    print("  entropy and energy are EXTENSIVE (~volume, exponents 3 and 3), like any local")
    print("  theory. The area-law requires the BLACK-HOLE bound -- equivalently the")
    print("  gravitational, SUB-EXTENSIVE mass-radius relation M ~ R (Schwarzschild) that turns")
    print("  Bekenstein's S~RE into S~R^2. QA contains no M~R relation. So in 't Hooft's")
    print("  two-ingredient derivation of holography [discreteness + BH bound], QA supplies")
    print("  ONLY the discreteness (the finite voxels of Phases C-D); the gravitational BH")
    print("  bound is the missing half. 'Does QA imply holography?' -> NO, alone.")
    print("  THE REAL FRONTIER (what would close it): derive the black-hole / gravitational")
    print("  M~R (sub-extensive) scaling as an EMERGENT property of QA dynamics -- i.e. QA ->")
    print("  gravity. That is a genuine open program, not something these 4 phases reach; but")
    print("  it is now precisely located: the one missing ingredient is emergent M~R.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
