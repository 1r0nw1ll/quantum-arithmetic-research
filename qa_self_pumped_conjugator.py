#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=loop_gain_and_amplitude (real, Theorem NT); QA phase layer is integer {1..m}; qa_neg = phase conjugation. No external pump supplied."
"""
QA SELF-PUMPED phase conjugator — the internal-reflection ("cat") conjugator, the
self-starting cousin of cert [518]'s EXTERNALLY-pumped FWM mirror.

[518] SUPPLIES two pump beams: fwm(pf, pb, s) = qa_mod(pf + pb - s), and with a
supplied conjugate pump pb = qa_neg(pf) the output is qa_neg(s). A SELF-PUMPED mirror
(Feinberg 1982, "self-pumped continuous-wave phase conjugator using internal
reflection"; Cronin-Golomb et al. 1984) supplies NO external pump: the incident beam
fans and internally reflects, generating its OWN counter-propagating pump pair, and
above a reflectivity threshold the loop self-oscillates into the phase conjugate.

The key that makes this work (and the reason it needs no external reference):

  fwm(p, qa_neg(p), s) = qa_mod(p + qa_neg(p) - s) = qa_mod(-s) = qa_neg(s)   FOR ALL p.

The FWM output is INDEPENDENT of the pump value p — only the *conjugate-pair
condition* matters, and the loop geometry (internal reflection: the return beam is
qa_neg of the forward loop field) enforces that condition automatically. So whatever
self-pump the loop settles on, the output is the correct conjugate.

QA/Theorem-NT split: the phase layer (pump phase, conjugate, signal) is integer in
{1..m}; qa_neg is the standard involution. The loop GAIN g and AMPLITUDE A are
observer-layer reals (the self-oscillation threshold), never QA state. No external
pump is provided at any point.

Demonstrated (run it):
  [1] PUMP_INDEPENDENCE   fwm(p, qa_neg(p), s) == qa_neg(s) for ALL p, s (exhaustive)
  [2] SELF_STARTING       from a random noise seed pump that WANDERS each step, the
                          phase output stays locked to qa_neg(s) (pump-independence in
                          action) while the loop amplitude self-builds
  [3] THRESHOLD           logistic loop gain A' = g*A/(1+A): below g=1 the mirror is
                          OFF (A -> 0, no conjugation); above, it self-oscillates
                          (A -> g-1) -- the self-pumped reflectivity threshold
  [4] SELF_PUMPED_DC      aberrate s by a phase screen phi, self-pumped-conjugate,
                          return through the SAME phi -> qa_neg(s), with NO external
                          reference beam (the [518] distortion-correction property,
                          now self-referenced)
"""
from __future__ import annotations
import numpy as np

M = 24


def qa_mod(x):
    return ((int(x) - 1) % M) + 1


def qa_add(a, b):
    return qa_mod(a + b)


def qa_neg(a):
    return qa_mod(-a)


def fwm(pf, pb, s):
    """FWM phase-sum: two pumps + conjugated signal."""
    return qa_mod(pf + pb - s)


def self_pumped_step(p_loop, s):
    """One pass of the self-pumped loop: internal reflection makes the counter-pump
    qa_neg(p_loop); FWM of the (self-generated) conjugate pump pair with the signal.
    Returns the conjugate output (phase). No external pump enters."""
    p_back = qa_neg(p_loop)                 # internal reflection = phase conjugation
    return fwm(p_loop, p_back, s)           # == qa_neg(s), independent of p_loop


def run():
    rng = np.random.default_rng(518)

    # [1] PUMP_INDEPENDENCE — exhaustive
    ok1 = all(fwm(p, qa_neg(p), s) == qa_neg(s) for p in range(1, M + 1) for s in range(1, M + 1))
    print("[1] PUMP_INDEPENDENCE: fwm(p, qa_neg(p), s) == qa_neg(s) for ALL p,s  "
          f"({M*M} cases): {ok1}")

    # [2] SELF_STARTING — the self-pump WANDERS from a noise seed; output stays locked
    print("\n[2] SELF_STARTING: random wandering self-pump, output vs qa_neg(s)")
    n_trials, n_steps = 200, 40
    locked = 0
    for _ in range(n_trials):
        s = int(rng.integers(1, M + 1))
        p = int(rng.integers(1, M + 1))          # noise-seed self-pump
        out_locked = True
        A = 0.01                                  # tiny noise amplitude seed
        g = 1.8                                   # above threshold
        for _ in range(n_steps):
            p = qa_add(p, int(rng.integers(0, M)))   # self-pump wanders (fanning noise)
            c = self_pumped_step(p, s)               # phase output
            A = g * A / (1.0 + A)                     # loop amplitude self-builds
            if c != qa_neg(s):
                out_locked = False
        locked += int(out_locked and A > 0.1)
    print(f"    {locked}/{n_trials} trials: output stayed == qa_neg(s) through a wandering "
          f"self-pump AND amplitude self-built (>0.1)")

    # [3] THRESHOLD — self-oscillation reflectivity threshold.
    # A' = g*A/(1+A) has fixed points A=0 (all g) and A*=g-1 (g>1). For g<=1, A=0 is
    # the only non-negative fixed point and A monotonically decays to it; at the
    # critical g=1 the decay is marginal (A_n ~ 1/n, slow but -> 0). Classify by the
    # ANALYTIC fixed point, not a finite-iteration cutoff.
    print("\n[3] THRESHOLD: loop amplitude A' = g*A/(1+A), seed A0=0.01")
    print(f"    {'gain g':>7s} {'A(2000)':>9s} {'A*':>6s} {'mirror':>10s}")
    for g in (0.5, 0.9, 1.0, 1.1, 1.5, 2.0):
        A = 0.01
        for _ in range(2000):
            A = g * A / (1.0 + A)
        Astar = max(g - 1.0, 0.0)
        state = "ON" if g > 1.0 else ("threshold" if g == 1.0 else "off")
        print(f"    {g:7.2f} {A:9.5f} {Astar:6.2f} {state:>10s}")
    print("    -> self-pumped reflectivity threshold at g_c = 1.0: g>1 self-oscillates")
    print("       (A*=g-1>0); g<=1 decays to A=0 (no conjugation); g=1 is the marginal edge.")

    # [4] SELF_PUMPED_DC — distortion correction with NO external reference
    print("\n[4] SELF_PUMPED_DISTORTION_CORRECTION (aberrate by phi, self-conjugate, "
          "return through same phi):")
    ok4 = True
    for s in range(1, M + 1):
        for phi in range(1, M + 1):
            aberrated = qa_add(s, phi)                 # forward through the screen
            conj = self_pumped_step(int(rng.integers(1, M + 1)), aberrated)  # self-pumped, random pump
            returned = qa_add(conj, phi)               # back through the SAME screen
            if returned != qa_neg(s):
                ok4 = False
    print(f"    return == qa_neg(s) for ALL s,phi (any self-pump, no external beam): {ok4}")

    print("\nCONCLUSION: the self-pumped QA conjugator produces qa_neg(s) with NO external")
    print("pump -- the internal reflection self-generates a conjugate pump pair and the FWM")
    print("output is pump-INDEPENDENT, so any self-selected pump conjugates correctly. It")
    print("self-starts above a reflectivity threshold (g_c=1) and carries the [518]")
    print("distortion-correction property, self-referenced. Distinct from [518] (external)")
    print("and from [519] (a resonator driven by stored patterns): here the pump is internal.")
    return ok1 and (locked > 0.95 * n_trials) and ok4


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
