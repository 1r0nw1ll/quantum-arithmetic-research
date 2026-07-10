#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=geometry_and_medium_to_phase, state_alphabet=mod24_A1_compliant; coherent-sum focus metric is observer-layer readout"
"""
QA Time-Reversal Focusing -- the distortion-correction theorem run IN REVERSE.

Cert [518] (distortion correction): a phase-conjugated wave sent back through the
SAME medium undoes the distortion at a receiver. Run the same operator (qa_neg =
the standard involution / phase conjugation = time reversal) the OTHER way and it
FOCUSES energy back onto the source through a scattering medium:

  source at x0 -> propagate (phase G(i,x0)) + scatter (medium H_i) -> array records
  r_i = qa_add(G(i,x0), H_i, s);  TIME-REVERSE each element r_i* = qa_neg(r_i);
  re-emit toward candidate x -> field_i(x) = qa_add(G(i,x), H_i, r_i*).

Because the medium term H_i appears on both record and re-emit it CANCELS:
  field_i(x) = qa_add( G(i,x) - G(i,x0) - s )   (mod m).
At x = x0 every element carries the same phase (-s) -> they add coherently -> a
focal peak, THROUGH the medium, at the source. Elsewhere the phases scatter.

Two claims, both falsifiable here:
  (A) time reversal refocuses at the source through an arbitrary phase screen;
  (B) SAME-MEDIUM SPECIFICITY: re-emitting through a DIFFERENT screen H' destroys
      the focus (the [518] fingerprint, now for focusing instead of correction).

A1/S2/Theorem-NT: all phases are integers in {1,...,m}; geometry and medium are
observer-layer inputs (boundary crossed once, in); the coherent-sum focus metric is
an observer-layer readout of the QA phase field (boundary crossed once, out).
"""
from __future__ import annotations
import numpy as np

M = 24


def qa_mod(x):
    return ((np.asarray(x, np.int64) - 1) % M) + 1


def qa_add(*xs):
    s = np.zeros_like(np.asarray(xs[0], np.int64))
    for x in xs:
        s = s + np.asarray(x, np.int64)
    return qa_mod(s)


def qa_neg(x):
    return qa_mod(-np.asarray(x, np.int64))


def prop_phase(elem_pos, x):
    """Observer projection: continuous geometry -> integer QA phase (crossed once).
    Path-length phase from array element position to location x, quantized to {1..m}."""
    d = np.abs(elem_pos - x)                      # continuous distance (observer)
    return qa_mod(np.rint(d * (M / 8.0)).astype(np.int64))   # -> integer phase


def make_screen(n, rng):
    return rng.integers(1, M + 1, n)             # per-element scattering phase (observer)


def focus_metric(field_phases):
    """Observer-layer readout: coherent-sum magnitude of the QA phase field
    (|sum exp(2pi i phase/m)| / n).  1.0 = perfectly focused, ~0 = scattered."""
    z = np.exp(2j * np.pi * (field_phases - 1) / M)
    return float(np.abs(z.mean()))


def run():
    rng = np.random.default_rng(42)
    N = 64                       # array elements
    positions = np.linspace(0.0, 10.0, N)
    x0 = 3.7                     # source location
    s = 5                        # source emission phase
    locations = np.linspace(0.0, 10.0, 201)

    print(f"QA TIME-REVERSAL FOCUSING  (m={M}, {N} elements, source at x0={x0})\n")

    H = make_screen(N, rng)                       # the scattering medium
    # record: array element i sees source through propagation + medium
    r = qa_add(prop_phase(positions, x0), H, np.full(N, s))
    r_tr = qa_neg(r)                              # TIME REVERSE (phase conjugate)

    # [1] re-emit through the SAME medium; scan candidate locations for the focus
    print("[1] Focus profile (re-emit through matched medium H):")
    focus = []
    for x in locations:
        field = qa_add(prop_phase(positions, x), H, r_tr)   # medium H cancels analytically
        focus.append(focus_metric(field))
    focus = np.array(focus)
    peak_x = locations[int(np.argmax(focus))]
    print(f"  peak focus {focus.max():.3f} at x={peak_x:.2f}  (source x0={x0})  "
          f"-> refocuses at source: {abs(peak_x - x0) < 0.1}")
    # background (median away from the source) vs the peak
    away = focus[np.abs(locations - x0) > 1.0]
    print(f"  focal gain: peak {focus.max():.3f} vs background median {np.median(away):.3f}")

    # [2] Same-medium specificity: re-emit through a DIFFERENT screen H'
    print("\n[2] Same-medium specificity (re-emit through matched H vs mismatched H'):")
    Hp = make_screen(N, rng)
    f_match = max(focus_metric(qa_add(prop_phase(positions, x), H, r_tr)) for x in locations)
    f_mis = max(focus_metric(qa_add(prop_phase(positions, x), Hp, r_tr)) for x in locations)
    print(f"  matched medium   : peak focus {f_match:.3f}")
    print(f"  mismatched medium: peak focus {f_mis:.3f}")
    print(f"  specificity gap  : {f_match - f_mis:+.3f}  "
          f"({'FOCUS ONLY THROUGH MATCHED MEDIUM' if f_match - f_mis > 0.3 else 'no specificity'})")

    # [3] It is literally the [518] operator: field reduces to G(i,x)-G(i,x0)-s (H gone)
    print("\n[3] Medium-cancellation check (the involution qa_neg does the work):")
    x = x0
    direct = qa_add(prop_phase(positions, x), H, r_tr)
    reduced = qa_add(prop_phase(positions, x), qa_neg(prop_phase(positions, x0)), np.full(N, qa_neg(s)))
    print(f"  field through medium == field with H analytically cancelled: "
          f"{np.array_equal(direct, reduced)}")
    print("\nTime reversal (qa_neg) = phase conjugation = the standard involution; deployed to")
    print("FOCUS through clutter, not just to CORRECT at a receiver. Same operator, reversed use.")


if __name__ == "__main__":
    run()
