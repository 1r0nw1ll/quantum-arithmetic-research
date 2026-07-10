#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=diagnostic_comparison; E8 alignment / HI are observer-layer readouts (Theorem NT); no QA state mutation here"
# RT1_OBSERVER_FILE: diagnostic only; the sin() input signal is an observer-layer projection, not QA state.
"""
Task 2: fix qa_core.e8_alignment to the icosian-grounded E8, and WATCH how the
Harmonic Index (HI) changes across a QASystem run. Honest before/after diagnostic.

Background (task b, qa_e8_icosian_grounding.sage): the historical
qa_core.e8_alignment did NOT use E8 -- it scored mean|cos| to a single hardcoded
Fibonacci vector [1,1,2,3,0,0,0,0] (not a root). The fix (qa_core/metrics.py) aligns
QA tuples to the 120 icosian 600-cell directions (the binary icosahedral 2I = QA's
genuine E8 over Q(sqrt5); see qa_icosian_order.py), keeping the old as
e8_alignment_legacy.

WHAT THIS DIAGNOSTIC FINDS (run it) -- a genuine trade-off, honestly:
  1. The fix RAISES HI (legacy ~0.79 -> icosian ~0.96, ~+21%), and both are
     near-CONSTANT over a run (tiny std). Since the QA loss ~ 0 (the ellipse
     identity a^2 = d^2+2de+e^2 holds), HI = alignment throughout.
  2. DISCRIMINATION IS THE CATCH. The LEGACY metric, though mislabeled (it is NOT
     E8 -- it is alignment to the Fibonacci direction [1,1,2,3]), actually WEAKLY
     DISCRIMINATES: structured Fibonacci/Cosmos-orbit tuples score higher than
     random (~0.88 vs ~0.79), precisely because QA orbits ARE Fibonacci. The
     grounded ICOSIAN metric is mathematically correct (real E8 over Q(sqrt5)) but
     does NOT discriminate: the 120-vertex 600-cell covers directions finely enough that the max-cosine
     readout saturates, so random
     tuples align as well as structured ones (~0.96 vs ~0.96).
  => So "E8 alignment" was a MISNOMER either way. What weakly worked was Fibonacci-
     direction alignment (the legacy, honestly relabeled); genuine E8/icosian
     alignment does not measure QA harmonicity. The fix buys mathematical legitimacy
     at the cost of the (weak) discrimination the legacy metric had. The Harmonic
     Index needs rethinking, not just a corrected root vector. Honest exposure.
"""
from __future__ import annotations
import numpy as np
import qa_core.engine as eng
from qa_core import QASystem
from qa_core.metrics import e8_alignment, e8_alignment_legacy, harmonic_loss, qa_tuples


def _run(align_fn, seed=42, steps=200):
    eng.e8_alignment = align_fn
    np.random.seed(seed)
    sys = QASystem(num_nodes=64, modulus=24, coupling=0.1, noise_base=0.05,
                   noise_annealing=0.99, signal_injection_strength=0.2, signal_mode="final")
    sys.run_simulation(steps, 0.3 * np.sin(np.linspace(0, 20, steps)), progress=False)
    h = sys.history
    return np.array(h["e8_alignment"]), np.array(h["hi"]), np.array(h["loss"])


def _discrimination(align_fn, m=24, n=300):
    """Do structured (Cosmos orbit) tuples align better than random ones?"""
    rng = np.random.default_rng(0)
    # structured: Fibonacci Cosmos orbits
    b = np.array([1, 1, 2, 3, 5, 8, 13, 21, 10, 7, 17, 24], dtype=np.int64)
    e = np.roll(b, -1)
    struct = align_fn(qa_tuples(np.tile(b, 8), np.tile(e, 8), m))
    rb, re = rng.integers(1, m + 1, n), rng.integers(1, m + 1, n)
    rand = align_fn(qa_tuples(rb, re, m))
    return struct, rand


def run():
    aL, hiL, lossL = _run(e8_alignment_legacy)
    aN, hiN, lossN = _run(e8_alignment)
    print("QA E8-ALIGNMENT / HARMONIC-INDEX: LEGACY (single Fib vector) vs NEW (icosian 600-cell)\n")
    print(f"{'metric':16s} {'mean':>8s} {'std':>8s} {'final':>8s} {'range':>8s}")
    for name, arr in [("align legacy", aL), ("align icosian", aN),
                      ("HI legacy", hiL), ("HI icosian", hiN)]:
        print(f"{name:16s} {arr.mean():8.4f} {arr.std():8.4f} {arr[-1]:8.4f} {arr.max()-arr.min():8.4f}")
    print(f"\n[1] HI shift from the fix: {hiL.mean():.4f} -> {hiN.mean():.4f} "
          f"({100*(hiN.mean()-hiL.mean())/hiL.mean():+.1f}%)")
    print(f"[2] loss over the run ~ 0 (ellipse identity holds): mean {lossN.mean():.2e} "
          f"-> HI = alignment")
    print(f"[3] near-constant: alignment std legacy {aL.std():.4f}, icosian {aN.std():.4f} "
          f"(both ~flat)")
    sL, rL = _discrimination(e8_alignment_legacy)
    sN, rN = _discrimination(e8_alignment)
    print(f"[4] discrimination (structured Cosmos orbit vs random tuples):")
    print(f"      legacy (Fib vector): struct {sL:.4f}  random {rL:.4f}  -> "
          f"{'struct HIGHER (weakly discriminates)' if sL > rL else 'no discrimination'}")
    print(f"      icosian (600-cell) : struct {sN:.4f}  random {rN:.4f}  -> "
          f"{'struct HIGHER' if sN > rN else 'random >= struct (does NOT discriminate)'}")
    print("\nCONCLUSION: a genuine trade-off. The legacy metric was MISLABELED (not E8; it")
    print("aligns to the Fibonacci direction [1,1,2,3]) but WEAKLY DISCRIMINATED QA structure")
    print("(orbits are Fibonacci). The icosian fix is mathematically CORRECT (real E8 over")
    print("Q(sqrt5), +21% HI) but does NOT discriminate (600-cell saturates). Since loss~0,")
    print("HI = alignment either way. 'E8 alignment' is a misnomer; genuine E8 alignment does")
    print("not measure QA harmonicity. The Harmonic Index needs rethinking, not just a fix.")


if __name__ == "__main__":
    run()
