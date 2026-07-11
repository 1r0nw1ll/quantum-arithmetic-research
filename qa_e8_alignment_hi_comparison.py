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

WHAT THIS DIAGNOSTIC FINDS (run it) -- a genuine trade-off, then the redesign:
  1. The icosian fix RAISES the E8-ALIGNMENT metric (legacy ~0.79 -> icosian ~0.96,
     ~+21%), and both are near-CONSTANT over a run. Since the QA loss ~ 0 (the
     ellipse identity a^2 = d^2+2de+e^2 holds), the OLD HI (= alignment*exp(-loss))
     tracked only this near-constant.
  2. DISCRIMINATION IS THE CATCH. The LEGACY metric, though mislabeled (it is NOT
     E8 -- it is alignment to the Fibonacci direction [1,1,2,3]), actually WEAKLY
     DISCRIMINATES: structured Fibonacci/Cosmos-orbit tuples score higher than
     random (~0.88 vs ~0.79), precisely because QA orbits ARE Fibonacci. The
     grounded ICOSIAN metric is mathematically correct (real E8 over Q(sqrt5)) but
     does NOT discriminate: the 120-vertex 600-cell covers directions finely enough that the max-cosine
     readout saturates, so random
     tuples align as well as structured ones (~0.96 vs ~0.96).
  => So "E8 alignment" was a MISNOMER either way; genuine E8/icosian alignment does
     not measure QA harmonicity.
  3. THE REDESIGN (done): HI is now driven by golden-orbit COHERENCE
     (harmonic_coherence = the Kuramoto order parameter of the QA orbit phases), NOT
     E8 alignment. It DISCRIMINATES cleanly (synced ~1 vs random ~0.06) and TRACKS
     self-organization (rises ~0.5 -> ~0.98 under strong coupling, stays ~0.08 under
     weak). e8_alignment (now the correct icosian metric) is still tracked as a
     geometric readout. HI = coherence * exp(-0.1*loss).
"""
from __future__ import annotations
import numpy as np
import qa_core.engine as eng
from qa_core import QASystem
from qa_core.metrics import (e8_alignment, e8_alignment_legacy, harmonic_coherence,
                             harmonic_loss, qa_tuples)


def _run_alignment(align_fn, seed=42, steps=200):
    """Run QASystem tracking the e8_alignment metric (what HI USED to be driven by)."""
    eng.e8_alignment = align_fn
    np.random.seed(seed)
    sys = QASystem(num_nodes=64, modulus=24, coupling=0.1, noise_base=0.05,
                   noise_annealing=0.99, signal_injection_strength=0.2, signal_mode="final")
    sys.run_simulation(steps, 0.3 * np.sin(np.linspace(0, 20, steps)), progress=False)
    h = sys.history
    return np.array(h["e8_alignment"]), np.array(h["loss"])


def _run_hi(coupling, seed=42, steps=150):
    """Run QASystem and return the REDESIGNED (coherence-driven) HI trajectory."""
    np.random.seed(seed)
    sys = QASystem(num_nodes=128, modulus=24, coupling=coupling, noise_base=0.02,
                   noise_annealing=0.98, signal_injection_strength=0.0, signal_mode="final")
    sys.run_simulation(steps, np.zeros(steps), progress=False)
    return np.array(sys.history["hi"])


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
    # PART A -- the E8-ALIGNMENT metric, which USED to drive HI (before the redesign)
    aL, lossL = _run_alignment(e8_alignment_legacy)
    aN, lossN = _run_alignment(e8_alignment)
    print("PART A: the E8-ALIGNMENT metric (the OLD HI driver): legacy vs icosian fix\n")
    print(f"{'metric':16s} {'mean':>8s} {'std':>8s} {'final':>8s} {'range':>8s}")
    for name, arr in [("align legacy", aL), ("align icosian", aN)]:
        print(f"{name:16s} {arr.mean():8.4f} {arr.std():8.4f} {arr[-1]:8.4f} {arr.max()-arr.min():8.4f}")
    print(f"\n[1] the icosian fix raises the alignment: {aL.mean():.4f} -> {aN.mean():.4f} "
          f"({100*(aN.mean()-aL.mean())/aL.mean():+.1f}%)")
    print(f"[2] loss over the run ~ 0 (ellipse identity holds): mean {lossN.mean():.2e} "
          f"-> when HI was = alignment*exp(-loss), HI tracked only this near-constant.")
    print(f"[3] near-constant: alignment std legacy {aL.std():.4f}, icosian {aN.std():.4f} "
          f"(both ~flat over the run)")
    # E8 metrics: tested on a Fibonacci orbit trajectory vs random (their notion of
    # 'structure' is alignment to a direction). Coherence: tested on a SYNCHRONIZED
    # population vs spread (its notion is self-organization) -- the right test for HI.
    sL, rL = _discrimination(e8_alignment_legacy)
    sN, rN = _discrimination(e8_alignment)
    rng = np.random.default_rng(1)
    synced = qa_tuples(np.full(200, 8), np.full(200, 5), 24)          # all nodes at one state
    spread = qa_tuples(rng.integers(1, 25, 200), rng.integers(1, 25, 200), 24)
    sC, rC = harmonic_coherence(synced, 24), harmonic_coherence(spread, 24)
    print(f"[4] discrimination -- what each metric can tell apart:")
    print(f"      legacy E8 (Fib vec)  : Fib-orbit {sL:.4f}  random {rL:.4f}  -> "
          f"{'weakly discriminates' if sL > rL else 'no discrimination'}")
    print(f"      icosian E8 (600-cell): Fib-orbit {sN:.4f}  random {rN:.4f}  -> "
          f"{'struct HIGHER' if sN > rN else 'does NOT discriminate'}")
    print(f"      NEW coherence (HI)   : synced   {sC:.4f}  spread {rC:.4f}  -> "
          f"{'DISCRIMINATES CLEANLY (self-organization)' if sC > rC + 0.3 else 'weak'}")

    # PART B -- the REDESIGNED HI (coherence-driven) genuinely tracks self-organization
    hi_strong, hi_weak = _run_hi(0.4), _run_hi(0.02)
    print("\nPART B: the REDESIGNED HI (now coherence-driven) through the real QASystem:")
    print(f"[5] STRONG coupling: HI {hi_strong[0]:.3f} -> {hi_strong[-1]:.3f} "
          f"(rises {hi_strong[-1]-hi_strong[0]:+.3f}: nodes SELF-ORGANIZE)")
    print(f"    WEAK   coupling: HI {hi_weak[0]:.3f} -> {hi_weak[-1]:.3f} "
          f"(stays low: disordered)")

    print("\nCONCLUSION: 'E8 alignment' was a misnomer -- legacy (mislabeled Fibonacci vector)")
    print("weakly discriminated, icosian (correct E8) does not discriminate at all (600-cell")
    print("saturates), and loss~0 meant the old HI (= alignment) was near-constant. REDESIGNED:")
    print("HI is now golden-orbit COHERENCE (Kuramoto order parameter of the QA orbit phases) --")
    print("it discriminates cleanly and TRACKS harmonic self-organization (rises under strong")
    print("coupling, flat under weak). e8_alignment (icosian) is kept as a geometric readout.")
    print("See qa_core/metrics.py::harmonic_coherence.")


if __name__ == "__main__":
    run()
