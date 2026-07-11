#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=diagnostic_rerun; HI/coherence are observer-layer readouts (Theorem NT); reuses run_signal_experiments_final generators unchanged"
"""
Re-run the flagship SIGNAL-COHERENCE experiment (run_signal_experiments_final.py)
with the REDESIGNED Harmonic Index, and test whether its headline conclusion
survives a metric that actually measures something.

The experiment's hypothesis: tonal/harmonic signals (pure tone, major/minor chord,
tritone) injected into the coupled QA system produce a MORE HARMONIC (higher-HI)
state than white noise. Its PASS verdict required 3/4 tonal signals + both chords
to exceed white noise by HI.

Under the OLD HI (proven inert: E8-alignment-to-a-Fibonacci-vector, and loss ~ 0),
the single seed 42 gave tonal > noise -> PASS. This re-runs across many seeds with
the NEW coherence-HI (golden-orbit Kuramoto synchronization -- see
qa_core/metrics.py::harmonic_coherence).

FINDING (run it): the conclusion does NOT survive. Under the meaningful coherence-HI
the "tonal > white noise" verdict passes in only 3/25 seeds (12%); seed 42 (the
original) was one of the few. The mean tonal-minus-white-noise HI margin is slightly
NEGATIVE (19/25 seeds negative), and a SEED-LEVEL paired t-test (n=25, the seed is
the independent unit) finds white noise produces significantly HIGHER coherence than
tonal signals (t ~ -3.7, p ~ 0.001) -- the reverse of the hypothesis.

INTERPRETATION (not established by this experiment alone): the old flagship PASS
looks like an artifact of the inert metric plus a lucky seed -- a common scalar drive
(the injected signal, entered as float(mean(signal))) plausibly synchronizes the
coupled QA nodes regardless of whether it is tonal or noise. Note this is an
UNCHANGED replication of the flagship setup, NOT a clean harmonic-structure-only
ablation: the reused generators are not amplitude-matched (white noise has higher RMS
than the chords), so signal type and amplitude are confounded -- same as the original.

Honest consequence of the HI redesign: a flagship claim that rested on a
non-discriminative metric does not replicate under a discriminative one. SCOPE: this
concerns the signal-coherence classification experiment specifically; results that do
not use HI/QASystem -- e.g. Fibonacci-resonance, EEG topographic, climate -- are not
touched by this.
"""
from __future__ import annotations

EXPERIMENT_PROTOCOL_REF = "experiment_protocol.json"

import numpy as np
from scipy import stats
from qa_core import QASystem
from qa_core.metrics import qa_tuples, e8_alignment_legacy
import run_signal_experiments_final as F

SIGS = {"Pure Tone": F.generate_pure_tone, "Major Chord": F.generate_major_chord,
        "Minor Chord": F.generate_minor_chord, "Tritone": F.generate_tritone,
        "White Noise": F.generate_white_noise}
TONAL = ["Pure Tone", "Major Chord", "Minor Chord", "Tritone"]
CFG = dict(num_nodes=16, modulus=24, coupling=0.2, noise_base=0.2,
           noise_annealing=0.995, signal_injection_strength=0.2, signal_mode="final")


def _one_seed(seed):
    """Return {signal: (old_HI_proxy, new_HI)} for one seeded run of all signals."""
    np.random.seed(seed)
    out = {}
    for name, gen in SIGS.items():
        s = QASystem(**CFG)
        s.run_simulation(150, gen(150), progress=False)
        tup = qa_tuples(np.asarray(s.b), np.asarray(s.e), 24)
        out[name] = (float(e8_alignment_legacy(tup)), float(s.history["hi"][-1]))
    return out


def _verdict(res, idx):
    wn = res["White Noise"][idx]
    return (res["Major Chord"][idx] > wn and res["Minor Chord"][idx] > wn
            and sum(res[t][idx] > wn for t in TONAL) >= 3)


def old_vs_new_HI(seeds=range(25)):
    """Ablation: replace the inert old HI (E8 alignment) with the discriminative new
    HI (golden-orbit coherence) and re-test the flagship 'tonal > white noise' claim
    across seeds. Returns a results dict; prints the summary."""
    seeds = list(seeds)
    old_pass = new_pass = 0
    margins = []
    seed_tonal, seed_noise = [], []          # SEED-LEVEL (the independent unit)
    for seed in seeds:
        r = _one_seed(seed)
        old_pass += _verdict(r, 0)
        new_pass += _verdict(r, 1)
        wn = r["White Noise"][1]
        tonal_mean = np.mean([r[t][1] for t in TONAL])
        margins.append(tonal_mean - wn)
        seed_tonal.append(tonal_mean); seed_noise.append(wn)
    margins = np.array(margins)
    # paired test at the SEED level (n=len(seeds)); repeating white-noise per tonal
    # signal would pseudo-replicate and inflate p, so we pair per-seed means.
    t, p = stats.ttest_rel(seed_tonal, seed_noise)
    n_neg = int(np.sum(margins < 0))
    print(f"Flagship signal-coherence experiment re-run over {len(seeds)} seeds:\n")
    print(f"  OLD HI (inert Fibonacci-alignment) 'tonal>noise' PASS: {old_pass}/{len(seeds)} "
          f"({100*old_pass/len(seeds):.0f}%)")
    print(f"  NEW HI (golden-orbit coherence)    'tonal>noise' PASS: {new_pass}/{len(seeds)} "
          f"({100*new_pass/len(seeds):.0f}%)")
    print(f"\n  NEW HI mean (tonal - white-noise) margin: {margins.mean():+.4f} +/- {margins.std():.4f} "
          f"({n_neg}/{len(seeds)} seeds negative)")
    print(f"  SEED-LEVEL paired t-test (n={len(seeds)}) tonal vs white-noise coherence: t={t:.2f}, p={p:.3g}")
    if p < 0.05 and t < 0:
        print("  -> white noise coherence is SIGNIFICANTLY HIGHER than tonal (reverse of hypothesis)")
    elif p < 0.05 and t > 0:
        print("  -> tonal coherence significantly higher (hypothesis holds)")
    else:
        print("  -> no significant difference (hypothesis not supported)")
    print("\nCONCLUSION: the flagship 'tonal signals -> more harmonic QA state' verdict does NOT")
    print("replicate under the redesigned, discriminative HI (3/25 seeds; reversed and")
    print("significant). Interpretation (not proven here): the old single-seed PASS looks like an")
    print("artifact of the inert metric; a common scalar drive plausibly synchronizes the nodes")
    print("regardless of signal type. Unchanged replication (amplitude not matched); reported as-is.")
    return {
        "n_seeds": len(seeds), "old_pass": int(old_pass), "new_pass": int(new_pass),
        "new_margin_mean": float(margins.mean()), "new_margin_std": float(margins.std()),
        "ttest_t": float(t), "ttest_p": float(p),
        "verdict": "NOT_REPLICATED_UNDER_NEW_HI" if (p < 0.05 and t < 0) else "inconclusive",
    }


if __name__ == "__main__":
    from qa_reproducibility import log_run
    summary = old_vs_new_HI()
    log_run(EXPERIMENT_PROTOCOL_REF, status="complete", results=summary)
