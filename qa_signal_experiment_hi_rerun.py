#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=diagnostic_rerun; HI/coherence are observer-layer readouts (Theorem NT); reuses run_signal_experiments_final generators unchanged"
# RT1_OBSERVER_FILE: signal generators (sin tones) are observer-layer input signals, not QA state.
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
looks like an artifact of the inert metric plus a lucky seed -- the injected signal
enters as a common per-timestep drive on the b-state, which plausibly synchronizes
the coupled QA nodes regardless of whether it is tonal or noise. Note this is an
UNCHANGED replication of the flagship setup, NOT a clean harmonic-structure-only
ablation: the reused generators are not amplitude-matched (white noise has higher RMS
than the chords), so signal type and amplitude are confounded -- same as the original.

FAMILY: this checks the signal-coherence flagship family. The 'final' config
(coupling 0.2, unnormalized generators) REVERSES (white noise significantly higher);
the multiseed-eval config (coupling 0.12, amplitude-normalized generators) is NULL
(no significant difference). The pisano / pac / tight_bounds variants share the same
tonal-vs-noise QASystem base, so they inherit the RISK to that shared HI-based claim
(not every result in those scripts -- they run longer and add PAC/Pisano analyses).
The seismic classifier (seismic_classifier_enhanced.py) uses HI only as a +/-0.5
tiebreaker AFTER P/S-wave-timing/amplitude rules, so the redesign is expected not to
materially change its verdict except possibly for borderline/tie cases.

Honest consequence of the HI redesign: a flagship claim that rested on a
non-discriminative metric does not replicate under a discriminative one. SCOPE:
only HI/QASystem-dependent claims; results that do not use HI -- e.g.
Fibonacci-resonance, EEG topographic, climate -- are not touched by this.
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

# the multiseed-eval variant of the same flagship family (signal_experiment_multiseed_eval.py):
# weaker coupling + amplitude-normalized generators (chords /3, tritone /2).
MULTISEED_CFG = dict(num_nodes=16, modulus=24, coupling=0.12, noise_base=0.1,
                     noise_annealing=0.995, signal_injection_strength=0.2, signal_mode="final")


def _norm_gens(n):
    t = np.linspace(0, 1.0, n, endpoint=False); f = 5.0
    s = lambda r: np.sin(2 * np.pi * f * r * t)
    return {"Pure Tone": s(1), "Major Chord": (s(1) + s(1.25) + s(1.5)) / 3,
            "Minor Chord": (s(1) + s(1.2) + s(1.5)) / 3, "Tritone": (s(1) + s(np.sqrt(2))) / 2,
            "White Noise": np.random.uniform(-1, 1, n)}


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


def _multiseed_family_check(seeds=range(8)):
    """Same claim under the amplitude-normalized multiseed-eval variant (weaker
    coupling). Returns the seed-level paired t-test of tonal vs white-noise new-HI."""
    seeds = list(seeds)
    tonal_means, wn = [], []
    for seed in seeds:
        vals = {}
        for i, name in enumerate(SIGS):
            np.random.seed(1000 * seed + i)
            g = _norm_gens(150)[name]
            s = QASystem(**MULTISEED_CFG)
            s.run_simulation(150, g, progress=False)
            vals[name] = float(s.history["hi"][-1])
        tonal_means.append(np.mean([vals[t] for t in TONAL])); wn.append(vals["White Noise"])
    t, p = stats.ttest_rel(tonal_means, wn)
    return t, p


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
    # the same claim under the other config in the family (multiseed-eval variant)
    tm, pm = _multiseed_family_check()
    print(f"\n[5] FAMILY check -- multiseed-eval variant (coupling 0.12, amplitude-normalized "
          f"generators):\n      seed-level paired t (n=8) tonal vs white-noise new-HI: "
          f"t={tm:.2f}, p={pm:.3g} -> {'no significant difference' if pm>=0.05 else ('WN higher' if tm<0 else 'tonal higher')}")
    print("\nCONCLUSION: across the signal-coherence flagship FAMILY, 'tonal signals -> more")
    print("harmonic QA state' does NOT hold under the redesigned, discriminative HI: the final")
    print("config REVERSES (white noise significantly higher), the multiseed/normalized config")
    print("is NULL (no significant difference). The pisano/pac/tight_bounds variants share this")
    print("tonal-vs-noise base, so they inherit the RISK to that shared HI-BASED claim only (not")
    print("every result -- they run longer + add PAC/Pisano). Seismic (seismic_classifier_enhanced")
    print(".py) uses HI only as a +/-0.5 tiebreaker AFTER P/S-timing + amplitude rules -- redesign")
    print("expected not to change its verdict except borderline/tie cases. Interpretation (unproven):")
    print("the old single-")
    print("seed PASS was an inert-metric artifact. SCOPE: only HI/QASystem-dependent claims;")
    print("Fibonacci-resonance/EEG/climate do not use HI and are untouched. Reported as-is.")
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
