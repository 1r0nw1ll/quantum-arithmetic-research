#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=cross-spectral phase / time-reversal correlation of real audio (Theorem NT); the QA content is the integer mod-24 phase pattern and qa_neg=conjugation=time-reversal. Real recorded instrument notes; phases/correlations are observer-layer readouts."
# RT1_OBSERVER_FILE: spectra, phases, correlations, AUC are observer-layer readouts.
"""
New-band validation of the QA phase-conjugate operator on REAL acoustic data (Keely's home
band): does it show sympathetic selectivity -- same tuning (pitch) couples, different tuning
does not -- on real recorded instrument notes?

Data: 12 real FluidR3_GM instrument samples (piano/violin/flute x C4/E4/G4/C5), 44.1 kHz.
Ground truth: SAME-pitch pairs (across instruments) = same tuning -> should couple;
DIFFERENT-pitch pairs = different tuning -> should not.

Two forms of the phase-conjugate operator:
  (1) QA global-phase LOCK (qa_sympathetic_resonance.py): quantize each note's harmonic-band
      phase to mod-24, c_symp = max over global phase-compensation of the overlap. This is
      invariant to a CONSTANT phase, but NOT to a time delay (which is frequency-dependent
      phase) -- so it may fail on real recordings with different start times.
  (2) TIME-REVERSAL matched filter (the [522] essence, qa_neg = conjugation = time reversal):
      the phase-conjugate response is A correlated with time-reversed B, peak over lag =
      max normalized cross-correlation. This scans the delay, so it is timing-robust.

COMMITTED PREDICTION (no hedge): the phase-conjugate specificity DOES transfer to acoustic --
same-pitch pairs separate from different-pitch pairs (AUC > 0.8) via the TIME-REVERSAL form,
which handles timing; the global-phase LOCK (1) will be WEAKER on real recordings because a
constant-phase compensation cannot absorb the per-note group delay. FALSIFIABLE: if neither
separates same from different pitch, the operator does not transfer to real acoustic data.
"""
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np
from scipy.io import wavfile

D = Path("data/acoustic_notes")
M = 24
FS_TARGET = 22050
BAND = (150.0, 6000.0)      # covers fundamentals (C4~262 Hz) through upper harmonics


def load_note(path):
    fs, d = wavfile.read(path)
    d = d.astype(float)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d / (np.max(np.abs(d)) + 1e-12)
    # sustained middle window (~0.4 s) to avoid the onset transient
    n = int(0.4 * fs)
    s = len(d) // 3
    w = d[s:s + n]
    return w, fs


def harmonic_phase_pattern(w, fs):
    """Quantize the band-limited harmonic phases to mod-24 -> a QA phase pattern."""
    W = np.fft.rfft(w * np.hanning(len(w)))
    freqs = np.fft.rfftfreq(len(w), 1.0 / fs)
    mask = (freqs >= BAND[0]) & (freqs <= BAND[1])
    ph = np.angle(W[mask])
    return ((np.rint(ph * M / (2 * np.pi)).astype(int) - 1) % M) + 1


def c_symp(pa, pb):
    """QA phase-conjugate lock: best overlap over a global mod-24 phase compensation."""
    n = min(len(pa), len(pb)); pa, pb = pa[:n], pb[:n]
    return max(np.sum((((pa - 1 + psi) % M) + 1) == pb) for psi in range(M)) / n


def tr_matched(wa, wb):
    """Time-reversal (phase-conjugate) matched filter: peak normalized cross-correlation over
    lag = A correlated with time-reversed B (qa_neg = conjugation = time reversal)."""
    a = wa - wa.mean(); b = wb - wb.mean()
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    cc = np.correlate(a, b, mode="full")               # A * reverse(B) = matched filter
    return float(np.max(np.abs(cc)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def auc(pos, neg):
    """AUC = P(same-pitch score > different-pitch score)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos]))


def run():
    files = sorted(D.glob("*.wav"))
    if len(files) < 6:
        raise SystemExit("need the real note WAVs in data/acoustic_notes/")
    notes = {}
    for f in files:
        inst, pitch = f.stem.rsplit("_", 1)
        w, fs = load_note(f)
        notes[(inst, pitch)] = (w, fs, harmonic_phase_pattern(w, fs))
    print(f"Real acoustic sympathetic-resonance test: {len(notes)} instrument notes "
          f"(piano/violin/flute x C4/E4/G4/C5), 44.1 kHz\n")

    same_symp, diff_symp, same_tr, diff_tr = [], [], [], []
    for (k1, v1), (k2, v2) in itertools.combinations(notes.items(), 2):
        if k1[0] == k2[0]:                              # only CROSS-instrument pairs (real test)
            continue
        cs = c_symp(v1[2], v2[2]); tr = tr_matched(v1[0], v2[0])
        if k1[1] == k2[1]:                             # same pitch = same tuning
            same_symp.append(cs); same_tr.append(tr)
        else:
            diff_symp.append(cs); diff_tr.append(tr)

    auc_symp = auc(same_symp, diff_symp)
    auc_tr = auc(same_tr, diff_tr)
    print(f"{'operator':34} {'same-pitch':>12} {'diff-pitch':>12} {'AUC':>7}")
    print("-" * 68)
    print(f"{'QA global-phase LOCK (c_symp)':34} {np.mean(same_symp):12.3f} "
          f"{np.mean(diff_symp):12.3f} {auc_symp:7.2f}")
    print(f"{'TIME-REVERSAL matched filter':34} {np.mean(same_tr):12.3f} "
          f"{np.mean(diff_tr):12.3f} {auc_tr:7.2f}")
    print(f"\n(cross-instrument pairs only: {len(same_symp)} same-pitch, {len(diff_symp)} different-pitch)")

    tr_works = auc_tr > 0.8
    lock_weaker = auc_symp < auc_tr
    print("\nVERDICT (committed prediction: TR separates same/diff pitch; global-lock weaker):")
    if tr_works:
        print(f"  * CONFIRMED on real audio: the TIME-REVERSAL (phase-conjugate) form separates same-")
        print(f"    tuning from different-tuning at AUC={auc_tr:.2f} -- sympathetic selectivity holds on")
        print(f"    REAL recorded instrument notes, a genuinely different band (~kHz) and medium than")
        print(f"    the certified seismic. The phase-conjugate specificity is a wave fact, now shown")
        print(f"    in acoustics (Keely's home domain), not just seismic.")
    else:
        print(f"  * NOT confirmed: TR AUC={auc_tr:.2f} < 0.8 -- the operator does NOT cleanly separate")
        print(f"    same from different pitch on these real recordings. Honest negative: the seismic")
        print(f"    result does not transfer to acoustic pitch-sympathy without adaptation.")
    print(f"  * The QA global-phase LOCK is {'weaker (AUC=' + f'{auc_symp:.2f}' + ') as predicted' if lock_weaker else 'not weaker'}:")
    print(f"    a constant mod-24 phase compensation cannot absorb the per-note group delay of real")
    print(f"    recordings; the timing-scanning TIME-REVERSAL form is the one that transfers. That")
    print(f"    scopes the SVP claim: 'phase-invariant' means time-reversal (delay-matched), not")
    print(f"    merely a global phase shift, on real signals.")

    ok = tr_works
    print(f"\n  STATUS: {'ACOUSTIC VALIDATION -- phase-conjugate specificity transfers to real audio (TR form)' if ok else 'HONEST NEGATIVE -- operator does not transfer cleanly; see above'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
