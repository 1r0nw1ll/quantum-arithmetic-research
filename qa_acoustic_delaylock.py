#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=cross-spectral phase / group-delay lock on real audio (Theorem NT); qa_neg=conjugation=time reversal; the delay scan IS the time-reversal. The discrete content is the mod-M phase quantization. Real recorded notes; phases/coherences/AUC are observer-layer readouts."
# RT1_OBSERVER_FILE: spectra, phases, delays, coherences, AUC are observer-layer readouts.
"""
Fix + honest audit of the acoustic phase-conjugate operator.

Two questions:
  (A) FIX: qa_acoustic_sympathetic.py's mod-24 GLOBAL-PHASE lock failed (AUC 0.37) because a
      constant phase cannot absorb the per-note GROUP DELAY. The fix: lock over group delay tau
      (a phase ramp 2*pi*f*tau) instead of a constant phase -- scanning tau IS the time reversal
      (qa_neg). Does the delay-locked QA operator recover the specificity?
  (B) CONFLATION CHECK (Will's question): the phase-quantization modulus was set to M=24 "because
      QA is mod 24" -- but 24 is the PISANO PERIOD of the mod-9 golden orbit (an iteration count),
      NOT a natural phase modulus. If AUC is flat across M, then "24" is just resolution, never
      load-bearing here, and using it as a phase modulus WAS a conflation. Sweep M and see.

Operator: per note-pair, cross-spectral phase phi(f) over the harmonic band; for a delay grid,
residual = phi - 2*pi*f*tau; magnitude-weighted mod-M coherence; take the max over tau (the
time-reversal lock). Same pitch (shared harmonics) -> a tau aligns them -> high; different pitch
-> no tau aligns -> low. Data: 12 real notes (piano/violin/flute x C4/E4/G4/C5), 44.1 kHz.

COMMITTED PREDICTION: (A) delay-lock RECOVERS specificity (AUC > 0.8), unlike the global-phase
lock; (B) M is NOT special -- AUC is flat for M above a small minimum, so "mod 24" was borrowed
resolution (the Pisano period), i.e. I WAS conflating. FALSIFIABLE either way.
"""
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np
from scipy.io import wavfile

D = Path("data/acoustic_notes")
BAND = (150.0, 6000.0)
TAUS = np.linspace(-6e-3, 6e-3, 241)     # group-delay scan, +/-6 ms


def load_spec(path):
    fs, d = wavfile.read(path)
    d = d.astype(float)
    if d.ndim > 1:
        d = d.mean(axis=1)
    d = d / (np.max(np.abs(d)) + 1e-12)
    n = int(0.4 * fs); s = len(d) // 3
    w = d[s:s + n] * np.hanning(n)
    W = np.fft.rfft(w); freqs = np.fft.rfftfreq(n, 1.0 / fs)
    m = (freqs >= BAND[0]) & (freqs <= BAND[1])
    return W[m], freqs[m]


def delay_locked_coherence(Wi, Wj, freqs, M):
    """Time-reversal (delay tau) lock with mod-M phase quantization; magnitude-weighted
    coherence, maxed over tau. qa_neg = conjugation = the cross-spectrum + the delay scan."""
    cs = Wi * np.conj(Wj)                              # cross-spectrum
    phi = np.angle(cs); wgt = np.abs(cs); wsum = wgt.sum() + 1e-12
    best = 0.0
    for tau in TAUS:
        res = phi - 2.0 * np.pi * freqs * tau          # remove the group delay (phase ramp)
        q = ((np.rint(res * M / (2 * np.pi)).astype(int)) % M)   # mod-M phase bin
        qphi = 2.0 * np.pi * q / M
        c = float(np.abs(np.sum(wgt * np.exp(1j * qphi))) / wsum)
        best = max(best, c)
    return best


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos]))


def run():
    files = sorted(D.glob("*.wav"))
    specs = {}
    for f in files:
        inst, pitch = f.stem.rsplit("_", 1)
        W, fr = load_spec(f)
        specs[(inst, pitch)] = (W, fr)
    pairs = [((k1, k2)) for k1, k2 in itertools.combinations(specs, 2) if k1[0] != k2[0]]
    print(f"Delay-locked QA phase-conjugate operator on {len(specs)} real notes; "
          f"{len(pairs)} cross-instrument pairs.\n")

    print(f"[A+B] modulus sweep (delay-locked): does specificity recover, and is M=24 special?")
    print(f"{'M (phase bins)':>16} {'same-pitch':>12} {'diff-pitch':>12} {'AUC':>7}")
    print("-" * 52)
    aucs = {}
    for M in (4, 6, 9, 12, 18, 24, 36, 48, 96):
        same, diff = [], []
        for k1, k2 in pairs:
            Wi, fr = specs[k1]; Wj, _ = specs[k2]
            c = delay_locked_coherence(Wi, Wj, fr, M)
            (same if k1[1] == k2[1] else diff).append(c)
        a = auc(same, diff); aucs[M] = a
        star = "  <- M=24 (Pisano period of mod 9)" if M == 24 else ""
        print(f"{M:>16} {np.mean(same):12.3f} {np.mean(diff):12.3f} {a:7.2f}{star}")

    a24 = aucs[24]
    a_high = [aucs[M] for M in aucs if M >= 6]
    flat = (max(a_high) - min(a_high)) < 0.08
    recovers = a24 > 0.8
    print("\nVERDICT:")
    print(f"  (A) FIX -- delay-lock: AUC={a24:.2f} at the QA modulus vs the failed global-phase lock")
    print(f"      (0.37). {'RECOVERED' if recovers else 'did NOT recover'}: scanning the group delay tau")
    print(f"      (= the time reversal) is the compensation the constant phase could not do. The")
    print(f"      QA-specific discrete operator now {'matches' if recovers else 'still misses'} the physics.")
    print(f"  (B) CONFLATION CHECK -- modulus sweep: AUC across M>=6 is "
          f"{'FLAT (' + f'{min(a_high):.2f}-{max(a_high):.2f}' + ')' if flat else 'NOT flat'}.")
    if flat:
        print(f"      -> M=24 is NOT special; it is just phase RESOLUTION. So using 'mod 24' as the")
        print(f"      phase-quantization modulus 'because QA is mod 24' WAS a conflation: 24 is the")
        print(f"      PISANO PERIOD of the mod-9 golden orbit (an iteration count), not a phase")
        print(f"      modulus. The operator works for any adequate M; the number 24 carries no")
        print(f"      meaning here. (You were right to check.)")
    else:
        print(f"      -> AUC varies with M; 24 may be doing something -- needs a real justification,")
        print(f"      not just 'QA is mod 24'.")
    ok = recovers and flat
    print(f"\n  STATUS: {'FIX works (delay-lock recovers) AND mod-24 was a conflation (flat sweep)' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
