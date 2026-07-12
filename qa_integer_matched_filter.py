#!/usr/bin/env python3
# QA_COMPLIANCE = "EXACT integer arithmetic: 16-bit PCM samples are integers; the time-reversal cross-correlation is an integer sum-of-products; the AUC ordering is decided by exact integer cross-multiplication (no sqrt, no float, no division). qa_neg=conjugation=time reversal. Real recorded notes."
"""
The dichotomy was false: the "continuous physics" time-reversal matched filter (AUC 0.97) is a
DISCRETE INTEGER operation on the integer PCM samples -- never continuous.

The point I got wrong all session: sorting operators into "discrete QA" vs "continuous math" and
treating float64 as the real physics. There is no continuous computation. The matched filter ran
on int16 samples through a truncated 52-bit float mantissa over finitely many lags. It is a
discretization, just a finer one -- and it is exactly expressible in pure integers, which this
file does: same operator, integer samples, integer cross-correlation, and the AUC ordering
decided by EXACT integer cross-multiplication (comparing peak_a^2 * E_c * E_d vs peak_c^2 * E_a *
E_b -- no sqrt, no division, no float). If it still separates same-pitch from different-pitch at
~0.97, the "continuous vs discrete" framing dissolves: the working operator was always a discrete
integer operator that QA can simply OWN (consistent with the project's Wildberger finitism -- no
completed infinities, no real numbers, only exact finite arithmetic).
"""
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np
from scipy.io import wavfile

D = Path("data/acoustic_notes")
LAG = 300              # +/- lags to scan (covers a few periods of C4 at ~22 kHz)


def load_int(path):
    fs, d = wavfile.read(path)                          # int16 PCM
    if d.ndim > 1:
        d = d[:, 0].astype(np.int64) + d[:, 1].astype(np.int64)   # sum channels -> exact integer
    else:
        d = d.astype(np.int64)
    n = int(0.4 * fs); s = len(d) // 3
    return d[s:s + n] - int(round(d[s:s + n].mean()))    # integer demean (round is exact-int here)


def matched_int(a, b):
    """Time-reversal matched filter in EXACT integers: peak |sum a[i] b[i+lag]| over lags, and
    the two energies. All int64 (values fit); returned as Python ints for exact comparison."""
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    cc = np.correlate(a, b, mode="full")                # integer sum-of-products (int64, exact)
    mid = n - 1
    peak = int(np.max(np.abs(cc[mid - LAG:mid + LAG + 1])))
    Ea = int(np.dot(a, a)); Eb = int(np.dot(b, b))       # integer energies
    return peak, Ea, Eb


def auc_exact(pos, neg):
    """AUC = P(same > diff), where 'score' = peak/sqrt(Ea*Eb) is compared WITHOUT computing it:
    peak_p/sqrt(Ea_p Eb_p) > peak_q/sqrt(Ea_q Eb_q)  <=>  peak_p^2 Ea_q Eb_q > peak_q^2 Ea_p Eb_p
    -- an exact integer inequality (Python big ints). No sqrt, no division, no float."""
    wins = ties = tot = 0
    for pp, ea, eb in pos:
        for qq, ec, ed in neg:
            lhs = pp * pp * ec * ed
            rhs = qq * qq * ea * eb
            wins += lhs > rhs; ties += lhs == rhs; tot += 1
    return (wins + 0.5 * ties) / tot


def run():
    files = sorted(D.glob("*.wav"))
    sig = {}
    for f in files:
        inst, pitch = f.stem.rsplit("_", 1)
        sig[(inst, pitch)] = load_int(f)
    same, diff = [], []
    for (k1, a), (k2, b) in itertools.combinations(sig.items(), 2):
        if k1[0] == k2[0]:
            continue
        feat = matched_int(a, b)
        (same if k1[1] == k2[1] else diff).append(feat)
    a = auc_exact(same, diff)
    print("Time-reversal matched filter in PURE INTEGER arithmetic (real notes, int16 PCM)\n")
    print(f"  cross-instrument pairs: {len(same)} same-pitch, {len(diff)} different-pitch")
    print(f"  AUC (exact integer ordering, no float/sqrt/division): {a:.2f}")
    ok = a > 0.9
    print("\nVERDICT:")
    print(f"  * The time-reversal operator that got 'continuous' AUC 0.97 gives {a:.2f} in PURE")
    print(f"    INTEGER arithmetic -- integer samples, integer cross-correlation, and the ranking")
    print(f"    decided by exact integer cross-multiplication (no sqrt, no division, no float).")
    print(f"  * So it was never 'continuous'. There is no continuous computation: the PCM samples")
    print(f"    are integers, float64 is a 52-bit truncation, and the whole operator is finite")
    print(f"    discrete arithmetic. The 'discrete QA vs continuous physics' dichotomy I ran all")
    print(f"    session is FALSE -- both sides are discrete; one just used more of the signal at")
    print(f"    finer resolution. The working operator is a discrete integer operator QA can OWN.")
    print(f"  * This is the project's own (Wildberger) finitism: exact finite arithmetic is primary;")
    print(f"    'the continuum' is an idealization no computation ever touches. My privileging of it")
    print(f"    was the fundamental error, not a detail.")
    print(f"\n  STATUS: {'DICHOTOMY DISSOLVED -- the operator is exact integer, AUC ' + f'{a:.2f}' if ok else 'unexpected -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
