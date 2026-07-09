#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=EEG_multiband_power_to_phase, state_alphabet=mod24_A1_compliant"
# RT1_OBSERVER_FILE: band-power via FFT / log / z-score are observer-layer signal
# features on continuous EEG voltages; never QA state. QA layer (phase vectors,
# phase-conjugate memory) is integer.
"""
Thread 2: richer EEG features for cert [520].

[520] used a single broadband log-power per channel -> topographic phase vector.
Here each channel contributes PER-BAND power (delta/theta/alpha/gamma), so the
brain-state phase vector is n_ch x n_bands. Question: does the richer
representation improve recall / artifact-robustness over single-broadband, or is
[520] already at the alphabet-redundancy ceiling? Honest either way.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_phase_conjugate_memory import QAPhaseConjugateMemory, qa_add, qa_mod, M  # noqa: E402
from eeg_chbmit_scale import load_patient_dataset  # noqa: E402

RNG = np.random.default_rng(42)
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "gamma": (30, 100)}


def band_power(sig, fs, lo, hi):
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    psd = np.abs(np.fft.rfft(sig)) ** 2
    return float(psd[(f >= lo) & (f <= hi)].sum())


def feats_broadband(ds):
    n_ch = min(w["multi_ch"].shape[0] for w in ds)
    F = np.array([np.log(np.mean(w["multi_ch"][:n_ch] ** 2, axis=1) + 1e-12) for w in ds])
    return F


def feats_multiband(ds):
    n_ch = min(w["multi_ch"].shape[0] for w in ds)
    rows = []
    for w in ds:
        mc, fs = w["multi_ch"][:n_ch], w["fs"]
        v = []
        for ch in range(n_ch):
            for (lo, hi) in BANDS.values():
                v.append(np.log(band_power(mc[ch], fs, lo, hi) + 1e-12))
        rows.append(v)
    return np.array(rows)


def to_phase(F):
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return qa_mod(np.rint(np.clip(Z, -3, 3) * (M / 6.0) + (M / 2.0)).astype(np.int64))


def dropout(x, frac, rng):
    x = x.copy(); idx = rng.choice(len(x), int(frac * len(x)), replace=False)
    x[idx] = rng.integers(1, M + 1, len(idx)); return x


def evaluate(X, y, tag):
    rng = np.random.default_rng(0)
    n = len(X); idx = rng.permutation(n); tr, te = idx[:n // 2], idx[n // 2:]
    P, ys = X[tr], y[tr]
    mem = QAPhaseConjugateMemory(P, sharpen=6.0)

    def cls_direct(probe): return ys[int(np.argmax(mem.overlap(probe)))]

    def cls_pl(probe):
        bk, best = 0, -1
        for psi in range(1, M + 1):
            C = mem.overlap(qa_add(probe, psi)); k = int(np.argmax(C))
            if C[k] > best: best, bk = C[k], k
        return ys[bk]

    # clean recall + phase-lock under systemic artifact phi=6
    clean = np.mean([cls_direct(X[i]) == y[i] for i in te])
    dn6 = np.mean([cls_direct(qa_add(dropout(X[i], 0.2, rng), 6)) == y[i] for i in te])
    pl6 = np.mean([cls_pl(qa_add(dropout(X[i], 0.2, rng), 6)) == y[i] for i in te])
    d80 = np.mean([ys[int(np.argmax(mem.overlap(dropout(X[i], 0.8, rng))))] == y[i] for i in te])
    print(f"  {tag:12s} dim={X.shape[1]:3d}  clean={clean:.3f}  80%dropout={d80:.3f}  "
          f"phi6 naive={dn6:.3f} phase-lock={pl6:.3f}")


def run(patient):
    print(f"### {patient}")
    ds = load_patient_dataset(Path("archive/phase_artifacts/phase2_data/eeg/chbmit") / patient)
    y = np.array([1 if w["type"] in ("seizure", "ictal") else 0 for w in ds])
    print(f"  windows={len(ds)} ictal={int(y.sum())} chance={max(y.mean(),1-y.mean()):.3f}")
    evaluate(to_phase(feats_broadband(ds)), y, "broadband")
    evaluate(to_phase(feats_multiband(ds)), y, "multiband")


if __name__ == "__main__":
    for p in (sys.argv[1:] or ["chb10", "chb23"]):
        run(p)
