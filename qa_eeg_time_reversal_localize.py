#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=eeg_hilbert_phase_and_geometry_to_QA_phase, state_alphabet=mod24_A1; coherent-sum focal peak is an observer-layer readout (Theorem NT)"
"""
QA Time-Reversal Focal Sharpness on real seizure EEG (cert [522] taken to data).

Cert [522]: qa_neg (the standard involution = phase conjugation = TIME REVERSAL)
re-emitted through an array FOCUSES back on a coherent source. Applied to EEG: a
scalp electrode array records a multichannel field; TIME-REVERSE the per-channel
phases and back-propagate across a scalp grid -> a FOCAL MAP whose peak measures
how well the recorded phases fit a single organized source.

FALSIFIABLE CLAIM (pre-registered, honest reporting): a SEIZURE is a spatially
organized focal source, so its time-reversal focus is SHARPER than diffuse baseline
activity. Test: ictal vs interictal focal-peak separation, SAME scan for both
classes (any 'best-of-N' inflation applies equally, so the difference is real).
This is NOT source localization (no ground-truth focus) and NOT the [520]
classifier -- it measures SPATIAL PHASE ORGANIZATION via the [522] operator.

Boundary (Theorem NT): EEG Hilbert phase + electrode geometry are observer inputs
(crossed once, in); QA phase arithmetic is integer mod m; the coherent-sum focal
peak is an observer readout (crossed once, out).

FINDINGS (real CHB-MIT, 5 patients, method fixed/pre-registered -- honest report):
  patient   ictal focal-peak   interictal   AUC(ictal sharper)
  chb01        0.604             0.589        0.562
  chb10        0.564             0.543        0.564
  chb17        0.605             0.585        0.562
  chb21        0.608             0.585        0.575
  chb16        0.555             0.580        0.381   (outlier; [520] flagged chb16
                                                       class-imbalanced/uninformative)
A REAL BUT WEAK effect: 4/5 patients show ictal focus consistently sharper (AUC
~0.56-0.58, same direction; ictal focal peak ~0.02 above interictal), chb16 the
lone reversal. Seizures ARE slightly more spatially phase-organized, but time-
reversal focal sharpness ALONE is a WEAK seizure marker -- far below the cert [520]
phase-conjugate RECALL classifier (0.83-1.00 on the same data). Honest negative-ish
result: the [522] focusing operator sees genuine ictal spatial organization, but as
a stand-alone discriminator it is weak; not a strong biomarker. Not tweaked to
inflate (single fixed method, same scan for both classes).
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

M = 24

# standard 10-20 electrode 2D positions (nose up; x=right, y=front; unit head)
POS = {
    "FP1": (-.31, .95), "FP2": (.31, .95),
    "F7": (-.81, .59), "F3": (-.40, .67), "FZ": (0, .71), "F4": (.40, .67), "F8": (.81, .59),
    "T7": (-1.0, 0), "T3": (-1.0, 0), "C3": (-.5, 0), "CZ": (0, 0), "C4": (.5, 0),
    "T8": (1.0, 0), "T4": (1.0, 0),
    "P7": (-.81, -.59), "T5": (-.81, -.59), "P3": (-.40, -.67), "PZ": (0, -.71),
    "P4": (.40, -.67), "P8": (.81, -.59), "T6": (.81, -.59),
    "O1": (-.31, -.95), "O2": (.31, -.95),
}


def chan_pos(label):
    """Bipolar channel 'FP1-F7' -> midpoint of its two electrodes, or None."""
    parts = label.upper().replace(" ", "").split("-")
    if len(parts) != 2 or parts[0] not in POS or parts[1] not in POS:
        return None
    a, b = POS[parts[0]], POS[parts[1]]
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def qa_mod(x):
    return ((np.asarray(x, np.int64) - 1) % M) + 1


def qa_add(a, b):
    return qa_mod(np.asarray(a, np.int64) + np.asarray(b, np.int64))


def qa_neg(a):
    return qa_mod(-np.asarray(a, np.int64))


def channel_phases(multi, fs, labels):
    """Observer projection: per-channel Hilbert phase (4-30 Hz) at window centre
    -> integer QA phase in {1..m}. Returns (positions Nx2, phases N)."""
    b, a = butter(4, [4.0 / (fs / 2), 30.0 / (fs / 2)], btype="band")
    pos, ph = [], []
    mid = multi.shape[1] // 2
    for i, lab in enumerate(labels):
        p = chan_pos(lab)
        if p is None:
            continue
        x = multi[i].astype(float)
        if np.std(x) < 1e-9:
            continue
        analytic = hilbert(filtfilt(b, a, x))
        theta = np.angle(analytic[mid])                 # instantaneous phase (rad)
        pos.append(p)
        ph.append(int(qa_mod(np.rint(theta / (2 * np.pi) * M + M))))
    return np.array(pos), np.array(ph, dtype=np.int64)


# grid + wavenumber scan (fixed for ALL windows -> fair ictal/interictal comparison)
GX, GY = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
GRID = np.column_stack([GX.ravel(), GY.ravel()])
KS = np.linspace(1.0, 6.0, 11)                          # candidate wavenumbers


def tr_focal_peak(pos, phases):
    """Time-reversal focus: for each grid point x and wavenumber k, back-propagate
    (qa_neg of the propagation phase k*dist) and coherently sum; return the peak
    focal magnitude over the (x,k) scan. 1.0 = all channels agree on one source."""
    if len(phases) < 6:
        return np.nan
    best = 0.0
    for k in KS:
        # propagation phase from each channel-position to every grid point
        d = np.sqrt(((pos[None, :, 0] - GRID[:, None, 0]) ** 2 +
                     (pos[None, :, 1] - GRID[:, None, 1]) ** 2))       # (Ngrid, Nch)
        prop = qa_mod(np.rint(d * (M / (2 * np.pi) * k)).astype(np.int64))
        field = qa_add(phases[None, :], qa_neg(prop))                  # back-propagated phase
        z = np.exp(2j * np.pi * (field - 1) / M).mean(axis=1)          # coherent sum per grid pt
        best = max(best, float(np.abs(z).max()))
    return best


def run(patient_dir, max_inter=200):
    from eeg_chbmit_scale import load_patient_dataset
    from eeg_orbit_observer_comparison import _read_edf_all_channels
    from pathlib import Path
    # channel labels are the fixed CHB-MIT montage; read once from the first EDF
    edfs = sorted(Path(patient_dir).glob("*.edf"))
    _, _, LABELS = _read_edf_all_channels(edfs[0])
    ds = load_patient_dataset(Path(patient_dir))
    ict = [w for w in ds if w["type"] in ("ictal", "seizure")]
    inter = [w for w in ds if w["type"] not in ("ictal", "seizure")]
    print(f"QA TIME-REVERSAL FOCAL SHARPNESS on {patient_dir}")
    print(f"  windows: {len(ict)} ictal, {len(inter)} interictal (sampling {min(len(inter),max_inter)})\n")
    rng = np.random.default_rng(42)
    if len(inter) > max_inter:
        inter = [inter[i] for i in rng.choice(len(inter), max_inter, replace=False)]

    def peaks(wins):
        out = []
        for w in wins:
            labs = LABELS[:w["multi_ch"].shape[0]]
            pos, ph = channel_phases(w["multi_ch"], w["fs"], labs)
            v = tr_focal_peak(pos, ph)
            if not np.isnan(v):
                out.append(v)
        return np.array(out)

    pi, pb = peaks(ict), peaks(inter)
    if len(pi) == 0 or len(pb) == 0:
        print("  insufficient windows"); return
    # AUC: P(ictal focal peak > interictal focal peak)
    auc = np.mean([1.0 * (a > b) + 0.5 * (a == b) for a in pi for b in pb])
    print(f"  ictal    focal peak: mean {pi.mean():.3f}  (n={len(pi)})")
    print(f"  interict focal peak: mean {pb.mean():.3f}  (n={len(pb)})")
    print(f"  separation AUC (ictal focus sharper than interictal): {auc:.3f}")
    print(f"  {'SEIZURE FOCUSES MORE SHARPLY' if auc > 0.6 else 'no clear separation' if auc < 0.55 else 'weak separation'}"
          f"  (0.5 = chance)")


if __name__ == "__main__":
    patient = sys.argv[1] if len(sys.argv) > 1 else \
        "/Volumes/lacie/signal_experiments_offload/archive/phase_artifacts/phase2_data/eeg/chbmit/chb16"
    run(patient)
