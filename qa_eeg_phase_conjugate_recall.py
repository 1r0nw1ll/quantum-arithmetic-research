#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=EEG_topographic_power_to_phase, state_alphabet=mod24_A1_compliant"
# RT1_OBSERVER_FILE: EEG band-power / log / z-score are observer-layer signal features on
# continuous voltages; never QA state. The QA layer (phase vectors, conjugate memory) is integer.
"""
QA Phase-Conjugate EEG Brain-State Recall — cert [519] applied to real EEG.

Clinical framing: electrode dropout and artifact contamination are routine in EEG.
Can a QA phase-conjugate associative memory (cert [519]), built on the exact FWM
conjugator (cert [518]), reconstruct the brain-state and recall its class
(ictal vs interictal) from a NOISY or PARTIAL electrode montage — and stay robust
to a global reference-shift artifact via phase-locking?

Pipeline (real data, CHB-MIT via the project's own loader):
  EEG 10s multi-channel window -> per-channel log band-power -> per-channel
  z-score across windows -> quantize to QA phase {1,...,m} (A1). That per-channel
  phase vector is the topographic brain-state pattern. Train windows are stored
  in the memory; test windows are corrupted (channel dropout / global shift) and
  recalled.

Observer boundary (Theorem NT) crossed twice: continuous EEG -> phase vector
[inbound]; recalled phase vector -> class/reconstruction [outbound]. QA layer
integer throughout.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qa_phase_conjugate_memory import (  # noqa: E402
    QAPhaseConjugateMemory, qa_add, qa_neg, qa_mod, M,
)
from eeg_chbmit_scale import load_patient_dataset  # noqa: E402

RNG = np.random.default_rng(42)
DEFAULT_PATIENT = "chb10"
PATIENT_DIR = Path("archive/phase_artifacts/phase2_data/eeg/chbmit") / (
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATIENT)


def topographic_vectors(dataset):
    """Each window -> per-channel z-scored log-power -> QA phase vector {1..M}.
    Returns (X: (n_win, n_ch) int phases, y: labels 1=ictal 0=interictal)."""
    feats, labels = [], []
    n_ch = min(w["multi_ch"].shape[0] for w in dataset)
    for w in dataset:
        mc = w["multi_ch"][:n_ch]                      # (n_ch, n_samp)
        power = np.log(np.mean(mc * mc, axis=1) + 1e-12)  # per-channel log-power (observer)
        feats.append(power)
        labels.append(1 if w["type"] in ("ictal", "seizure") else 0)
    F = np.array(feats)                                 # (n_win, n_ch), continuous
    mu = F.mean(0); sd = F.std(0) + 1e-9
    Z = (F - mu) / sd                                   # per-channel z-score (observer)
    # Observer boundary: map z -> phase {1..M}. z in ~[-3,3] spread across M bins.
    phases = qa_mod(np.rint(np.clip(Z, -3, 3) * (M / 6.0) + (M / 2.0)).astype(np.int64))
    return phases, np.array(labels)


def dropout(x, frac, rng):
    """Electrode-dropout artifact: replace a fraction of channels with random phase."""
    x = x.copy()
    idx = rng.choice(len(x), int(frac * len(x)), replace=False)
    x[idx] = rng.integers(1, M + 1, len(idx))
    return x


def classify_direct(mem, y_store, probe):
    """Baseline: nearest stored pattern by phase-match (no recall dynamics)."""
    C = mem.overlap(probe)
    return y_store[int(np.argmax(C))]


def classify_recall(mem, y_store, probe):
    """Associative recall, then class of the exact stored pattern it lands on."""
    rec = mem.recall(probe)
    C = np.array([np.sum(rec == mem.P[k]) for k in range(mem.K)])
    return y_store[int(np.argmax(C))]


def classify_phase_locked(mem, y_store, probe):
    """Global-shift-robust classification (cert [518] property): scan the global
    compensation phase psi, take the (psi, stored-pattern) with maximal overlap,
    and read the class in that compensated frame. This is the phase-conjugate
    mirror self-locking to the reference before recall."""
    best_k, best = 0, -1.0
    for psi in range(1, M + 1):
        C = mem.overlap(qa_add(probe, psi))
        k = int(np.argmax(C))
        if C[k] > best:
            best, best_k = C[k], k
    return y_store[best_k]


def run():
    print(f"Loading {PATIENT_DIR.name} ...")
    ds = load_patient_dataset(PATIENT_DIR)
    X, y = topographic_vectors(ds)
    n, n_ch = X.shape
    print(f"windows={n}  channels={n_ch}  ictal={int(y.sum())}  interictal={int((1-y).sum())}\n")

    # stratified train(store)/test split
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    P, y_store = X[tr], y[tr]
    mem = QAPhaseConjugateMemory(P, sharpen=6.0)
    chance = max(y.mean(), 1 - y.mean())

    print(f"stored patterns={len(P)}  test probes={len(te)}  chance(majority)={chance:.3f}\n")

    print("[1] Class recall vs electrode dropout  (memory vs direct-NN baseline):")
    print(f"{'dropout':>8s} {'direct-NN':>10s} {'PC-memory':>10s}")
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8):
        dn = pc = 0
        for i in te:
            probe = dropout(X[i], frac, rng)
            dn += int(classify_direct(mem, y_store, probe) == y[i])
            pc += int(classify_recall(mem, y_store, probe) == y[i])
        print(f"{frac:8.0%} {dn/len(te):10.3f} {pc/len(te):10.3f}")

    print("\n[2] Reconstruction fidelity vs dropout (recall vs stored nearest, exact-site rate):")
    for frac in (0.2, 0.4, 0.6):
        fid = []
        for i in te:
            probe = dropout(X[i], frac, rng)
            rec = mem.recall(probe)
            k = int(np.argmax([np.sum(rec == mem.P[j]) for j in range(mem.K)]))
            fid.append(np.mean(rec == mem.P[k]))
        print(f"      dropout {frac:.0%}   mean reconstruction {np.mean(fid):.3f}")

    print("\n[3] Distortion tolerance — global reference-shift artifact (phi):")
    print(f"{'phi':>5s} {'direct-NN':>10s} {'naive-mem':>10s} {'phase-lock':>11s}")
    for phi in (0, 2, 6, 12):
        dn = nm = pl = 0
        for i in te:
            probe = qa_add(dropout(X[i], 0.2, rng), phi)
            dn += int(classify_direct(mem, y_store, probe) == y[i])
            nm += int(classify_recall(mem, y_store, probe) == y[i])
            pl += int(classify_phase_locked(mem, y_store, probe) == y[i])
        print(f"{phi:5d} {dn/len(te):10.3f} {nm/len(te):10.3f} {pl/len(te):11.3f}")
    print(f"\n(chance/majority = {chance:.3f})")


if __name__ == "__main__":
    run()
