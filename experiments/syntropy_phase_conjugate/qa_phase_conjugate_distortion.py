#!/usr/bin/env python3
# RT1_OBSERVER_FILE: continuous-signal analysis (coherent signal through FIR distortion channel driving the QA engine); trig acts on observer-layer signals, never QA state.
"""
QA phase-conjugate distortion correction — the same-medium test (cert [518]).

Grounds the coherence-extraction result in the DISTORTION-CORRECTION THEOREM
(Yariv/Zel'dovich; Agarwal-Friberg scattering-theory proof): a phase-conjugated
wave returned through the SAME distorting medium precisely undoes the distortion;
a different medium does not. That same-medium specificity is the fingerprint that
separates true phase-conjugate reconstruction from generic denoising.

Holographic mapping to QA:
  RECORD    : self-organizing QASystem coupling adapts while driven by a coherent
              signal passed through a fixed distorting channel D (FIR + noise).
              The settled coupling W is the "recorded hologram" of medium D.
  READOUT   : continue driving with the coherent signal through either the SAME D
              or a statistically identical but independent D'. Read the QA-native
              coherence of the response: QCI = T-operator coherence index
              (rolling fraction of triples with (b+e) mod m recurrence), exactly
              per qa_observer.core.QCI / cert [155] Bearden module. Gap = local-global.

Committed prediction (distortion-correction theorem, no hedge):
  QCI_same > QCI_diff   -- reconstruction is medium-specific.
Falsifier:
  QCI_same == QCI_diff  -- medium-agnostic => generic smoothing, PC claim dies.

Controls: NO_ADAPT (fresh coupling on the test medium) isolates the contribution
of the recording; clean/distorted reference QCI bracket the scale.

Real engine: qa_core.engine.QASystem (signal_mode='final'). QCI reimplemented
standalone (qa_observer import chain broken) faithfully per qa_observer/core.py.
A1/S2/Theorem-NT compliance in the engine.
"""
from __future__ import annotations

QA_COMPLIANCE = "observer=continuous_signal_injection_into_QASystem_b_state, state_alphabet=mod24_A1; negative-result companion to cert [518], see docs/theory/QA_SYNTROPY_PHASE_CONJUGATE_INVESTIGATION.md"
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from qa_core.engine import QASystem

M, N = 24, 24
T_ADAPT, T_TEST = 3000, 2000
QCI_LOCAL_W, QCI_GLOBAL_W = 7, 63

def qa_mod(x, m=M):
    return ((np.asarray(x, dtype=np.int64) - 1) % m) + 1

def coherent(n, phase=0.0):
    tt = np.arange(n)
    s = np.sin(2*np.pi*tt/220 + phase) + 0.6*np.sin(2*np.pi*tt/90 + 0.5 + phase)
    return (s - s.mean())/s.std()

def fir_medium(rng, L=8):
    k = rng.standard_normal(L)
    return k/np.linalg.norm(k)

def distort(s, kernel, sigma, rng):
    d = np.convolve(s, kernel, mode="same")
    return d + sigma*rng.standard_normal(len(s))

def qci_series(states, window):
    """Faithful QCI per qa_observer/core.py: rolling mean of T-operator match
    pred=(b+e) mod m == actual, identity cmap."""
    b = states[:-2]; e = states[1:-1]; actual = states[2:]
    pred = qa_mod(b + e)
    t_match = (pred == actual).astype(float)
    return pd.Series(t_match).rolling(window, min_periods=window//2).mean().values

def drive_record_readout(adapt_sig, test_sig, seed, adapt=True):
    """Return full distributed state trajectory (T_test, 2N) over the test phase."""
    np.random.seed(seed)
    sys = QASystem(num_nodes=N, modulus=M, coupling=0.15, noise_base=0.5,
                   noise_annealing=0.999, signal_injection_strength=0.5,
                   signal_mode="final")
    if adapt:
        for t, v in enumerate(adapt_sig):
            sys.step(t, float(v))
        off = len(adapt_sig)
    else:
        off = 0
    S = np.empty((len(test_sig), 2*N))
    for j, v in enumerate(test_sig):
        sys.step(off + j, float(v))
        S[j, :N] = sys.b; S[j, N:] = sys.e
    return S

def cv_decode_corr(state, clean, burn=200, l2=10.0):
    """Leakage-free reconstruction quality: ridge-decode clean signal from the
    distributed state on 1st half, score |corr| on held-out 2nd half."""
    from scipy.stats import pearsonr
    X = state[burn:]; y = clean[burn:]
    half = len(y)//2
    Xtr, ytr = X[:half], y[:half]; Xte, yte = X[half:], y[half:]
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    w = np.linalg.solve(Xtr.T@Xtr + l2*np.eye(Xtr.shape[1]), Xtr.T@(ytr-ytr.mean()))
    pred = Xte@w
    if pred.std() < 1e-9: return 0.0
    return abs(pearsonr(pred, yte)[0])

def run(sigma, seeds=range(18)):
    s = coherent(T_ADAPT + T_TEST)
    s_adapt_clean, s_test_clean = s[:T_ADAPT], s[T_ADAPT:]
    rows = {k: [] for k in ["SAME","DIFF","NOADAPT_SAME"]}
    for seed in seeds:
        rng = np.random.default_rng(3000+seed)
        D = fir_medium(rng); Dp = fir_medium(rng)   # record medium D, novel D'
        adapt_sig = distort(s_adapt_clean, D, sigma, rng)
        test_same  = distort(s_test_clean, D,  sigma, rng)   # same medium
        test_diff  = distort(s_test_clean, Dp, sigma, rng)   # different medium
        # reconstruction quality = decode clean s_test_clean from the readout state
        rows["SAME"].append(cv_decode_corr(
            drive_record_readout(adapt_sig, test_same, seed, adapt=True), s_test_clean))
        rows["DIFF"].append(cv_decode_corr(
            drive_record_readout(adapt_sig, test_diff, seed, adapt=True), s_test_clean))
        rows["NOADAPT_SAME"].append(cv_decode_corr(
            drive_record_readout(None, test_same, seed, adapt=False), s_test_clean))
    return {k: np.array(v) for k, v in rows.items()}

if __name__ == "__main__":
    # chance QCI for random {1..M}: P((b+e) mod M == c) = 1/M
    print(f"QA phase-conjugate distortion correction — same-medium test")
    print(f"engine QASystem, N={N}, M={M}, adapt={T_ADAPT}, test={T_TEST}, 18 seeds")
    print(f"Metric: reconstruction |corr| (CV distributed decode of clean signal).\n")
    print(f"{'sigma':>6s} | {'SAME':>14s} {'DIFF':>14s} {'NOADAPT_SAME':>14s} | "
          f"{'same-diff':>10s} {'p(same>diff)':>12s}")
    print("-"*84)
    for sigma in (0.5, 1.0, 2.0, 4.0):
        R = run(sigma)
        def ms(k): return f"{R[k].mean():.4f}+/-{R[k].std():.4f}"
        dg = R["SAME"].mean() - R["DIFF"].mean()
        t, p = ttest_rel(R["SAME"], R["DIFF"])
        p1 = p/2 if dg > 0 else 1 - p/2
        print(f"{sigma:6.1f} | {ms('SAME'):>14s} {ms('DIFF'):>14s} {ms('NOADAPT_SAME'):>14s} | "
              f"{dg:+10.4f} {p1:12.4f}")
    print("-"*84)
    print("PC theorem predicts SAME_g > DIFF_g (same-medium specificity).")
    print("SAME_g == DIFF_g => medium-agnostic => generic smoothing, PC claim falsified.")
