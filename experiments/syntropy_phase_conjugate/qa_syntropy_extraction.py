#!/usr/bin/env python3
# RT1_OBSERVER_FILE: continuous-signal analysis (coherent tones + noise driving the QA engine); trig acts on observer-layer signals, never QA state.
"""
QA Syntropy as coherence extraction — the FAITHFUL test, on the real engine.

Definition under test (from the source material): syntropy = "extraction of
usable signal from entropy (noise)". Operational, falsifiable form:

  Bury a coherent signal s(t) in noise: x(t) = s(t) + sigma*eta(t).
  Drive the self-organizing QASystem (qa_core.engine, signal_mode='final':
  signal injection via b-state -> resonance matrix -> Markovian coupling ->
  noise-annealed propagation -> E8/HI order). Read out the system response
  r(t) = mean_node b-state trajectory.

  Syntropy claim: the self-organizing coupling concentrates the coherent
  component, so corr(r, s_clean) exceeds what the noisy input already carries.

Controls that let it FAIL and isolate the QA-specific mechanism:
  RAW      : corr(x, s)            -- SNR already in the input
  COUPL-OFF: same engine, coupling=0 (injection+noise only, no resonance)
  COUPL-ON : full QA self-organization
  SHUFFLE  : coupling-on driven by time-shuffled x (kills temporal structure)
  MOVAVG   : classical moving-average of x matched to signal bandwidth

Syntropy is supported ONLY if COUPL-ON > {RAW, COUPL-OFF, SHUFFLE} beyond
noise -- i.e. the resonance coupling, not trivial smoothing or the raw SNR,
is what recovers the signal.

Uses the real qa_core.engine.QASystem. A1/S2/Theorem-NT compliance lives in
the engine (observer boundary crossed twice per step; integer state).
"""
from __future__ import annotations

QA_COMPLIANCE = "observer=continuous_signal_injection_into_QASystem_b_state, state_alphabet=mod24_A1; negative-result companion to cert [518], see docs/theory/QA_SYNTROPY_PHASE_CONJUGATE_INVESTIGATION.md"
import numpy as np
from scipy.stats import pearsonr, ttest_rel
from qa_core.engine import QASystem

M = 24
N = 24
T = 4000
NOISE_BASE = 0.5
INJ = 0.1

def coherent_signal(t):
    """Slow multi-tone coherent signal, normalized to unit std."""
    tt = np.arange(t)
    s = np.sin(2*np.pi*tt/220) + 0.6*np.sin(2*np.pi*tt/90 + 0.5)
    return (s - s.mean())/s.std()

def drive_state(signal_data, seed, coupling):
    """Return full distributed state trajectory: (T, 2N) array of (b|e)."""
    np.random.seed(seed)
    sys = QASystem(num_nodes=N, modulus=M, coupling=coupling,
                   noise_base=NOISE_BASE, noise_annealing=0.999,
                   signal_injection_strength=INJ, signal_mode="final")
    S = np.empty((len(signal_data), 2*N))
    for t in range(len(signal_data)):
        sys.step(t, float(signal_data[t]))
        S[t, :N] = sys.b
        S[t, N:] = sys.e
    return S

def movavg(x, w=31):
    k = np.ones(w)/w
    return np.convolve(x, k, mode="same")

def cv_decode_corr(state, clean, burn=500, l2=10.0):
    """Cross-validated |corr|: fit a ridge decode of clean from the distributed
    state on the 1st half (post burn-in), score |corr| on the held-out 2nd half.
    No leakage -> can't inflate by overfitting."""
    X = state[burn:]; y = clean[burn:]
    n = len(y); half = n//2
    Xtr, ytr = X[:half], y[:half]; Xte, yte = X[half:], y[half:]
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    A = Xtr.T@Xtr + l2*np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T@(ytr-ytr.mean()))
    pred = Xte@w
    if pred.std() < 1e-9: return 0.0
    return abs(pearsonr(pred, yte)[0])

def track_corr(vec1d, clean, burn=500):
    r, _ = pearsonr(vec1d[burn:], clean[burn:])
    return abs(r)

def run(sigma, seeds=range(20)):
    s = coherent_signal(T)
    rows = {k: [] for k in ["RAW", "COUPL_OFF", "COUPL_ON", "SHUFFLE", "MOVAVG"]}
    for seed in seeds:
        rng = np.random.default_rng(1000+seed)
        x = s + sigma*rng.standard_normal(T)
        xs = rng.permutation(x)
        rows["RAW"].append(track_corr(x, s))
        rows["MOVAVG"].append(track_corr(movavg(x), s))
        rows["COUPL_OFF"].append(cv_decode_corr(drive_state(x, seed, 0.0), s))
        rows["COUPL_ON"].append(cv_decode_corr(drive_state(x, seed, 0.15), s))
        rows["SHUFFLE"].append(cv_decode_corr(drive_state(xs, seed, 0.15), s))
    return {k: np.array(v) for k, v in rows.items()}

if __name__ == "__main__":
    print(f"QA syntropy / coherence-extraction  (N={N} nodes, T={T}, {20} seeds)")
    print("metric = |corr(response, clean signal)|, mean +/- std over seeds\n")
    print(f"{'sigma':>6s} | {'RAW':>13s} {'MOVAVG':>13s} {'COUPL_OFF':>13s} "
          f"{'COUPL_ON':>13s} {'SHUFFLE':>13s} | {'ON>OFF p':>10s} {'ON>RAW p':>10s}")
    print("-"*104)
    for sigma in (1.0, 2.0, 4.0, 8.0):
        R = run(sigma)
        def ms(k): return f"{R[k].mean():.3f}+/-{R[k].std():.3f}"
        # paired one-sided tests: is COUPL_ON better than OFF / RAW?
        _, p_off = ttest_rel(R["COUPL_ON"], R["COUPL_OFF"])
        _, p_raw = ttest_rel(R["COUPL_ON"], R["RAW"])
        p_off = p_off/2 if R["COUPL_ON"].mean() > R["COUPL_OFF"].mean() else 1-p_off/2
        p_raw = p_raw/2 if R["COUPL_ON"].mean() > R["RAW"].mean() else 1-p_raw/2
        print(f"{sigma:6.1f} | {ms('RAW'):>13s} {ms('MOVAVG'):>13s} {ms('COUPL_OFF'):>13s} "
              f"{ms('COUPL_ON'):>13s} {ms('SHUFFLE'):>13s} | {p_off:10.4f} {p_raw:10.4f}")
    print("-"*104)
    print("syntropy supported iff COUPL_ON > RAW and > COUPL_OFF and > SHUFFLE (p<0.05)")
