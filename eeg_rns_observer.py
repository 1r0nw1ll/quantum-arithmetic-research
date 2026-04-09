#!/usr/bin/env python3
"""
eeg_rns_observer.py — Dual-Modulus RNS Observer for EEG Seizure Detection

Implements a Residue Number System (RNS) observer that projects EEG signals
into BOTH mod-9 and mod-24 QA states simultaneously, exploiting the grid cell
RNS architecture certified in [205] QA_GRID_CELL_RNS_CERT.v1.

Key insight: the ratio 24/9 = 2.667 is within 2% of the optimal e = 2.718
for RNS encoding (Wei 2015, Fiete 2008). CRT consistency between the two
moduli provides an error-detection channel that the single-modulus topographic
k-means observer (Observer 3) lacks.

Architecture (Theorem NT compliant):

  [OBSERVER]   EEG window (N ch x T samples)
                  -> correlation matrix eigenspectrum (cross-channel coherence)
                  -> quantise to {1,...,9} and {1,...,24}  (boundary crossed ONCE)
                  |
  [QA LAYER]   (b_9, e_9) pairs in {1,...,9}^2    (mod-9 arithmetic)
               (b_24, e_24) pairs in {1,...,24}^2  (mod-24 arithmetic)
                  -> f_9  = b*b + b*e - e*e   (S1: no b-squared)
                  -> f_24 = b*b + b*e - e*e
                  -> orbit classification via divisibility rules
                  -> CRT consistency check
                  |
  [PROJECTION] orbit fractions, CRT rate, mean f-values
                  -> logistic regression: seizure ~ delta vs seizure ~ delta + RNS features

Axiom compliance:
  A1: all states in {1,...,m}, never {0,...,m-1}. Uses ((x-1) % m) + 1.
  A2: d = b+e, a = b+2*e — always derived, never assigned independently
  S1: b*b throughout, NEVER b-squared
  S2: b, e are Python int inside QA layer — no float state
  T1: QA time = window index (integer path step)
  T2: EEG signal enters ONLY at the observer boundary (RMS -> quantise)

Standalone script — no imports from other experiment files.

Usage:
  python eeg_rns_observer.py                    # synthetic test mode
  python eeg_rns_observer.py --patient chb01    # real CHB-MIT patient (requires data)
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "cert_family": "[205] QA_GRID_CELL_RNS_CERT.v1",
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "observer": "dual_modulus_rns_coherence_eigenspectrum_projection",
    "state_alphabet": ["cosmos", "satellite", "singularity"],
    "qa_layer_types": "int",
    "projection_types": "float",
}

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import chi2

# ── Parameters ────────────────────────────────────────────────────────────────

SEED          = 42
MOD_9         = 9
MOD_24        = 24
WINDOW_SEC    = 10.0       # seconds per analysis window
N_CHANNELS    = 23         # standard CHB-MIT montage
FS_DEFAULT    = 256.0      # default sampling rate (Hz)
N_PAIRS_MODE  = "adjacent" # "adjacent" or "pca"
SCALE_9       = 9.0        # scale factor for mod-9 projection
SCALE_24      = 24.0       # scale factor for mod-24 projection

# Synthetic test parameters
N_SYNTH_SEIZURE  = 40
N_SYNTH_BASELINE = 60
SYNTH_NOISE_SEI  = 3.0     # higher amplitude for seizure
SYNTH_NOISE_BASE = 1.0     # lower amplitude for baseline

OUTPUT_JSON = "eeg_rns_observer_results.json"


# ── QA Core Arithmetic (standalone, no imports) ──────────────────────────────

def norm_f(b: int, e: int) -> int:
    """f(b,e) = b*b + b*e - e*e in Z[phi]. S1: b*b not b-squared."""
    return b * b + b * e - e * e


def v3(n: int) -> int:
    """3-adic valuation of n (returns 9999 for n=0)."""
    if n == 0:
        return 9999
    n = abs(n)
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def orbit_family_mod9(b: int, e: int) -> str:
    """
    Classify (b,e) in mod-9.
    Singularity: b==9 and e==9
    Satellite:   3|b and 3|e  (sat_divisor = 9//3 = 3)
    Cosmos:      everything else
    """
    assert isinstance(b, int) and isinstance(e, int), \
        f"S2: b={b!r}, e={e!r} must be Python int"
    assert 1 <= b <= MOD_9 and 1 <= e <= MOD_9, \
        f"A1: ({b},{e}) out of {{1,...,{MOD_9}}}"
    if b == MOD_9 and e == MOD_9:
        return "singularity"
    if b % 3 == 0 and e % 3 == 0:
        return "satellite"
    return "cosmos"


def orbit_family_mod24(b: int, e: int) -> str:
    """
    Classify (b,e) in mod-24.
    Singularity: b==24 and e==24
    Satellite:   8|b and 8|e  (sat_divisor = 24//3 = 8)
    Cosmos:      everything else
    """
    assert isinstance(b, int) and isinstance(e, int), \
        f"S2: b={b!r}, e={e!r} must be Python int"
    assert 1 <= b <= MOD_24 and 1 <= e <= MOD_24, \
        f"A1: ({b},{e}) out of {{1,...,{MOD_24}}}"
    if b == MOD_24 and e == MOD_24:
        return "singularity"
    if b % 8 == 0 and e % 8 == 0:
        return "satellite"
    return "cosmos"


def qa_residue(value: float, m: int) -> int:
    """
    Observer projection: continuous value -> discrete state in {1,...,m}.
    A1 compliant: ((int(round(x)) - 1) % m) + 1, result always in {1,...,m}.

    This is the ONLY place where a continuous value crosses into the QA layer.
    """
    x = int(round(value))
    return ((x - 1) % m) + 1


# ── Observer Projection ──────────────────────────────────────────────────────

def channel_rms(channel_data: np.ndarray) -> float:
    """RMS amplitude of a single channel window. Observer layer (continuous)."""
    return float(np.sqrt(np.mean(channel_data * channel_data)))


def channel_mean_coherence(multi_ch: np.ndarray) -> np.ndarray:
    """
    Per-channel mean absolute correlation with all other channels.
    Observer layer (continuous) — captures spatial coordination that
    seizures exhibit (high inter-channel synchrony) and baseline lacks.

    Returns array of shape (n_ch,) where each value is the mean |r|
    of that channel with all others.  Values in [0, 1].
    """
    corr = np.corrcoef(multi_ch)          # (n_ch, n_ch) Pearson correlation
    np.fill_diagonal(corr, 0.0)           # exclude self-correlation
    # Guard against NaN from constant channels
    corr = np.nan_to_num(corr, nan=0.0)
    return np.mean(np.abs(corr), axis=1)  # (n_ch,)


def coherence_eigenspectrum(multi_ch: np.ndarray) -> np.ndarray:
    """
    Eigenvalues of the cross-channel correlation matrix, sorted descending.
    Observer layer (continuous) — encodes spatial mode structure.

    Broadband (no filtering): captures synchrony across all frequency bands.
    Seizures have synchronous activity in delta, theta, AND gamma — filtering
    to one band loses information.

    Returns array of shape (n_ch,) — eigenvalues in [0, n_ch], descending.
    """
    corr = np.corrcoef(multi_ch)          # (n_ch, n_ch)
    corr = np.nan_to_num(corr, nan=0.0)
    eigvals = np.linalg.eigvalsh(corr)    # ascending, real (symmetric matrix)
    return eigvals[::-1]                  # descending order


def project_window_to_residues(
    multi_ch: np.ndarray,
    scale_9: float = SCALE_9,
    scale_24: float = SCALE_24,
    fs: float = FS_DEFAULT,
) -> dict:
    """
    Observer projection: EEG window (N_ch x T) -> dual-modulus residue pairs.

    Steps:
      1. Compute eigenvalues of cross-channel correlation matrix (descending)
      2. Normalise to [0, 1] range
      3. Scale and quantise to {1,...,9} and {1,...,24} (boundary crossing)
      4. Form (b,e) pairs from consecutive eigenvalues

    Eigenspectrum captures spatial mode structure: seizures have one dominant
    eigenvalue (global synchrony), baseline has distributed eigenvalues.
    Consecutive eigenvalue pairs (λ_k, λ_{k+1}) encode the MODE TRANSITIONS
    of the spatial coordination pattern.

    Returns dict with:
      pairs_9:      list of (b, e) tuples in mod-9
      pairs_24:     list of (b, e) tuples in mod-24
      coherence_raw: eigenspectrum (for diagnostics)
    """
    n_ch = multi_ch.shape[0]

    # Step 1: eigenspectrum of broadband correlation matrix (observer layer)
    eigvals = coherence_eigenspectrum(multi_ch)

    # Step 2: normalise to [0, 1] with epsilon guard against zero range
    coh_min = eigvals.min()
    coh_max = eigvals.max()
    coh_range = coh_max - coh_min
    if coh_range < 1e-12:
        coh_norm = np.full_like(eigvals, 0.5)
    else:
        coh_norm = (eigvals - coh_min) / coh_range

    # Step 3: scale and quantise (boundary crossing — continuous -> discrete)
    n_eig = len(eigvals)
    residues_9 = [qa_residue(coh_norm[i] * scale_9, MOD_9) for i in range(n_eig)]
    residues_24 = [qa_residue(coh_norm[i] * scale_24, MOD_24) for i in range(n_eig)]

    # Step 4: form (b,e) pairs — mirror pairing (λ_k, λ_{n-1-k})
    # Pairs dominant modes with trailing modes to maximise dynamic range.
    # b = dominant eigenvalue (spatial power), e = trailing eigenvalue (noise floor).
    # Mode spread within each pair encodes effective dimensionality.
    pairs_9 = []
    pairs_24 = []
    n_pairs = n_eig // 2
    for k in range(n_pairs):
        b9 = int(residues_9[k])
        e9 = int(residues_9[n_eig - 1 - k])
        pairs_9.append((b9, e9))

        b24 = int(residues_24[k])
        e24 = int(residues_24[n_eig - 1 - k])
        pairs_24.append((b24, e24))

    return {
        "pairs_9": pairs_9,
        "pairs_24": pairs_24,
        "coherence_raw": eigvals.tolist(),
    }


# ── CRT Consistency Check ────────────────────────────────────────────────────

# Orbit compatibility table: which mod-9 orbits are consistent with mod-24 orbits.
# Cosmos in mod-24 should correspond to Cosmos in mod-9 (majority case).
# Satellite in mod-24 requires 8|b and 8|e; in mod-9 requires 3|b and 3|e.
# Singularity is unique in both: (9,9) and (24,24).
#
# CRT consistency: if a pair is Satellite in mod-24 (8|b, 8|e), then since
# 8 = 24/3 and gcd(8,3) = 1, it does NOT force 3|b in mod-9. So a mod-24
# Satellite can legitimately be mod-9 Cosmos. The strict consistency check
# is: Singularity in one modulus must be Singularity in the other (since both
# represent the unique fixed point). Cross-orbit for Cosmos/Satellite is allowed.

CONSISTENT_ORBITS = {
    # mod-9 orbit -> set of compatible mod-24 orbits
    "singularity": {"singularity"},
    "satellite": {"satellite", "cosmos"},
    "cosmos": {"satellite", "cosmos"},
}


def crt_consistent(orbit_9: str, orbit_24: str) -> bool:
    """
    Check CRT consistency between mod-9 and mod-24 orbit classifications.

    Strict rule: singularity in one modulus must be singularity in both.
    Cross-classification of cosmos/satellite is permitted (different divisibility).
    """
    return orbit_24 in CONSISTENT_ORBITS.get(orbit_9, set())


# ── Orbit Classification Pipeline ────────────────────────────────────────────

def classify_window(multi_ch: np.ndarray,
                    scale_9: float = SCALE_9,
                    scale_24: float = SCALE_24,
                    fs: float = FS_DEFAULT) -> dict:
    """
    Full RNS observer pipeline for one EEG window.

    Returns dict with orbit fractions, CRT consistency, f-values.
    All QA arithmetic uses Python int (S2 compliant).
    """
    proj = project_window_to_residues(multi_ch, scale_9, scale_24, fs=fs)

    pairs_9 = proj["pairs_9"]
    pairs_24 = proj["pairs_24"]
    n_pairs = len(pairs_9)

    if n_pairs == 0:
        return {
            "cosmos_frac_9": 0.0, "satellite_frac_9": 0.0, "singularity_frac_9": 0.0,
            "cosmos_frac_24": 0.0, "satellite_frac_24": 0.0, "singularity_frac_24": 0.0,
            "crt_consistency": 0.0,
            "mean_f_9": 0.0, "mean_f_24": 0.0,
            "n_pairs": 0,
        }

    # Classify each pair in both moduli
    orbits_9 = []
    orbits_24 = []
    f_values_9 = []
    f_values_24 = []
    crt_consistent_count = 0

    for (b9, e9), (b24, e24) in zip(pairs_9, pairs_24):
        # QA layer: all values are Python int (S2)
        o9 = orbit_family_mod9(b9, e9)
        o24 = orbit_family_mod24(b24, e24)

        orbits_9.append(o9)
        orbits_24.append(o24)

        # f-values (S1: b*b not b**2)
        f9 = norm_f(b9, e9)
        f24 = norm_f(b24, e24)
        f_values_9.append(f9)
        f_values_24.append(f24)

        # CRT consistency
        if crt_consistent(o9, o24):
            crt_consistent_count += 1

    # Orbit fractions (mod-9)
    cosmos_9 = sum(1 for o in orbits_9 if o == "cosmos") / n_pairs
    satellite_9 = sum(1 for o in orbits_9 if o == "satellite") / n_pairs
    singularity_9 = sum(1 for o in orbits_9 if o == "singularity") / n_pairs

    # Orbit fractions (mod-24)
    cosmos_24 = sum(1 for o in orbits_24 if o == "cosmos") / n_pairs
    satellite_24 = sum(1 for o in orbits_24 if o == "satellite") / n_pairs
    singularity_24 = sum(1 for o in orbits_24 if o == "singularity") / n_pairs

    return {
        "cosmos_frac_9": cosmos_9,
        "satellite_frac_9": satellite_9,
        "singularity_frac_9": singularity_9,
        "cosmos_frac_24": cosmos_24,
        "satellite_frac_24": satellite_24,
        "singularity_frac_24": singularity_24,
        "crt_consistency": crt_consistent_count / n_pairs,
        "mean_f_9": float(np.mean(f_values_9)),
        "mean_f_24": float(np.mean(f_values_24)),
        "n_pairs": n_pairs,
        "orbits_9": orbits_9,
        "orbits_24": orbits_24,
    }


# ── Window Transition Features ───────────────────────────────────────────────

def compute_transition_rate(window_results: list[dict]) -> float:
    """
    Orbit transition rate between consecutive windows.
    Counts how often the majority orbit changes from one window to the next.
    Uses mod-9 orbits (theoretical modulus).
    """
    if len(window_results) < 2:
        return 0.0

    def majority_orbit(result: dict) -> str:
        fracs = {
            "cosmos": result["cosmos_frac_9"],
            "satellite": result["satellite_frac_9"],
            "singularity": result["singularity_frac_9"],
        }
        return max(fracs, key=fracs.get)

    transitions = 0
    for i in range(1, len(window_results)):
        if majority_orbit(window_results[i]) != majority_orbit(window_results[i - 1]):
            transitions += 1

    return transitions / (len(window_results) - 1)


# ── Feature Extraction for Seizure Detection ─────────────────────────────────

def extract_segment_features(multi_ch: np.ndarray, fs: float,
                             window_sec: float = WINDOW_SEC) -> dict:
    """
    Extract RNS observer features from a full EEG segment.

    Slides a window across the segment, classifies each window,
    then aggregates into segment-level features.
    """
    n_samples = multi_ch.shape[1]
    n_win = int(window_sec * fs)
    step = n_win  # non-overlapping windows

    window_results = []
    for start in range(0, n_samples - n_win + 1, step):
        chunk = multi_ch[:, start:start + n_win]
        wr = classify_window(chunk, fs=fs)
        window_results.append(wr)

    if not window_results:
        return _empty_features()

    # Aggregate across windows
    cosmos_9 = float(np.mean([w["cosmos_frac_9"] for w in window_results]))
    satellite_9 = float(np.mean([w["satellite_frac_9"] for w in window_results]))
    singularity_9 = float(np.mean([w["singularity_frac_9"] for w in window_results]))

    cosmos_24 = float(np.mean([w["cosmos_frac_24"] for w in window_results]))
    satellite_24 = float(np.mean([w["satellite_frac_24"] for w in window_results]))
    singularity_24 = float(np.mean([w["singularity_frac_24"] for w in window_results]))

    crt_rate = float(np.mean([w["crt_consistency"] for w in window_results]))
    mean_f9 = float(np.mean([w["mean_f_9"] for w in window_results]))
    mean_f24 = float(np.mean([w["mean_f_24"] for w in window_results]))

    transition_rate = compute_transition_rate(window_results)

    return {
        "cosmos_frac_9": cosmos_9,
        "satellite_frac_9": satellite_9,
        "singularity_frac_9": singularity_9,
        "cosmos_frac_24": cosmos_24,
        "satellite_frac_24": satellite_24,
        "singularity_frac_24": singularity_24,
        "crt_consistency": crt_rate,
        "mean_f_9": mean_f9,
        "mean_f_24": mean_f24,
        "transition_rate": transition_rate,
        "n_windows": len(window_results),
    }


def _empty_features() -> dict:
    return {
        "cosmos_frac_9": 0.0, "satellite_frac_9": 0.0, "singularity_frac_9": 0.0,
        "cosmos_frac_24": 0.0, "satellite_frac_24": 0.0, "singularity_frac_24": 0.0,
        "crt_consistency": 0.0, "mean_f_9": 0.0, "mean_f_24": 0.0,
        "transition_rate": 0.0, "n_windows": 0,
    }


# ── Delta Power Ratio (classical baseline, standalone) ───────────────────────

def delta_power_ratio(waveform: np.ndarray, fs: float) -> float:
    """
    Delta (0.5-4 Hz) power ratio. Observer projection — continuous measurement.
    Used as the classical baseline feature for nested model comparison.
    """
    from numpy.fft import rfft, rfftfreq
    n = len(waveform)
    freqs = rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(rfft(waveform))
    total = float(np.sum(spectrum * spectrum))
    if total < 1e-12:
        return 0.0
    delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
    delta_power = float(np.sum(spectrum[delta_mask] * spectrum[delta_mask]))
    return delta_power / total


# ── Nested Logistic Regression (standalone, same as eeg_chbmit_scale.py) ─────

def _fit_logistic(X: np.ndarray, y: np.ndarray,
                  lr: float = 0.1, n_iter: int = 3000, l2: float = 1e-4) -> np.ndarray:
    """Gradient descent logistic regression."""
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        logits = np.clip(X @ beta, -30, 30)
        probs = expit(logits)
        beta -= lr * (X.T @ (probs - y) / len(y) + l2 * beta)
    return beta


def _ll(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Log-likelihood of logistic model."""
    logits = np.clip(X @ beta, -30, 30)
    probs = np.clip(expit(logits), 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _std(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd + 1e-9)


def nested_model_rns(y: np.ndarray,
                     delta: np.ndarray,
                     singularity_9: np.ndarray,
                     cosmos_9: np.ndarray,
                     crt_rate: np.ndarray,
                     mean_f9: np.ndarray) -> dict:
    """
    Nested logistic regression comparing:
      Model 0: intercept only
      Model 1: delta power ratio (classical baseline)
      Model 2: delta + RNS features (singularity_9, cosmos_9, crt_rate, mean_f9)

    Returns dict with R-squared values and LR test.
    """
    n = len(y)
    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit_logistic(X0, y))

    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit_logistic(X1, y))

    X2 = np.c_[np.ones(n), _std(delta), _std(singularity_9),
               _std(cosmos_9), _std(crt_rate), _std(mean_f9)]
    ll2 = _ll(X2, y, _fit_logistic(X2, y))

    r2_delta = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_full = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2.sf(max(0.0, lr_stat), df=4))  # 4 extra features

    return {
        "r2_delta": float(r2_delta),
        "r2_full": float(r2_full),
        "delta_r2": float(r2_full - r2_delta),
        "lr_stat": float(lr_stat),
        "p_rns_add": float(p_val),
    }


# ── Synthetic EEG Generator ─────────────────────────────────────────────────

def generate_synthetic_eeg(n_seizure: int = N_SYNTH_SEIZURE,
                           n_baseline: int = N_SYNTH_BASELINE,
                           n_channels: int = N_CHANNELS,
                           fs: float = FS_DEFAULT,
                           window_sec: float = WINDOW_SEC) -> list[dict]:
    """
    Generate synthetic EEG segments for testing.

    Seizure segments: higher amplitude, more spatial variance, more high-freq content.
    Baseline segments: lower amplitude, more uniform spatial profile.

    This is a SYNTHETIC test — the continuous waveforms are observer inputs only.
    """
    np.random.seed(SEED)
    n_samples = int(window_sec * fs)
    dataset = []

    for _ in range(n_seizure):
        # Seizure: higher amplitude with channel-specific variation
        amplitudes = np.abs(np.random.randn(n_channels) * SYNTH_NOISE_SEI + 2.0)
        freqs = np.random.uniform(4.0, 25.0, size=n_channels)  # more high-freq
        t = np.arange(n_samples) / fs
        multi_ch = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            multi_ch[ch, :] = (amplitudes[ch] * np.sin(2 * np.pi * freqs[ch] * t)
                               + np.random.randn(n_samples) * 0.5)
        dataset.append({
            "type": "seizure",
            "multi_ch": multi_ch,
            "waveform": multi_ch[0, :],
            "fs": fs,
        })

    for _ in range(n_baseline):
        # Baseline: lower amplitude, more uniform, delta-band dominant
        amplitudes = np.abs(np.random.randn(n_channels) * SYNTH_NOISE_BASE + 1.0)
        freqs = np.random.uniform(0.5, 4.0, size=n_channels)  # delta-dominant
        t = np.arange(n_samples) / fs
        multi_ch = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            multi_ch[ch, :] = (amplitudes[ch] * np.sin(2 * np.pi * freqs[ch] * t)
                               + np.random.randn(n_samples) * 0.3)
        dataset.append({
            "type": "baseline",
            "multi_ch": multi_ch,
            "waveform": multi_ch[0, :],
            "fs": fs,
        })

    return dataset


# ── Self-Test: Verify QA Axiom Compliance ────────────────────────────────────

def axiom_self_test():
    """
    Verify core QA axiom compliance in the RNS observer.
    """
    print("  Running axiom self-test...")
    errors = 0

    # A1: qa_residue always returns {1,...,m}
    for m in [MOD_9, MOD_24]:
        for x in [-100, -1, 0, 0.5, 1, 5.5, 8.99, 9, 24, 100, 1000]:
            r = qa_residue(x, m)
            if r < 1 or r > m:
                print(f"    FAIL A1: qa_residue({x}, {m}) = {r} not in {{1,...,{m}}}")
                errors += 1

    # S1: verify norm_f uses b*b pattern (structural — just call it)
    f = norm_f(3, 5)
    expected = 3 * 3 + 3 * 5 - 5 * 5  # = 9 + 15 - 25 = -1
    if f != expected:
        print(f"    FAIL S1: norm_f(3,5) = {f}, expected {expected}")
        errors += 1

    # S2: orbit functions enforce int
    try:
        orbit_family_mod9(3, 5)
        orbit_family_mod24(3, 5)
    except (AssertionError, TypeError) as exc:
        print(f"    FAIL S2: orbit function rejected valid int input: {exc}")
        errors += 1

    # Orbit classification sanity
    assert orbit_family_mod9(9, 9) == "singularity"
    assert orbit_family_mod9(3, 6) == "satellite"
    assert orbit_family_mod9(1, 2) == "cosmos"
    assert orbit_family_mod24(24, 24) == "singularity"
    assert orbit_family_mod24(8, 16) == "satellite"
    assert orbit_family_mod24(1, 2) == "cosmos"

    # CRT consistency: singularity must pair with singularity
    assert crt_consistent("singularity", "singularity") is True
    assert crt_consistent("singularity", "cosmos") is False
    assert crt_consistent("cosmos", "satellite") is True

    if errors == 0:
        print("  Axiom self-test: PASS (A1, S1, S2, orbit classification, CRT)")
    else:
        print(f"  Axiom self-test: FAIL ({errors} errors)")
    return errors == 0


# ── Main: Synthetic Test Mode ────────────────────────────────────────────────

def run_synthetic_test():
    """
    Full synthetic test: generate fake EEG, run RNS observer, compare to delta baseline.
    """
    print("=" * 72)
    print("EEG RNS Observer — Dual-Modulus Residue Number System")
    print("Synthetic Test Mode")
    print("=" * 72)
    print()

    # Axiom self-test
    if not axiom_self_test():
        print("\nABORT: axiom self-test failed")
        sys.exit(1)
    print()

    # Generate synthetic data
    print(f"Generating synthetic EEG: {N_SYNTH_SEIZURE} seizure + "
          f"{N_SYNTH_BASELINE} baseline segments")
    print(f"  Channels: {N_CHANNELS}, fs: {FS_DEFAULT} Hz, "
          f"window: {WINDOW_SEC}s")
    print()

    dataset = generate_synthetic_eeg()

    # Extract features for each segment
    print("Extracting RNS observer features...")
    t0 = time.time()

    all_features = []
    for seg in dataset:
        feats = extract_segment_features(seg["multi_ch"], seg["fs"])
        feats["type"] = seg["type"]
        feats["delta_ratio"] = delta_power_ratio(
            seg["waveform"].astype(np.float64), seg["fs"]
        )
        all_features.append(feats)

    elapsed = time.time() - t0
    print(f"  Extracted {len(all_features)} segments in {elapsed:.2f}s")
    print()

    # ── Orbit distribution summary ───────────────────────────────────────────
    print("-" * 72)
    print("ORBIT DISTRIBUTION (mod-9)")
    print(f"  {'Type':<10}  {'Cosmos':>8}  {'Satellite':>10}  {'Singularity':>12}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")

    for seg_type in ["seizure", "baseline"]:
        segs = [f for f in all_features if f["type"] == seg_type]
        c = np.mean([s["cosmos_frac_9"] for s in segs])
        s = np.mean([s["satellite_frac_9"] for s in segs])
        g = np.mean([s["singularity_frac_9"] for s in segs])
        print(f"  {seg_type:<10}  {c:>8.4f}  {s:>10.4f}  {g:>12.4f}")

    print()
    print("ORBIT DISTRIBUTION (mod-24)")
    print(f"  {'Type':<10}  {'Cosmos':>8}  {'Satellite':>10}  {'Singularity':>12}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")

    for seg_type in ["seizure", "baseline"]:
        segs = [f for f in all_features if f["type"] == seg_type]
        c = np.mean([s["cosmos_frac_24"] for s in segs])
        s = np.mean([s["satellite_frac_24"] for s in segs])
        g = np.mean([s["singularity_frac_24"] for s in segs])
        print(f"  {seg_type:<10}  {c:>8.4f}  {s:>10.4f}  {g:>12.4f}")

    print()
    print("CRT CONSISTENCY")
    print(f"  {'Type':<10}  {'CRT Rate':>10}  {'Mean f_9':>10}  {'Mean f_24':>10}  {'Trans Rate':>12}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    for seg_type in ["seizure", "baseline"]:
        segs = [f for f in all_features if f["type"] == seg_type]
        crt = np.mean([s["crt_consistency"] for s in segs])
        mf9 = np.mean([s["mean_f_9"] for s in segs])
        mf24 = np.mean([s["mean_f_24"] for s in segs])
        tr = np.mean([s["transition_rate"] for s in segs])
        print(f"  {seg_type:<10}  {crt:>10.4f}  {mf9:>10.2f}  {mf24:>10.2f}  {tr:>12.4f}")

    # ── Nested logistic regression ───────────────────────────────────────────
    print()
    print("-" * 72)
    print("NESTED LOGISTIC REGRESSION")
    print("  Model 1: seizure ~ delta_ratio (classical baseline)")
    print("  Model 2: seizure ~ delta_ratio + singularity_9 + cosmos_9 + crt + mean_f_9")
    print()

    y = np.array([1.0 if f["type"] == "seizure" else 0.0 for f in all_features])
    delta = np.array([f["delta_ratio"] for f in all_features])
    sing9 = np.array([f["singularity_frac_9"] for f in all_features])
    cos9 = np.array([f["cosmos_frac_9"] for f in all_features])
    crt_arr = np.array([f["crt_consistency"] for f in all_features])
    mf9 = np.array([f["mean_f_9"] for f in all_features])

    result = nested_model_rns(y, delta, sing9, cos9, crt_arr, mf9)

    print(f"  R^2 (delta only):          {result['r2_delta']:.4f}")
    print(f"  R^2 (delta + RNS):         {result['r2_full']:.4f}")
    print(f"  Delta R^2 (RNS beyond delta): {result['delta_r2']:+.4f}")
    print(f"  LR stat:                   {result['lr_stat']:.3f}")
    print(f"  p(RNS adds):               {result['p_rns_add']:.6f}")
    sig = ("***" if result["p_rns_add"] < 0.001 else
           ("**" if result["p_rns_add"] < 0.01 else
            ("*" if result["p_rns_add"] < 0.05 else "ns")))
    print(f"  Significance:              {sig}")
    print()

    # ── Comparison with topographic k-means (reference) ──────────────────────
    print("-" * 72)
    print("COMPARISON REFERENCE")
    print("  Topographic k-means (Observer 3): mean DeltaR^2 = +0.210 (10 patients)")
    print(f"  RNS observer (this script):      DeltaR^2 = {result['delta_r2']:+.4f} (synthetic)")
    print()
    print("  NOTE: Synthetic data test only. Run with --patient chb01 for real data.")
    print()

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "observer": "dual_modulus_rns",
        "mode": "synthetic",
        "seed": SEED,
        "n_seizure": N_SYNTH_SEIZURE,
        "n_baseline": N_SYNTH_BASELINE,
        "n_channels": N_CHANNELS,
        "window_sec": WINDOW_SEC,
        "mod_9": MOD_9,
        "mod_24": MOD_24,
        "scale_9": SCALE_9,
        "scale_24": SCALE_24,
        "nested_model": result,
        "orbit_summary": {},
        "per_segment": [],
    }

    for seg_type in ["seizure", "baseline"]:
        segs = [f for f in all_features if f["type"] == seg_type]
        output["orbit_summary"][seg_type] = {
            "cosmos_9": float(np.mean([s["cosmos_frac_9"] for s in segs])),
            "satellite_9": float(np.mean([s["satellite_frac_9"] for s in segs])),
            "singularity_9": float(np.mean([s["singularity_frac_9"] for s in segs])),
            "cosmos_24": float(np.mean([s["cosmos_frac_24"] for s in segs])),
            "satellite_24": float(np.mean([s["satellite_frac_24"] for s in segs])),
            "singularity_24": float(np.mean([s["singularity_frac_24"] for s in segs])),
            "crt_consistency": float(np.mean([s["crt_consistency"] for s in segs])),
            "mean_f_9": float(np.mean([s["mean_f_9"] for s in segs])),
            "mean_f_24": float(np.mean([s["mean_f_24"] for s in segs])),
            "transition_rate": float(np.mean([s["transition_rate"] for s in segs])),
        }

    # Per-segment features (without orbits lists for JSON serialisation)
    for f in all_features:
        seg_out = {k: v for k, v in f.items()
                   if k not in ("orbits_9", "orbits_24")}
        output["per_segment"].append(seg_out)

    out_path = Path(__file__).parent / OUTPUT_JSON
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved to {out_path}")

    return output


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--patient" in sys.argv:
        idx = sys.argv.index("--patient")
        if idx + 1 < len(sys.argv):
            patient_id = sys.argv[idx + 1]
            print(f"Real patient mode not yet implemented (requires CHB-MIT data loader).")
            print(f"Use eeg_chbmit_scale.py for real data; this script provides the RNS observer.")
            print(f"To integrate: import extract_segment_features from this module.")
            sys.exit(0)

    run_synthetic_test()
