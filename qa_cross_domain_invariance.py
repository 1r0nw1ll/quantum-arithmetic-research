#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cross_domain_general, state_alphabet=mod24"
"""
QA Cross-Domain Invariance Test — The Kill-or-Prove Experiment

PRE-REGISTERED DESIGN (Tier 4 candidate):

FIXED GLOBALLY (zero per-domain tuning):
  - T-operator: Fibonacci shift (b,e) -> (e, ((b+e-1)%m)+1)
  - Modulus m = 24
  - K clusters = 6
  - QCI window = 63 steps
  - Cluster-to-QA mapping: sequential 1-indexed (cluster 0→4, 1→8, 2→12, 3→16, 4→20, 5→24)
    (evenly spaced in {1,...,24}, NO hand-tuning)
  - Standardization: z-score with 63-step rolling mean/std
  - Train/test split: first half train, second half OOS

DOMAINS:
  1. Finance: multi-asset returns (SPY, QQQ, IWM, TLT, GLD, BTC-USD)
  2. EEG: multi-channel scalp EEG (CHB-MIT, synthesized from existing results)
  3. Seismology: USGS earthquake features (count, mean mag, max mag, depth)
  4. Audio: synthesized signals (sine, noise, mixed)

INVARIANTS MEASURED:
  A. QCI predictive sign (negative = stress → disorder?)
  B. Orbit distribution at stress vs calm
  C. Satellite fraction at stress events
  D. QCI autocorrelation structure (persistence)

SUCCESS: Same qualitative pattern across all 4 domains with zero retuning.
FAILURE: Any domain shows opposite pattern or no signal.

This script generates all domain data internally (no external downloads)
to ensure reproducibility.
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import json
import sys

# ══════════════════════════════════════════════════════
# GLOBALLY FIXED PARAMETERS — DO NOT CHANGE PER DOMAIN
# ══════════════════════════════════════════════════════
MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
RANDOM_SEED = 42

# Cluster-to-QA mapping: evenly spaced in {1,...,24}
# No hand-tuning — mechanical assignment
CLUSTER_MAP = {i: 4*(i+1) for i in range(N_CLUSTERS)}
# {0:4, 1:8, 2:12, 3:16, 4:20, 5:24}


def qa_mod(x, m):
    """A1-compliant modular reduction to {1,...,m}."""
    return ((int(x) - 1) % m) + 1


def qa_step(b, e, m):
    """T-operator: Fibonacci shift."""
    return e, qa_mod(b + e, m)


def compute_qci(labels, cmap, m, window):
    """Compute QA Coherence Index from cluster label sequence."""
    t_match = []
    for t in range(len(labels) - 2):
        b_t = cmap[labels[t]]
        e_t = cmap[labels[t + 1]]
        actual_next = cmap[labels[t + 2]]
        _, pred_next = qa_step(b_t, e_t, m)
        t_match.append(1 if pred_next == actual_next else 0)

    t_match = np.array(t_match, dtype=float)

    # Rolling mean
    qci = np.full(len(t_match), np.nan)
    for i in range(window, len(t_match)):
        qci[i] = np.mean(t_match[i-window:i])

    return qci


def classify_orbit(b, e, m):
    """Classify (b,e) pair into orbit type."""
    seen = set()
    state = (b, e)
    length = 0
    while state not in seen and length < m * m:
        seen.add(state)
        state = qa_step(state[0], state[1], m)
        length += 1
    if length == 1:
        return 'singularity'
    elif length <= 8:
        return 'satellite'
    else:
        return 'cosmos'


def orbit_distribution(labels, cmap, m):
    """Get orbit type distribution from a label sequence."""
    counts = {'cosmos': 0, 'satellite': 0, 'singularity': 0}
    for t in range(len(labels) - 1):
        b = cmap[labels[t]]
        e = cmap[labels[t + 1]]
        otype = classify_orbit(b, e, m)
        counts[otype] += 1
    total = sum(counts.values())
    return {k: v/total for k, v in counts.items()}


def run_domain(name, data_matrix, stress_indicator, description=""):
    """
    Run the invariance test on one domain.

    Args:
        name: domain name
        data_matrix: (T, D) array of observations
        stress_indicator: (T,) binary array (1=stress event)
        description: what the data represents
    """
    print(f'\n{"="*70}')
    print(f'DOMAIN: {name}')
    print(f'  {description}')
    print(f'  Shape: {data_matrix.shape}, stress events: {stress_indicator.sum()}/{len(stress_indicator)}')
    print(f'{"="*70}')

    T, D = data_matrix.shape
    half = T // 2

    # Standardize with rolling z-score (same window as QCI)
    std_data = np.zeros_like(data_matrix)
    for i in range(QCI_WINDOW, T):
        window_data = data_matrix[i-QCI_WINDOW:i]
        mu = window_data.mean(axis=0)
        sigma = window_data.std(axis=0) + 1e-10
        std_data[i] = (data_matrix[i] - mu) / sigma

    # Use only the standardized portion
    valid = std_data[QCI_WINDOW:]
    stress_valid = stress_indicator[QCI_WINDOW:]

    # K-means on first half
    train_end = len(valid) // 2
    np.random.seed(RANDOM_SEED)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_SEED)
    km.fit(valid[:train_end])
    all_labels = km.predict(valid)

    # Compute QCI
    qci = compute_qci(all_labels, CLUSTER_MAP, MODULUS, QCI_WINDOW)

    # QCI is 2 shorter than labels (needs triples)
    # Align stress and labels to QCI length
    qci_len = len(qci)
    stress_aligned = stress_valid[:qci_len]
    labels_aligned = all_labels[:qci_len]

    # OOS only
    oos_start = train_end
    oos_qci = qci[oos_start:]
    oos_stress = stress_aligned[oos_start:]
    oos_labels_raw = labels_aligned[oos_start:]

    # Trim NaNs
    valid_mask = ~np.isnan(oos_qci)
    oos_qci = oos_qci[valid_mask]
    oos_stress = oos_stress[valid_mask]
    oos_labels = oos_labels_raw[valid_mask]

    if len(oos_qci) < 50:
        print(f'  SKIP: insufficient OOS data ({len(oos_qci)} points)')
        return None

    # ── INVARIANT A: QCI correlation with stress ──
    if oos_stress.std() > 0 and not np.isnan(oos_qci).any():
        r, p = stats.pearsonr(oos_qci, oos_stress.astype(float))
    else:
        r, p = 0.0, 1.0
    print(f'\n  INVARIANT A: QCI vs stress correlation')
    print(f'    r = {r:+.4f}, p = {p:.6f}')
    print(f'    Sign: {"NEGATIVE (low QCI → stress)" if r < 0 else "POSITIVE (high QCI → stress)"}')

    # ── INVARIANT B: Orbit distribution at stress vs calm ──
    stress_mask = oos_stress > 0
    calm_mask = ~stress_mask

    stress_labels = oos_labels[stress_mask[: len(oos_labels)]] if stress_mask.sum() > 10 else np.array([])
    calm_labels = oos_labels[calm_mask[:len(oos_labels)]] if calm_mask.sum() > 10 else np.array([])

    print(f'\n  INVARIANT B: Orbit distribution')
    orbit_stress = None
    orbit_calm = None
    if len(stress_labels) > 10 and len(calm_labels) > 10:
        orbit_stress = orbit_distribution(stress_labels, CLUSTER_MAP, MODULUS)
        orbit_calm = orbit_distribution(calm_labels, CLUSTER_MAP, MODULUS)
        print(f'    Stress:  cosmos={orbit_stress["cosmos"]:.3f}  satellite={orbit_stress["satellite"]:.3f}  singularity={orbit_stress["singularity"]:.3f}')
        print(f'    Calm:    cosmos={orbit_calm["cosmos"]:.3f}  satellite={orbit_calm["satellite"]:.3f}  singularity={orbit_calm["singularity"]:.3f}')
    else:
        print(f'    Insufficient stress/calm segments for orbit analysis')

    # ── INVARIANT C: Satellite fraction at stress ──
    sat_stress = orbit_stress['satellite'] if orbit_stress else None
    sat_calm = orbit_calm['satellite'] if orbit_calm else None
    print(f'\n  INVARIANT C: Satellite fraction')
    if sat_stress is not None and sat_calm is not None:
        print(f'    Stress satellite: {sat_stress:.3f}')
        print(f'    Calm satellite:   {sat_calm:.3f}')
        print(f'    Δ satellite:      {sat_stress - sat_calm:+.3f}')
        print(f'    Pattern: {"STRESS → MORE SATELLITE" if sat_stress > sat_calm else "STRESS → LESS SATELLITE"}')

    # ── INVARIANT D: QCI autocorrelation (persistence) ──
    if len(oos_qci) > 10:
        ac1 = np.corrcoef(oos_qci[:-1], oos_qci[1:])[0, 1]
        ac5 = np.corrcoef(oos_qci[:-5], oos_qci[5:])[0, 1] if len(oos_qci) > 10 else 0
    else:
        ac1, ac5 = 0, 0
    print(f'\n  INVARIANT D: QCI persistence')
    print(f'    AC(1) = {ac1:.4f}')
    print(f'    AC(5) = {ac5:.4f}')

    result = {
        'domain': name,
        'n_oos': len(oos_qci),
        'qci_stress_r': float(r),
        'qci_stress_p': float(p),
        'qci_sign': 'negative' if r < 0 else 'positive',
        'orbit_stress': orbit_stress,
        'orbit_calm': orbit_calm,
        'satellite_delta': float(sat_stress - sat_calm) if sat_stress is not None else None,
        'ac1': float(ac1),
        'ac5': float(ac5),
    }
    return result


# ══════════════════════════════════════════════════════
# DOMAIN DATA GENERATORS
# ══════════════════════════════════════════════════════

def generate_finance_data():
    """Generate synthetic multi-asset returns with vol clustering."""
    np.random.seed(RANDOM_SEED)
    T = 2500  # ~10 years daily
    D = 6     # 6 assets

    # Base returns with vol clustering (GARCH-like)
    vol = np.ones(T) * 0.01
    for t in range(1, T):
        vol[t] = 0.01 + 0.85 * (vol[t-1] - 0.01) + 0.1 * abs(np.random.randn()) * 0.01

    returns = np.zeros((T, D))
    correlation = 0.3
    for t in range(T):
        common = np.random.randn() * vol[t]
        for d in range(D):
            idio = np.random.randn() * vol[t] * (1 + 0.5 * d/D)
            returns[t, d] = correlation * common + (1-correlation) * idio

    # Stress indicator: periods where vol > 75th percentile
    rv = np.array([returns[max(0,t-21):t, 0].std() for t in range(1, T+1)])
    stress = (rv > np.percentile(rv, 75)).astype(float)

    return returns, stress, "Synthetic multi-asset returns with GARCH-like vol clustering"


def generate_eeg_data():
    """Generate synthetic multi-channel EEG with seizure events."""
    np.random.seed(RANDOM_SEED + 1)
    T = 3000  # 3000 epochs
    D = 8     # 8 channels

    # Background EEG: pink noise + alpha rhythm
    eeg = np.zeros((T, D))
    for d in range(D):
        # Pink noise
        white = np.random.randn(T)
        pink = np.cumsum(white) * 0.01
        pink -= pink.mean()
        # Alpha rhythm (8-12 Hz equivalent, but in epoch domain)
        alpha = 0.3 * np.sin(2 * np.pi * 10 * np.arange(T) / 256 + np.random.rand() * 2 * np.pi)
        eeg[:, d] = pink + alpha + np.random.randn(T) * 0.1

    # Inject seizures: 5 seizure epochs of 100 steps each
    stress = np.zeros(T)
    seizure_starts = [400, 900, 1500, 2000, 2600]
    for start in seizure_starts:
        end = min(start + 100, T)
        stress[start:end] = 1
        for d in range(D):
            # Seizure: high amplitude, synchronized, fast
            eeg[start:end, d] += 2.0 * np.sin(2*np.pi*25*np.arange(end-start)/256) * np.random.uniform(0.5, 1.5)
            eeg[start:end, d] += np.random.randn(end-start) * 0.5

    return eeg, stress, "Synthetic 8-channel EEG with 5 seizure events"


def generate_seismology_data():
    """Generate synthetic seismic catalog features."""
    np.random.seed(RANDOM_SEED + 2)
    T = 2000  # 2000 days
    D = 4     # daily count, mean mag, max mag, mean depth

    # Background seismicity with periodic clustering
    base_rate = 5.0  # events per day
    rates = base_rate + 3 * np.sin(2*np.pi*np.arange(T)/365) + np.random.randn(T) * 1.5
    rates = np.maximum(rates, 0.5)

    # Aftershock sequences (stress periods)
    stress = np.zeros(T)
    mainshock_days = [150, 500, 800, 1200, 1700]
    for day in mainshock_days:
        if day < T:
            # Omori-law decay
            duration = min(60, T - day)
            stress[day:day+duration] = 1
            for dt in range(duration):
                rates[day+dt] += 30.0 / (1 + dt)

    count = np.random.poisson(rates)
    mean_mag = 4.0 + 0.3 * np.log1p(count) + np.random.randn(T) * 0.2
    max_mag = mean_mag + np.random.exponential(0.5, T)
    mean_depth = 30 + 10 * np.random.randn(T)

    data = np.column_stack([count, mean_mag, max_mag, mean_depth])
    return data, stress, "Synthetic seismic catalog: count, mean/max magnitude, depth"


def generate_audio_data():
    """Generate synthetic audio features with signal transitions."""
    np.random.seed(RANDOM_SEED + 3)
    T = 2000
    D = 6  # spectral features

    # Background: white noise spectral features
    data = np.random.randn(T, D) * 0.5

    # Signal periods: structured harmonic content
    stress = np.zeros(T)  # here "stress" = structured signal present
    signal_periods = [(200, 400), (600, 800), (1000, 1300), (1500, 1650), (1800, 1950)]
    for start, end in signal_periods:
        stress[start:end] = 1
        # Inject harmonic structure
        for d in range(D):
            freq = (d + 1) * 2
            data[start:end, d] += 1.5 * np.sin(2*np.pi*freq*np.arange(end-start)/100)

    return data, stress, "Synthetic audio features: noise background + harmonic signal periods"


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('QA CROSS-DOMAIN INVARIANCE TEST')
    print('Pre-registered Tier 4 candidate experiment')
    print('=' * 70)
    print()
    print('FIXED PARAMETERS (identical across ALL domains):')
    print(f'  Modulus:       {MODULUS}')
    print(f'  K clusters:    {N_CLUSTERS}')
    print(f'  QCI window:    {QCI_WINDOW}')
    print(f'  Cluster map:   {CLUSTER_MAP}')
    print(f'  Random seed:   {RANDOM_SEED}')
    print(f'  Train/test:    first half / second half')
    print(f'  Standardize:   rolling z-score (window={QCI_WINDOW})')
    print()
    print('NO per-domain tuning. Same operator, same parameters, same pipeline.')

    # Generate all domain data
    domains = [
        ('Finance', *generate_finance_data()),
        ('EEG', *generate_eeg_data()),
        ('Seismology', *generate_seismology_data()),
        ('Audio', *generate_audio_data()),
    ]

    results = []
    for name, data, stress, desc in domains:
        r = run_domain(name, data, stress, desc)
        if r:
            results.append(r)

    # ══════════════════════════════════════════════════════
    # CROSS-DOMAIN COMPARISON
    # ══════════════════════════════════════════════════════
    print(f'\n\n{"="*70}')
    print('CROSS-DOMAIN INVARIANCE SCORECARD')
    print(f'{"="*70}\n')

    print(f'{"Domain":<14} {"QCI-stress r":<14} {"Sign":<10} {"Sat Δ":<10} {"Sat pattern":<24} {"AC(1)":<8}')
    print('─' * 80)

    signs = []
    sat_patterns = []
    for r in results:
        sign = r['qci_sign']
        signs.append(sign)
        sat_delta = r['satellite_delta']
        sat_pat = 'MORE satellite' if sat_delta and sat_delta > 0 else ('LESS satellite' if sat_delta else 'N/A')
        sat_patterns.append(sat_pat)
        print(f'{r["domain"]:<14} {r["qci_stress_r"]:>+.4f}       {sign:<10} '
              f'{sat_delta if sat_delta else "N/A":>8}  {sat_pat:<24} {r["ac1"]:.4f}')

    print()

    # ── INVARIANCE TESTS ──
    print('INVARIANCE TEST RESULTS:')
    print()

    # Test A: Same QCI sign across domains?
    unique_signs = set(signs)
    sign_invariant = len(unique_signs) == 1
    print(f'  A. QCI-stress sign consistency: {unique_signs}')
    print(f'     {"✓ INVARIANT — same sign across all domains" if sign_invariant else "✗ VARIANT — sign differs across domains"}')
    print()

    # Test B: Same satellite pattern?
    unique_sat = set(sat_patterns)
    sat_invariant = len(unique_sat) == 1 and 'N/A' not in unique_sat
    print(f'  B. Satellite pattern consistency: {unique_sat}')
    print(f'     {"✓ INVARIANT — same satellite shift across all domains" if sat_invariant else "✗ VARIANT — satellite pattern differs"}')
    print()

    # Test C: QCI persistence (all should show high AC)
    ac_values = [r['ac1'] for r in results]
    ac_invariant = all(ac > 0.5 for ac in ac_values)
    print(f'  C. QCI persistence (AC(1) > 0.5): {[f"{ac:.3f}" for ac in ac_values]}')
    print(f'     {"✓ INVARIANT — QCI persistent across all domains" if ac_invariant else "✗ VARIANT — QCI persistence differs"}')
    print()

    # Test D: Statistical significance across domains
    sig_count = sum(1 for r in results if r['qci_stress_p'] < 0.05)
    sig_invariant = sig_count == len(results)
    print(f'  D. Significance (p < 0.05): {sig_count}/{len(results)} domains')
    print(f'     {"✓ INVARIANT — significant across all domains" if sig_invariant else "✗ VARIANT — not all domains significant"}')
    print()

    # ── OVERALL VERDICT ──
    n_pass = sum([sign_invariant, sat_invariant, ac_invariant, sig_invariant])
    print(f'{"="*70}')
    print(f'OVERALL: {n_pass}/4 invariance tests passed')
    print()
    if n_pass == 4:
        print('VERDICT: All invariants hold. Tier 4 CANDIDATE — proceed to')
        print('adversarial testing (encoding perturbation, standard model comparison).')
    elif n_pass >= 2:
        print('VERDICT: Partial invariance. QA shows cross-domain structure but')
        print('not full invariance. Remains Tier 3 (useful predictive formalism).')
        print('Investigate which domains break and why.')
    else:
        print('VERDICT: Invariance fails. QA predictive signal is domain-specific,')
        print('not universal. Remains Tier 3 at best, possibly Tier 2.')
        print('The signal may reflect encoding choices, not physics.')
    print(f'{"="*70}')

    # Save results
    with open('qa_cross_domain_results.json', 'w') as f:
        json.dump({
            'parameters': {
                'modulus': MODULUS,
                'n_clusters': N_CLUSTERS,
                'qci_window': QCI_WINDOW,
                'cluster_map': {str(k): v for k, v in CLUSTER_MAP.items()},
                'random_seed': RANDOM_SEED,
            },
            'results': results,
            'invariance': {
                'sign': sign_invariant,
                'satellite': sat_invariant,
                'persistence': ac_invariant,
                'significance': sig_invariant,
                'total_pass': n_pass,
            }
        }, f, indent=2)
    print('\nSaved: qa_cross_domain_results.json')


if __name__ == '__main__':
    main()
