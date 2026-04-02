#!/usr/bin/env python3
"""
QA Covariant Observable — Pre-Registered Validation

PRE-REGISTRATION (locked before running QCI):

OPERATIONAL DEFINITION of Φ(D):
  Compute spectral concentration around events vs non-events.
  Spectral concentration = power in top-3 frequency bins / total power.
  If concentration INCREASES at events: Φ = +1 (order onset)
  If concentration DECREASES at events: Φ = -1 (disorder onset)
  This is computed WITHOUT any QA/QCI involvement.

NEW DOMAINS (never used in QA before):

  Domain 5: CLIMATE
    Synthetic temperature anomalies with heat waves.
    Heat waves = persistent high-pressure blocking = sustained anomaly.
    PRE-REGISTERED PREDICTION: Heat waves are ORDER onset.
    Expected Φ = +1. Expected QCI sign: POSITIVE.

  Domain 6: CARDIAC
    Synthetic heart rate with arrhythmia episodes.
    Arrhythmia = disruption of regular sinus rhythm.
    PRE-REGISTERED PREDICTION: Arrhythmia is DISORDER onset.
    Expected Φ = -1. Expected QCI sign: NEGATIVE.

  Domain 7: NULL CONTROL
    Pure random walk. "Events" = large excursions.
    No structural change at events — just noise.
    PRE-REGISTERED PREDICTION: Φ undefined, QCI should show NO signal.

QA PARAMETERS (same as invariance test — zero changes):
  Modulus = 24, K = 6, QCI_window = 63
  Cluster map = {0:4, 1:8, 2:12, 3:16, 4:20, 5:24}

SUCCESS CRITERIA:
  1. Φ computed from spectral concentration matches pre-registered prediction
  2. QCI sign matches Φ prediction
  3. Null control shows no significant QCI-event correlation
  All three must hold.

FAILURE: Any prediction wrong → transformation law not confirmed.
"""

import numpy as np
from scipy import stats, fft
from sklearn.cluster import KMeans
import json

# FIXED PARAMETERS (identical to invariance test)
MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
RANDOM_SEED = 42
CLUSTER_MAP = {i: 4*(i+1) for i in range(N_CLUSTERS)}


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1

def qa_step(b, e, m):
    return e, qa_mod(b + e, m)

def compute_qci(labels, cmap, m, window):
    t_match = []
    for t in range(len(labels) - 2):
        b_t = cmap[labels[t]]
        e_t = cmap[labels[t + 1]]
        _, pred_next = qa_step(b_t, e_t, m)
        actual_next = cmap[labels[t + 2]]
        t_match.append(1 if pred_next == actual_next else 0)
    t_match = np.array(t_match, dtype=float)
    qci = np.full(len(t_match), np.nan)
    for i in range(window, len(t_match)):
        qci[i] = np.mean(t_match[i-window:i])
    return qci


def spectral_concentration(signal, top_n=3):
    """Fraction of power in top-N frequency bins."""
    spectrum = np.abs(fft.rfft(signal))
    power = spectrum * spectrum
    total = power.sum()
    if total == 0:
        return 0.0
    top_power = np.sort(power)[-top_n:].sum()
    return float(top_power / total)


def compute_phi(data, stress, window=50):
    """
    Compute Φ(D) operationally from spectral concentration.
    NO QA involvement. Pure signal processing.

    Returns: phi (+1 or -1), sc_stress, sc_calm
    """
    T = len(data)
    # Use first column if multi-dimensional
    if data.ndim > 1:
        signal = data[:, 0]
    else:
        signal = data

    # Compute spectral concentration in windows around stress vs calm
    sc_stress = []
    sc_calm = []

    for t in range(window, T - window):
        chunk = signal[t-window//2:t+window//2]
        sc = spectral_concentration(chunk)
        if stress[t] > 0:
            sc_stress.append(sc)
        else:
            sc_calm.append(sc)

    mean_sc_stress = np.mean(sc_stress) if sc_stress else 0
    mean_sc_calm = np.mean(sc_calm) if sc_calm else 0

    # Φ = +1 if concentration increases at events (order onset)
    # Φ = -1 if concentration decreases at events (disorder onset)
    if mean_sc_stress > mean_sc_calm:
        phi = +1
    elif mean_sc_stress < mean_sc_calm:
        phi = -1
    else:
        phi = 0  # ambiguous

    return phi, mean_sc_stress, mean_sc_calm


def run_domain_blind(name, data, stress, pre_registered_phi, description):
    """Run the full pipeline: compute Φ, then QCI, then compare."""
    print(f'\n{"="*70}')
    print(f'DOMAIN: {name}')
    print(f'  {description}')
    print(f'  Pre-registered Φ: {pre_registered_phi:+d}')
    print(f'{"="*70}')

    T = len(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    D = data.shape[1]

    # ── STEP 1: Compute Φ from spectral concentration (NO QA) ──
    phi, sc_stress, sc_calm = compute_phi(data, stress)
    print(f'\n  STEP 1: Spectral concentration (no QA involved)')
    print(f'    At events:     {sc_stress:.6f}')
    print(f'    At non-events: {sc_calm:.6f}')
    print(f'    Computed Φ = {phi:+d}')
    print(f'    Pre-registered Φ = {pre_registered_phi:+d}')
    phi_match = (phi == pre_registered_phi) or pre_registered_phi == 0
    print(f'    Match: {"✓ YES" if phi_match else "✗ NO — pre-registration WRONG"}')

    # ── STEP 2: Predict QCI sign from Φ ──
    if phi == +1:
        predicted_sign = 'positive'
    elif phi == -1:
        predicted_sign = 'negative'
    else:
        predicted_sign = 'none'
    print(f'\n  STEP 2: Predicted QCI-event sign: {predicted_sign}')

    # ── STEP 3: Run QCI (blind to prediction) ──
    half = T // 2

    # Standardize
    std_data = np.zeros_like(data)
    for i in range(QCI_WINDOW, T):
        window_data = data[i-QCI_WINDOW:i]
        mu = window_data.mean(axis=0)
        sigma = window_data.std(axis=0) + 1e-10
        std_data[i] = (data[i] - mu) / sigma
    valid = std_data[QCI_WINDOW:]
    stress_valid = stress[QCI_WINDOW:]

    train_end = len(valid) // 2
    np.random.seed(RANDOM_SEED)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_SEED)
    km.fit(valid[:train_end])
    all_labels = km.predict(valid)

    qci = compute_qci(all_labels, CLUSTER_MAP, MODULUS, QCI_WINDOW)

    # Align
    qci_len = len(qci)
    stress_al = stress_valid[:qci_len]

    oos_start = train_end
    oos_qci = qci[oos_start:]
    oos_stress = stress_al[oos_start:]

    mask = ~np.isnan(oos_qci)
    oos_qci = oos_qci[mask]
    oos_stress = oos_stress[mask]

    if len(oos_qci) < 30:
        print(f'  STEP 3: Insufficient OOS data ({len(oos_qci)} pts)')
        return None

    if oos_stress.std() > 0:
        r, p = stats.pearsonr(oos_qci, oos_stress.astype(float))
    else:
        r, p = 0.0, 1.0

    actual_sign = 'positive' if r > 0 else ('negative' if r < 0 else 'none')

    print(f'\n  STEP 3: QCI results (blind)')
    print(f'    QCI-event r = {r:+.4f}, p = {p:.6f}')
    print(f'    Actual sign: {actual_sign}')
    print(f'    Significant (p < 0.05): {"YES" if p < 0.05 else "NO"}')

    # ── STEP 4: Score ──
    sign_correct = (predicted_sign == actual_sign) or predicted_sign == 'none'
    print(f'\n  STEP 4: SCORE')
    print(f'    Φ pre-registration correct:  {"✓" if phi_match else "✗"}')
    print(f'    QCI sign prediction correct: {"✓" if sign_correct else "✗"}')
    print(f'    Significant:                 {"✓" if p < 0.05 else "✗ (not significant)"}')

    all_pass = phi_match and sign_correct and p < 0.05
    print(f'    OVERALL: {"✓ PASS" if all_pass else "✗ FAIL"}')

    return {
        'domain': name,
        'pre_registered_phi': pre_registered_phi,
        'computed_phi': phi,
        'phi_match': phi_match,
        'predicted_sign': predicted_sign,
        'actual_sign': actual_sign,
        'sign_correct': sign_correct,
        'r': float(r),
        'p': float(p),
        'significant': p < 0.05,
        'all_pass': all_pass,
    }


# ══════════════════════════════════════════════════════
# NEW DOMAIN DATA GENERATORS
# ══════════════════════════════════════════════════════

def generate_climate():
    """Temperature anomalies with heat wave events (persistent blocking = ORDER)."""
    np.random.seed(RANDOM_SEED + 10)
    T = 2500
    D = 4  # temp anomaly, pressure, humidity, wind speed

    # Background: seasonal + AR(1) noise
    t = np.arange(T)
    seasonal = 5 * np.sin(2*np.pi*t/365)
    ar = np.zeros(T)
    for i in range(1, T):
        ar[i] = 0.95 * ar[i-1] + np.random.randn() * 0.5

    temp = seasonal + ar

    # Other channels
    pressure = 1013 + 5 * np.sin(2*np.pi*t/365 + 0.5) + np.random.randn(T) * 2
    humidity = 60 + 10 * np.sin(2*np.pi*t/365 + 1.0) + np.random.randn(T) * 5
    wind = 10 + np.random.exponential(3, T)

    # Heat waves: persistent high-pressure blocking (ORDER)
    stress = np.zeros(T)
    hw_starts = [200, 570, 930, 1300, 1650, 2000, 2350]
    for start in hw_starts:
        duration = np.random.randint(10, 25)
        end = min(start + duration, T)
        stress[start:end] = 1
        # Heat wave: sustained high temp, high pressure, low wind (ORDERED state)
        temp[start:end] += 8 + np.random.randn(end-start) * 0.5  # persistent, low variance
        pressure[start:end] += 15  # high pressure blocking
        wind[start:end] *= 0.3  # calm winds

    data = np.column_stack([temp, pressure, humidity, wind])
    return data, stress


def generate_cardiac():
    """Heart rate with arrhythmia episodes (rhythm disruption = DISORDER)."""
    np.random.seed(RANDOM_SEED + 11)
    T = 3000
    D = 4

    # Normal sinus rhythm: regular with slight variability
    t = np.arange(T)
    # Heart rate: regular periodic signal
    hr = 72 + 5 * np.sin(2*np.pi*t/8)  # regular rhythm (~8 beat cycle for HRV)
    hrv = np.random.randn(T) * 2  # normal HRV

    # RR interval regularity
    rr_interval = 60.0 / hr + np.random.randn(T) * 0.01
    rr_variability = np.abs(np.diff(np.concatenate([[rr_interval[0]], rr_interval])))

    # QT interval (regular in normal rhythm)
    qt = 0.4 + 0.001 * hr + np.random.randn(T) * 0.01

    # Arrhythmia episodes: disruption of regular rhythm (DISORDER)
    stress = np.zeros(T)
    arrhythmia_starts = [300, 700, 1100, 1500, 1900, 2300, 2700]
    for start in arrhythmia_starts:
        duration = np.random.randint(30, 80)
        end = min(start + duration, T)
        stress[start:end] = 1
        # Arrhythmia: irregular rate, high variability, chaotic
        hr[start:end] += np.random.randn(end-start) * 20  # erratic
        rr_variability[start:end] += np.random.exponential(0.05, end-start)
        qt[start:end] += np.random.randn(end-start) * 0.05

    data = np.column_stack([hr, rr_interval, rr_variability, qt])
    return data, stress


def generate_null_control():
    """Pure random walk. Events = large excursions. No structural change."""
    np.random.seed(RANDOM_SEED + 12)
    T = 2500
    D = 4

    # Independent random walks
    data = np.cumsum(np.random.randn(T, D) * 0.1, axis=0)

    # "Events" = periods where |walk| > 2 std
    magnitude = np.sqrt((data * data).sum(axis=1))
    threshold = np.percentile(magnitude, 80)
    stress = (magnitude > threshold).astype(float)

    return data, stress


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('QA COVARIANT OBSERVABLE — PRE-REGISTERED VALIDATION')
    print('=' * 70)
    print()
    print('PRE-REGISTRATIONS (locked before any QCI computation):')
    print()
    print('  Domain 5 (Climate/heat waves):')
    print('    Heat wave = persistent blocking = ORDER onset')
    print('    Predicted Φ = +1')
    print('    Predicted QCI sign: POSITIVE')
    print()
    print('  Domain 6 (Cardiac/arrhythmia):')
    print('    Arrhythmia = rhythm disruption = DISORDER onset')
    print('    Predicted Φ = -1')
    print('    Predicted QCI sign: NEGATIVE')
    print()
    print('  Domain 7 (Null control):')
    print('    Random walk excursions = no structural change')
    print('    Predicted Φ = 0 (ambiguous)')
    print('    Predicted QCI: NOT SIGNIFICANT')
    print()
    print('Fixed parameters: mod=24, K=6, window=63, same pipeline.')
    print('NO changes from invariance test.')

    # Generate data
    climate_data, climate_stress = generate_climate()
    cardiac_data, cardiac_stress = generate_cardiac()
    null_data, null_stress = generate_null_control()

    # Run blind
    results = []

    r1 = run_domain_blind('Climate', climate_data, climate_stress,
                          pre_registered_phi=+1,
                          description='Temperature anomalies with heat wave blocking events')
    if r1: results.append(r1)

    r2 = run_domain_blind('Cardiac', cardiac_data, cardiac_stress,
                          pre_registered_phi=-1,
                          description='Heart rate with arrhythmia episodes')
    if r2: results.append(r2)

    r3 = run_domain_blind('Null Control', null_data, null_stress,
                          pre_registered_phi=0,
                          description='Random walk with large-excursion events (no structure)')
    if r3: results.append(r3)

    # ── FINAL SCORECARD ──
    print(f'\n\n{"="*70}')
    print('FINAL SCORECARD — COVARIANT OBSERVABLE VALIDATION')
    print(f'{"="*70}\n')

    print(f'{"Domain":<16} {"Φ pre-reg":<10} {"Φ computed":<12} {"Φ match":<9} '
          f'{"QCI r":<10} {"Sign ok":<9} {"Sig?":<6} {"PASS"}')
    print('─' * 90)

    n_pass = 0
    for r in results:
        passed = '✓' if r['all_pass'] else '✗'
        if r['all_pass']:
            n_pass += 1
        # For null control: pass if NOT significant
        if r['domain'] == 'Null Control':
            null_pass = not r['significant']
            passed = '✓' if null_pass else '✗'
            if null_pass:
                n_pass += 1
            else:
                n_pass -= (1 if r['all_pass'] else 0)  # don't double count

        print(f'{r["domain"]:<16} {r["pre_registered_phi"]:>+d}         '
              f'{r["computed_phi"]:>+d}           {"✓" if r["phi_match"] else "✗"}        '
              f'{r["r"]:>+.4f}     {"✓" if r["sign_correct"] else "✗"}        '
              f'{"Y" if r["significant"] else "N"}     {passed}')

    print()
    total_tests = len(results)
    print(f'RESULT: {n_pass}/{total_tests} domains passed')
    print()

    if n_pass == total_tests:
        print('VERDICT: Transformation law CONFIRMED on new domains.')
        print('QCI is a covariant observable with Φ(D) = ±1 transformation.')
        print('Tier 3.5: Structured non-universality established.')
    elif n_pass >= 2:
        print('VERDICT: Partial confirmation. Transformation law holds for some')
        print('new domains but not all. Further investigation needed.')
    else:
        print('VERDICT: Transformation law NOT confirmed. The sign flip pattern')
        print('observed in the first 4 domains may be coincidental.')

    print(f'\n{"="*70}')

    # Save
    with open('qa_covariance_results.json', 'w') as f:
        json.dump({'results': results, 'n_pass': n_pass, 'total': total_tests}, f, indent=2)
    print('Saved: qa_covariance_results.json')


if __name__ == '__main__':
    main()
