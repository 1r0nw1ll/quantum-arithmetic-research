#!/usr/bin/env python3
"""
QCI v2 — Null-Calibrated QA Coherence Index

The v1 pipeline (rolling z-score → k-means → T-matching → rolling QCI)
produces spurious signal on random walks due to smoothing/discretization
artifacts. This was identified on 2026-04-01 via null control failure.

QCI v2 fixes this by:
1. Computing QCI on the real data (same pipeline as v1)
2. Computing QCI on matched surrogate processes (same pipeline)
3. Defining QCI* = QCI_real - E[QCI_surrogates]
4. Testing: is QCI* significantly different from zero?

Surrogate types (process-level nulls):
  A. IID noise (destroys all structure)
  B. AR(1) fitted to data (preserves autocorrelation)
  C. Phase-randomized (preserves spectrum, destroys phase coupling)
  D. Block-shuffled (preserves local clustering, destroys long-range)

Also decomposes artifact contribution by pipeline stage:
  Stage 1: rolling z-score only
  Stage 2: + k-means clustering
  Stage 3: + T-operator matching
  Stage 4: + rolling QCI

Acceptance criterion:
  QCI* must be significant (p < 0.05) after calibrating against
  ALL four surrogate types. If not, the signal is pipeline artifact.
"""

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import json

# FIXED PARAMETERS (unchanged from v1)
MODULUS = 24
N_CLUSTERS = 6
QCI_WINDOW = 63
RANDOM_SEED = 42
CLUSTER_MAP = {i: 4*(i+1) for i in range(N_CLUSTERS)}
N_SURROGATES = 100  # number of surrogate realizations per type


def qa_mod(x, m):
    return ((int(x) - 1) % m) + 1

def qa_step(b, e, m):
    return e, qa_mod(b + e, m)


def compute_qci_from_data(data, seed=42):
    """Full pipeline: standardize → cluster → T-match → rolling QCI.
    Returns QCI array and intermediate diagnostics."""
    T = data.shape[0]
    D = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Stage 1: Rolling z-score
    std_data = np.zeros_like(data)
    for i in range(QCI_WINDOW, T):
        window_data = data[i-QCI_WINDOW:i]
        mu = window_data.mean(axis=0)
        sigma = window_data.std(axis=0) + 1e-10
        std_data[i] = (data[i] - mu) / sigma
    valid = std_data[QCI_WINDOW:]

    if len(valid) < QCI_WINDOW * 3:
        return None, {}

    # Stage 2: k-means
    half = len(valid) // 2
    np.random.seed(seed)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=seed)
    km.fit(valid[:half])
    labels = km.predict(valid)

    # Stage 3: T-operator matching
    t_match = []
    for t in range(len(labels) - 2):
        b_t = CLUSTER_MAP[labels[t]]
        e_t = CLUSTER_MAP[labels[t + 1]]
        _, pred = qa_step(b_t, e_t, MODULUS)
        actual = CLUSTER_MAP[labels[t + 2]]
        t_match.append(1 if pred == actual else 0)
    t_match = np.array(t_match, dtype=float)

    # Stage 4: Rolling QCI
    qci = np.full(len(t_match), np.nan)
    for i in range(QCI_WINDOW, len(t_match)):
        qci[i] = np.mean(t_match[i-QCI_WINDOW:i])

    # OOS portion
    oos_start = half
    oos_qci = qci[oos_start:]

    diagnostics = {
        'label_entropy': stats.entropy(np.bincount(labels, minlength=N_CLUSTERS) + 1),
        'mean_t_match': float(np.mean(t_match)),
        'oos_qci_mean': float(np.nanmean(oos_qci)),
        'oos_qci_std': float(np.nanstd(oos_qci)),
    }

    return oos_qci, diagnostics


# ══════════════════════════════════════════════════════
# SURROGATE GENERATORS
# ══════════════════════════════════════════════════════

def surrogate_iid(data):
    """IID Gaussian with same mean and variance per column."""
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    return np.random.randn(*data.shape) * sigma + mu


def surrogate_ar1(data):
    """AR(1) process fitted to each column."""
    T, D = data.shape
    out = np.zeros_like(data)
    for d in range(D):
        col = data[:, d]
        # Fit AR(1): x_t = phi * x_{t-1} + eps
        phi = np.corrcoef(col[:-1], col[1:])[0, 1]
        sigma_eps = np.std(col) * np.sqrt(1 - phi * phi) if abs(phi) < 1 else np.std(col) * 0.1
        out[0, d] = col[0]
        for t in range(1, T):
            out[t, d] = phi * out[t-1, d] + np.random.randn() * sigma_eps
    return out


def surrogate_phase_randomized(data):
    """Phase-randomized surrogate: preserves power spectrum, destroys phase coupling."""
    T, D = data.shape
    out = np.zeros_like(data)
    for d in range(D):
        col = data[:, d]
        ft = np.fft.rfft(col)
        # Randomize phases but keep amplitudes
        phases = np.random.uniform(0, 2*np.pi, len(ft))
        phases[0] = 0  # keep DC component
        if T % 2 == 0:
            phases[-1] = 0  # keep Nyquist
        ft_rand = np.abs(ft) * np.exp(1j * phases)
        out[:, d] = np.fft.irfft(ft_rand, n=T)
    return out


def surrogate_block_shuffled(data, block_size=50):
    """Block-shuffled: preserves local structure, destroys long-range order."""
    T, D = data.shape
    n_blocks = T // block_size
    indices = np.arange(n_blocks)
    np.random.shuffle(indices)
    out = np.zeros_like(data)
    for i, idx in enumerate(indices):
        src_start = idx * block_size
        dst_start = i * block_size
        length = min(block_size, T - dst_start, T - src_start)
        out[dst_start:dst_start+length] = data[src_start:src_start+length]
    # Handle remainder
    remainder = T - n_blocks * block_size
    if remainder > 0:
        out[-remainder:] = data[-remainder:]
    return out


# ══════════════════════════════════════════════════════
# QCI v2: NULL-CALIBRATED
# ══════════════════════════════════════════════════════

def qci_v2(data, stress, domain_name, n_surrogates=N_SURROGATES):
    """
    Compute null-calibrated QCI*.

    Returns QCI*, surrogate distributions, and verdict.
    """
    print(f'\n{"="*70}')
    print(f'QCI v2: {domain_name}')
    print(f'  Data shape: {data.shape}')
    print(f'  Stress events: {stress.sum():.0f}/{len(stress)}')
    print(f'  Surrogates per type: {n_surrogates}')
    print(f'{"="*70}')

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # ── Real QCI ──
    real_qci, real_diag = compute_qci_from_data(data, seed=RANDOM_SEED)
    if real_qci is None:
        print('  ERROR: insufficient data')
        return None

    # Align stress to QCI
    stress_trimmed = stress[QCI_WINDOW:]
    qci_len = len(real_qci)
    stress_al = stress_trimmed[:qci_len]

    # Trim NaNs
    mask = ~np.isnan(real_qci)
    real_qci_clean = real_qci[mask]
    stress_clean = stress_al[mask]

    if len(real_qci_clean) < 50 or stress_clean.std() == 0:
        print('  ERROR: insufficient valid OOS data or no stress variance')
        return None

    real_r, real_p = stats.pearsonr(real_qci_clean, stress_clean.astype(float))
    print(f'\n  Real QCI:')
    print(f'    mean={real_diag["oos_qci_mean"]:.4f}, std={real_diag["oos_qci_std"]:.4f}')
    print(f'    QCI-stress r = {real_r:+.4f}, p = {real_p:.6f}')

    # ── Surrogate QCIs ──
    surrogate_types = {
        'IID': surrogate_iid,
        'AR(1)': surrogate_ar1,
        'Phase-randomized': surrogate_phase_randomized,
        'Block-shuffled': surrogate_block_shuffled,
    }

    surrogate_results = {}
    for stype, generator in surrogate_types.items():
        surr_rs = []
        for i in range(n_surrogates):
            np.random.seed(RANDOM_SEED + 1000 * hash(stype) % 10000 + i)
            surr_data = generator(data)
            surr_qci, _ = compute_qci_from_data(surr_data, seed=RANDOM_SEED + i)
            if surr_qci is None:
                continue
            surr_qci_clean = surr_qci[mask[:len(surr_qci)]]
            surr_stress = stress_clean[:len(surr_qci_clean)]
            if len(surr_qci_clean) < 30 or surr_stress.std() == 0:
                continue
            sr, _ = stats.pearsonr(surr_qci_clean, surr_stress.astype(float))
            surr_rs.append(sr)

        if surr_rs:
            surr_rs = np.array(surr_rs)
            mean_r = surr_rs.mean()
            std_r = surr_rs.std()
            # Calibrated: how many SDs is real_r from surrogate mean?
            if std_r > 0:
                z_score = (real_r - mean_r) / std_r
                # Two-sided p-value from surrogate distribution
                p_surr = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_surr = 1.0
            # Also: rank-based p-value
            rank_p = np.mean(np.abs(surr_rs) >= np.abs(real_r))

            surrogate_results[stype] = {
                'mean_r': float(mean_r),
                'std_r': float(std_r),
                'z_score': float(z_score),
                'p_parametric': float(p_surr),
                'p_rank': float(rank_p),
                'n_valid': len(surr_rs),
            }

            print(f'\n  {stype} surrogates ({len(surr_rs)} valid):')
            print(f'    E[r_null] = {mean_r:+.4f} ± {std_r:.4f}')
            print(f'    QCI* z-score = {z_score:+.2f}')
            print(f'    p (parametric) = {p_surr:.4f}')
            print(f'    p (rank) = {rank_p:.4f}')
            beats = 'YES' if p_surr < 0.05 and rank_p < 0.10 else 'NO'
            print(f'    Beats null? {beats}')

    # ── VERDICT ──
    print(f'\n  {"─"*50}')
    print(f'  VERDICT for {domain_name}:')
    print(f'    Real r = {real_r:+.4f}')

    all_beaten = True
    for stype, sr in surrogate_results.items():
        beaten = sr['p_parametric'] < 0.05
        if not beaten:
            all_beaten = False
        status = '✓ beaten' if beaten else '✗ NOT beaten'
        print(f'    vs {stype}: z={sr["z_score"]:+.2f}, p={sr["p_parametric"]:.4f} → {status}')

    if all_beaten:
        print(f'\n  ✓ QCI* SIGNIFICANT: Signal survives ALL process-level nulls.')
        print(f'    This is GENUINE structure beyond pipeline artifact.')
    else:
        print(f'\n  ✗ QCI* NOT SIGNIFICANT against all nulls.')
        print(f'    Signal may be partially or fully a pipeline artifact.')

    return {
        'domain': domain_name,
        'real_r': float(real_r),
        'real_p': float(real_p),
        'surrogates': surrogate_results,
        'all_beaten': all_beaten,
    }


# ══════════════════════════════════════════════════════
# PIPELINE STAGE DECOMPOSITION
# ══════════════════════════════════════════════════════

def decompose_artifact(data, stress, domain_name):
    """Measure artifact contribution from each pipeline stage."""
    print(f'\n  ARTIFACT DECOMPOSITION for {domain_name}:')

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T = data.shape[0]

    # Baseline: raw correlation of first column with stress
    col = data[:, 0]
    if stress.std() > 0:
        r_raw, _ = stats.pearsonr(col, stress.astype(float))
    else:
        r_raw = 0
    print(f'    Stage 0 (raw signal):        r = {r_raw:+.4f}')

    # Stage 1: after rolling z-score
    std_data = np.zeros_like(data)
    for i in range(QCI_WINDOW, T):
        w = data[i-QCI_WINDOW:i]
        std_data[i] = (data[i] - w.mean(0)) / (w.std(0) + 1e-10)
    valid = std_data[QCI_WINDOW:]
    stress_v = stress[QCI_WINDOW:]
    if stress_v.std() > 0:
        r_zscore, _ = stats.pearsonr(valid[:, 0], stress_v.astype(float))
    else:
        r_zscore = 0
    print(f'    Stage 1 (rolling z-score):    r = {r_zscore:+.4f}')

    # Stage 2: after k-means (cluster label entropy with stress)
    half = len(valid) // 2
    np.random.seed(RANDOM_SEED)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_SEED)
    km.fit(valid[:half])
    labels = km.predict(valid)
    if stress_v.std() > 0:
        r_cluster, _ = stats.pearsonr(labels.astype(float), stress_v.astype(float))
    else:
        r_cluster = 0
    print(f'    Stage 2 (+ k-means label):   r = {r_cluster:+.4f}')

    # Stage 3: T-match rate (before rolling)
    t_match = []
    for t in range(len(labels) - 2):
        b_t = CLUSTER_MAP[labels[t]]
        e_t = CLUSTER_MAP[labels[t + 1]]
        _, pred = qa_step(b_t, e_t, MODULUS)
        actual = CLUSTER_MAP[labels[t + 2]]
        t_match.append(1 if pred == actual else 0)
    t_match = np.array(t_match, dtype=float)
    stress_tm = stress_v[:len(t_match)]
    if stress_tm.std() > 0:
        r_tmatch, _ = stats.pearsonr(t_match, stress_tm.astype(float))
    else:
        r_tmatch = 0
    print(f'    Stage 3 (+ T-match raw):     r = {r_tmatch:+.4f}')

    # Stage 4: rolling QCI
    qci = np.full(len(t_match), np.nan)
    for i in range(QCI_WINDOW, len(t_match)):
        qci[i] = np.mean(t_match[i-QCI_WINDOW:i])
    stress_qci = stress_v[:len(qci)]
    mask = ~np.isnan(qci)
    if mask.sum() > 30 and stress_qci[mask].std() > 0:
        r_qci, _ = stats.pearsonr(qci[mask], stress_qci[mask].astype(float))
    else:
        r_qci = 0
    print(f'    Stage 4 (+ rolling QCI):     r = {r_qci:+.4f}')
    print(f'    Artifact amplification: Stage 0→4 = {abs(r_qci)/max(abs(r_raw),0.001):.1f}x')


# ══════════════════════════════════════════════════════
# DATA GENERATORS (same as invariance test)
# ══════════════════════════════════════════════════════

def generate_finance():
    np.random.seed(RANDOM_SEED)
    T, D = 2500, 6
    vol = np.ones(T) * 0.01
    for t in range(1, T):
        vol[t] = 0.01 + 0.85 * (vol[t-1] - 0.01) + 0.1 * abs(np.random.randn()) * 0.01
    returns = np.zeros((T, D))
    for t in range(T):
        common = np.random.randn() * vol[t]
        for d in range(D):
            returns[t, d] = 0.3 * common + 0.7 * np.random.randn() * vol[t] * (1 + 0.5*d/D)
    rv = np.array([returns[max(0,t-21):t, 0].std() for t in range(1, T+1)])
    stress = (rv > np.percentile(rv, 75)).astype(float)
    return returns, stress, "Synthetic finance (GARCH-like)"

def generate_eeg():
    np.random.seed(RANDOM_SEED + 1)
    T, D = 3000, 8
    eeg = np.zeros((T, D))
    for d in range(D):
        pink = np.cumsum(np.random.randn(T)) * 0.01
        pink -= pink.mean()
        eeg[:, d] = pink + 0.3*np.sin(2*np.pi*10*np.arange(T)/256 + np.random.rand()*2*np.pi) + np.random.randn(T)*0.1
    stress = np.zeros(T)
    for start in [400, 900, 1500, 2000, 2600]:
        end = min(start + 100, T)
        stress[start:end] = 1
        for d in range(D):
            eeg[start:end, d] += 2.0*np.sin(2*np.pi*25*np.arange(end-start)/256)*np.random.uniform(0.5,1.5) + np.random.randn(end-start)*0.5
    return eeg, stress, "Synthetic EEG (5 seizures)"

def generate_null():
    np.random.seed(RANDOM_SEED + 12)
    T, D = 2500, 4
    data = np.cumsum(np.random.randn(T, D) * 0.1, axis=0)
    mag = np.sqrt((data*data).sum(axis=1))
    stress = (mag > np.percentile(mag, 80)).astype(float)
    return data, stress, "Null control (random walk)"


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('QCI v2 — NULL-CALIBRATED QA COHERENCE INDEX')
    print('=' * 70)
    print()
    print(f'Parameters: mod={MODULUS}, K={N_CLUSTERS}, window={QCI_WINDOW}')
    print(f'Surrogates: {N_SURROGATES} per type × 4 types = {N_SURROGATES*4} total per domain')
    print(f'Surrogate types: IID, AR(1), Phase-randomized, Block-shuffled')
    print()

    domains = [
        ('Finance', *generate_finance()),
        ('EEG', *generate_eeg()),
        ('Null Control', *generate_null()),
    ]

    all_results = []
    for name, data, stress, desc in domains:
        print(f'\n{"#"*70}')
        print(f'# {name}: {desc}')
        print(f'{"#"*70}')

        # Artifact decomposition first
        decompose_artifact(data, stress, name)

        # QCI v2
        result = qci_v2(data, stress, name, n_surrogates=N_SURROGATES)
        if result:
            all_results.append(result)

    # ── FINAL SUMMARY ──
    print(f'\n\n{"="*70}')
    print('QCI v2 FINAL SUMMARY')
    print(f'{"="*70}\n')

    print(f'{"Domain":<16} {"Real r":<10} {"vs IID":<12} {"vs AR(1)":<12} {"vs Phase":<12} {"vs Block":<12} {"ALL BEATEN?"}')
    print('─' * 90)

    for r in all_results:
        row = f'{r["domain"]:<16} {r["real_r"]:>+.4f}     '
        for stype in ['IID', 'AR(1)', 'Phase-randomized', 'Block-shuffled']:
            if stype in r['surrogates']:
                s = r['surrogates'][stype]
                beaten = s['p_parametric'] < 0.05
                row += f'{"✓" if beaten else "✗"} p={s["p_parametric"]:.3f}   '
            else:
                row += 'N/A          '
        row += '✓ YES' if r['all_beaten'] else '✗ NO'
        print(row)

    print()
    genuine = [r for r in all_results if r['all_beaten'] and r['domain'] != 'Null Control']
    null_clean = any(not r['all_beaten'] for r in all_results if r['domain'] == 'Null Control')

    if genuine and null_clean:
        print('VERDICT: QCI v2 shows GENUINE signal in real domains')
        print('while correctly rejecting null control. CLEAN TIER 3.')
    elif genuine and not null_clean:
        print('VERDICT: Signal in real domains BUT null control also passes.')
        print('Pipeline artifact not fully eliminated.')
    elif not genuine:
        print('VERDICT: No domain survives all process-level nulls.')
        print('QCI signal is fully explained by pipeline artifacts.')
    print(f'{"="*70}')

    with open('qci_v2_results.json', 'w') as f:
        # Convert bools for JSON
        def convert(obj):
            if isinstance(obj, (np.bool_, bool)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj
        json.dump({r['domain']: {k: convert(v) if not isinstance(v, dict) else
                   {k2: convert(v2) for k2, v2 in v.items()} if not isinstance(v, dict) else v
                   for k, v in r.items()} for r in all_results},
                  f, indent=2, default=str)
    print('\nSaved: qci_v2_results.json')


if __name__ == '__main__':
    main()
