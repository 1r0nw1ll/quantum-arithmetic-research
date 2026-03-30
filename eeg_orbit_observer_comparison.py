#!/usr/bin/env python3
"""
eeg_orbit_observer_comparison.py — Three-Observer QA Specificity Study

Compares three observer designs feeding the same QA layer + nested model test.
Tests whether observer architecture determines QA independence from delta.

Observer 1 (threshold-fallback): current design — gamma/alpha/theta thresholds,
  D_baseline as catch-all fallback. Delta-seizure → D_baseline → singularity.
  KNOWN ISSUE: QA inherits delta proxy (ΔR²=0.002, ns).

Observer 2 (dominant-band): whichever spectral band has peak power wins.
  D_baseline assigned explicitly when delta wins. Architecture is symmetric.
  Fixes the fallback bias but delta → D_baseline path still exists.

Observer 3 (topographic k-means): read all 23 channels, compute per-window
  broadband topographic amplitude vector, k-means into 4 clusters (unsupervised,
  no spectral band features, no seizure labels used). Truly feature-independent.
  Tests whether spatial brain organisation is more QA-informative than spectral.

All three use identical QA layer (MICROSTATE_STATES unchanged) and identical
nested logistic regression test (seizure ~ delta | seizure ~ delta + QA).
"""

QA_COMPLIANCE = "empirical_observer — EEG signal is observer input; QA discrete orbit is the classifier state"


import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import expit
from sklearn.cluster import KMeans

from eeg_orbit_classifier import (
    load_chbmit_dataset, DEFAULT_DATA_DIR,
    compute_orbit_sequence, orbit_statistics,
    classify_window_eeg, MICROSTATE_STATES, MODULUS,
    _read_edf_channel, CHBMIT_ANNOTATIONS, WINDOW_SECONDS
)
from eeg_autocorrelation_baseline import delta_power_ratio
from qa_orbit_rules import orbit_family


# ── Multi-channel EDF reader ───────────────────────────────────────────────────

def _read_edf_all_channels(edf_path: Path) -> tuple[np.ndarray, int, list[str]]:
    """
    Observer: read ALL channels from an EDF file.
    Returns (signals, sample_rate, channel_labels).
    signals shape: (n_channels, n_samples)
    """
    with open(edf_path, "rb") as fh:
        header = fh.read(256)
        ns = int(header[252:256].decode("ascii").strip())
        sig_header_raw = fh.read(ns * 256)

        def _field(offset, width, count):
            return [sig_header_raw[offset + i * width: offset + (i + 1) * width
                                   ].decode("ascii").strip()
                    for i in range(count)]

        labels    = _field(0,                              16, ns)
        phys_mins = [float(x) for x in _field(ns * (16 + 80 + 8),             8, ns)]
        phys_maxs = [float(x) for x in _field(ns * (16 + 80 + 8 + 8),         8, ns)]
        dig_mins  = [int(x)   for x in _field(ns * (16 + 80 + 8 + 8 + 8),     8, ns)]
        dig_maxs  = [int(x)   for x in _field(ns * (16 + 80 + 8 + 8 + 8 + 8), 8, ns)]
        n_samp    = [int(x)   for x in _field(ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80), 8, ns)]

        record_duration = float(header[244:252].decode("ascii").strip())
        n_records       = int(header[236:244].decode("ascii").strip())
        sample_rate     = int(n_samp[0] / record_duration)

        gains   = [(phys_maxs[i] - phys_mins[i]) / (dig_maxs[i] - dig_mins[i])
                   for i in range(ns)]
        offsets = [phys_maxs[i] - gains[i] * dig_maxs[i] for i in range(ns)]

        total_per_ch = n_records * n_samp[0]
        signals = np.empty((ns, total_per_ch), dtype=np.float32)
        record_total = sum(n_samp)

        for rec in range(n_records):
            raw = fh.read(record_total * 2)
            if len(raw) < record_total * 2:
                break
            all_samp = np.frombuffer(raw, dtype=np.int16)
            pos = 0
            for ch in range(ns):
                sl = all_samp[pos: pos + n_samp[ch]].astype(np.float32)
                signals[ch, rec * n_samp[ch]: (rec + 1) * n_samp[ch]] = (
                    sl * gains[ch] + offsets[ch]
                )
                pos += n_samp[ch]

    return signals, sample_rate, labels


# ── Observer 2: dominant-band ──────────────────────────────────────────────────

BAND_RANGES = {
    "gamma": (30.0, 100.0),
    "alpha": (8.0,  13.0),
    "theta": (4.0,   8.0),
    "delta": (0.5,   4.0),
}

def _band_power(signal: np.ndarray, fs: int, lo: float, hi: float) -> float:
    f = np.fft.rfftfreq(len(signal), 1.0 / fs)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    mask = (f >= lo) & (f <= hi)
    return float(psd[mask].sum())


def classify_window_dominant_band(window: np.ndarray, fs: int) -> str:
    """
    Observer 2: assign microstate by which spectral band has peak power.
    All four states are explicit — no fallback.

    Physiological mapping:
      A_frontal  → gamma dominant  (fast high-frequency activity: ictal, attention)
      B_occipital → alpha dominant (8-13 Hz resting rhythm: dominant in interictal)
      C_right     → theta dominant (4-8 Hz: transitional, drowsiness, early seizure)
      D_baseline  → delta dominant (0.5-4 Hz: deep sleep, delta-dominant seizure)
    """
    powers = {band: _band_power(window, fs, lo, hi)
              for band, (lo, hi) in BAND_RANGES.items()}
    dominant = max(powers, key=powers.__getitem__)
    mapping = {
        "gamma": "A_frontal",
        "alpha": "B_occipital",
        "theta": "C_right",
        "delta": "D_baseline",
    }
    return mapping[dominant]


def classify_segment_dominant_band(eeg_1d: np.ndarray, fs: int,
                                   window_sec: float = WINDOW_SECONDS) -> list[str]:
    """Observer 2: slide dominant-band classifier over segment."""
    n_win = int(window_sec * fs)
    step  = max(1, n_win // 2)
    classes = []
    for start in range(0, len(eeg_1d) - n_win + 1, step):
        window = eeg_1d[start: start + n_win]
        classes.append(classify_window_dominant_band(window, fs))
    return classes if classes else ["D_baseline"]


# ── Observer 3: topographic k-means ───────────────────────────────────────────

def _topographic_feature_vector(multi_ch_window: np.ndarray) -> np.ndarray:
    """
    Compute topographic feature vector for a multi-channel EEG window.

    Feature: RMS amplitude per channel, L2-normalized (direction only).
    This captures spatial distribution independent of overall power level.
    Shape: (n_channels,)
    """
    rms = np.sqrt(np.mean(multi_ch_window ** 2, axis=1))  # (n_channels,)
    norm = np.linalg.norm(rms)
    if norm < 1e-9:
        return rms
    return rms / norm


def build_topographic_features(multi_ch_signal: np.ndarray, fs: int,
                                window_sec: float = WINDOW_SECONDS,
                                start_s: float = 0.0,
                                end_s: float = None) -> np.ndarray:
    """
    Compute topographic feature vectors for all windows in a time range.
    Returns array of shape (n_windows, n_channels).
    """
    n_win = int(window_sec * fs)
    step  = max(1, n_win // 2)
    total = multi_ch_signal.shape[1]

    start_samp = int(start_s * fs)
    end_samp   = int(end_s * fs) if end_s is not None else total

    features = []
    for s in range(start_samp, end_samp - n_win + 1, step):
        window = multi_ch_signal[:, s: s + n_win]  # (n_channels, n_win_samples)
        features.append(_topographic_feature_vector(window))

    return np.array(features) if features else np.empty((0, multi_ch_signal.shape[0]))


# ── Fit topographic k-means on ALL data (train on unlabeled pool) ─────────────

def fit_topographic_kmeans(data_dir: Path, n_clusters: int = 4,
                           seed: int = 42) -> KMeans:
    """
    Fit k-means on topographic feature vectors from all EDF files.
    Uses NO seizure labels — purely spatial clustering.
    """
    print("  Fitting topographic k-means (all channels, no labels)...")
    all_features = []
    for ann in CHBMIT_ANNOTATIONS:
        edf_path = data_dir / ann["file"]
        if not edf_path.exists():
            continue
        try:
            sig, fs, _ = _read_edf_all_channels(edf_path)
        except Exception as e:
            print(f"    [SKIP] {ann['file']}: {e}")
            continue
        # Sample from first 600s to avoid loading entire 1-hour file into memory
        sample_end = min(600.0, sig.shape[1] / fs)
        feats = build_topographic_features(sig, fs, end_s=sample_end)
        if len(feats) > 0:
            all_features.append(feats)
        print(f"    {ann['file']}: {len(feats)} windows")

    if not all_features:
        raise RuntimeError("No topographic features extracted")

    X = np.vstack(all_features)
    print(f"  Total feature matrix: {X.shape}")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return km


def classify_segment_topographic(multi_ch_window: np.ndarray,
                                  km: KMeans,
                                  cluster_to_state: dict[int, str],
                                  fs: int,
                                  window_sec: float = WINDOW_SECONDS) -> list[str]:
    """
    Observer 3: classify a multi-channel EEG segment using topographic k-means.
    multi_ch_window: (n_channels, n_samples)
    """
    n_win = int(window_sec * fs)
    step  = max(1, n_win // 2)
    total = multi_ch_window.shape[1]

    classes = []
    for s in range(0, total - n_win + 1, step):
        window = multi_ch_window[:, s: s + n_win]
        fv = _topographic_feature_vector(window).reshape(1, -1)
        cluster_id = int(km.predict(fv)[0])
        classes.append(cluster_to_state[cluster_id])

    return classes if classes else ["D_baseline"]


# ── Common: nested logistic regression ────────────────────────────────────────

def _fit_logistic(X: np.ndarray, y: np.ndarray,
                  lr: float = 0.1, n_iter: int = 3000, l2: float = 1e-4) -> np.ndarray:
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(n_iter):
        logits = np.clip(X @ beta, -30, 30)
        probs = expit(logits)
        grad = X.T @ (probs - y) / n + l2 * beta
        beta -= lr * grad
    return beta


def _ll(X, y, beta):
    logits = np.clip(X @ beta, -30, 30)
    probs = np.clip(expit(logits), 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def mcfadden_r2(ll_model, ll_null):
    return float(1.0 - ll_model / ll_null) if ll_null != 0 else 0.0


def lr_test(ll_restricted, ll_full, df):
    lr_stat = 2.0 * (ll_full - ll_restricted)
    return float(lr_stat), float(stats.chi2.sf(lr_stat, df))


def nested_model_test(y: np.ndarray, delta: np.ndarray,
                      sing: np.ndarray, cos_: np.ndarray,
                      label: str) -> dict:
    """Run nested model test and return result dict."""
    def _std(x):
        mu, sd = x.mean(), x.std()
        return (x - mu) / (sd + 1e-9)

    delta_s = _std(delta)
    sing_s  = _std(sing)
    cos_s   = _std(cos_)

    X0 = np.ones((len(y), 1))
    beta0 = _fit_logistic(X0, y); ll0 = _ll(X0, y, beta0)

    X1 = np.column_stack([np.ones(len(y)), delta_s])
    beta1 = _fit_logistic(X1, y); ll1 = _ll(X1, y, beta1)

    X2 = np.column_stack([np.ones(len(y)), delta_s, sing_s, cos_s])
    beta2 = _fit_logistic(X2, y); ll2 = _ll(X2, y, beta2)

    r2_1 = mcfadden_r2(ll1, ll0)
    r2_2 = mcfadden_r2(ll2, ll0)
    lr_stat, p_val = lr_test(ll1, ll2, df=2)

    return {
        "label": label,
        "r2_delta":    r2_1,
        "r2_delta_qa": r2_2,
        "delta_r2":    r2_2 - r2_1,
        "lr_stat":     lr_stat,
        "p_qa_add":    p_val,
    }


# ── Dataset loader with multi-channel support ─────────────────────────────────

def load_dataset_multichannel(data_dir: Path) -> list[dict]:
    """
    Load CHB-MIT with both single-channel (ch 0) and all-channel signals.
    Returns dicts with keys: type, waveform (ch0, 1D), multi_ch (all, 2D), source.
    """
    from eeg_orbit_classifier import CHBMIT_ANNOTATIONS
    dataset = []
    window_sec = 10.0

    for ann in CHBMIT_ANNOTATIONS:
        edf_path = data_dir / ann["file"]
        if not edf_path.exists():
            continue

        try:
            multi, fs, labels = _read_edf_all_channels(edf_path)
        except Exception as e:
            print(f"  [SKIP] {ann['file']}: {e}")
            continue

        ch0 = multi[0]  # FP1-F7
        n_window = int(window_sec * fs)
        total_samples = multi.shape[1]
        onset_s  = ann["seizure_start_s"]
        offset_s = ann["seizure_end_s"]
        seizure_duration = offset_s - onset_s

        n_ictal = max(1, seizure_duration // int(window_sec))
        for i in range(n_ictal):
            start = int((onset_s + i * window_sec) * fs)
            end = start + n_window
            if end <= total_samples:
                dataset.append({
                    "type": "seizure",
                    "waveform": ch0[start:end],
                    "multi_ch": multi[:, start:end],
                    "source": ann["file"],
                    "fs": fs,
                })

        buffer = int(300 * fs)
        safe_end_before = int(onset_s * fs) - buffer
        safe_start_after = int(offset_s * fs) + buffer
        interictal_starts = []
        pos = n_window
        while pos + n_window <= safe_end_before and len(interictal_starts) < 4:
            interictal_starts.append(pos); pos += n_window * 3
        pos = safe_start_after
        while pos + n_window <= total_samples and len(interictal_starts) < 8:
            interictal_starts.append(pos); pos += n_window * 3

        for start in interictal_starts[:8]:
            dataset.append({
                "type": "baseline",
                "waveform": ch0[start: start + n_window],
                "multi_ch": multi[:, start: start + n_window],
                "source": ann["file"],
                "fs": fs,
            })

        sei_n  = sum(1 for d in dataset if d["type"] == "seizure" and d["source"] == ann["file"])
        base_n = sum(1 for d in dataset if d["type"] == "baseline" and d["source"] == ann["file"])
        print(f"  {ann['file']}: {sei_n} ictal, {base_n} interictal")

    return dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EEG Observer Comparison — Three Observer Designs")
    print("Does observer architecture determine QA independence from delta?")
    print("=" * 70)
    print()

    print("Loading CHB-MIT chb01 (multi-channel)...")
    dataset = load_dataset_multichannel(DEFAULT_DATA_DIR)
    n_sei  = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  Total: {len(dataset)} segments ({n_sei} seizure, {n_base} baseline)")
    print()

    if not dataset:
        print("ERROR: no data loaded"); return

    fs = dataset[0]["fs"]
    y  = np.array([1 if d["type"] == "seizure" else 0 for d in dataset], dtype=float)

    # Delta ratio (same for all observers)
    delta = np.array([delta_power_ratio(d["waveform"].astype(np.float64), fs)
                      for d in dataset])

    # ── Observer 1: threshold-fallback (current) ──────────────────────────────
    print("Observer 1: threshold-fallback (current design)...")
    from eeg_orbit_classifier import classify_eeg_segment

    sing1, cos1 = [], []
    for d in dataset:
        ms = classify_eeg_segment(d["waveform"].ravel(), fs)
        orb = orbit_statistics(compute_orbit_sequence(ms))
        sing1.append(orb["singularity_frac"])
        cos1.append(orb["cosmos_frac"])
    sing1, cos1 = np.array(sing1), np.array(cos1)

    res1 = nested_model_test(y, delta, sing1, cos1, "Observer 1 (threshold-fallback)")
    print(f"  Singularity: seizure {sing1[y==1].mean():.3f} vs baseline {sing1[y==0].mean():.3f}")

    # ── Observer 2: dominant-band ─────────────────────────────────────────────
    print("Observer 2: dominant-band (which band wins)...")

    sing2, cos2 = [], []
    for d in dataset:
        ms = classify_segment_dominant_band(d["waveform"].ravel(), fs)
        orb = orbit_statistics(compute_orbit_sequence(ms))
        sing2.append(orb["singularity_frac"])
        cos2.append(orb["cosmos_frac"])
    sing2, cos2 = np.array(sing2), np.array(cos2)

    # Report microstate distribution
    all_ms2 = []
    for d in dataset:
        all_ms2.extend(classify_segment_dominant_band(d["waveform"].ravel(), fs))
    from collections import Counter
    dist2 = Counter(all_ms2)
    total2 = sum(dist2.values())
    print("  Microstate distribution:")
    for st in ["A_frontal", "B_occipital", "C_right", "D_baseline"]:
        print(f"    {st}: {dist2.get(st,0)/total2*100:.1f}%")

    # Separation
    t2, p2 = stats.ttest_ind(sing2[y==1], sing2[y==0], equal_var=False)
    print(f"  Singularity t-test: t={t2:.3f} p={p2:.4f}")

    res2 = nested_model_test(y, delta, sing2, cos2, "Observer 2 (dominant-band)")

    # ── Observer 3: topographic k-means ───────────────────────────────────────
    print()
    print("Observer 3: topographic k-means (all 23 channels, no spectral features)...")

    # Fit k-means
    km = fit_topographic_kmeans(DEFAULT_DATA_DIR, n_clusters=4)

    # Assign cluster IDs to QA states (by cluster centroid RMS — higher power = more active state)
    centroid_rms = np.linalg.norm(km.cluster_centers_, axis=1)
    rank_order = np.argsort(centroid_rms)  # 0=lowest, 3=highest RMS
    # Map: highest RMS cluster → A_frontal (most active), descending
    state_order = ["D_baseline", "C_right", "B_occipital", "A_frontal"]
    cluster_to_state = {int(rank_order[i]): state_order[i] for i in range(4)}
    print(f"  Cluster → state mapping: {cluster_to_state}")

    sing3, cos3 = [], []
    for d in dataset:
        ms = classify_segment_topographic(d["multi_ch"], km, cluster_to_state, fs)
        orb = orbit_statistics(compute_orbit_sequence(ms))
        sing3.append(orb["singularity_frac"])
        cos3.append(orb["cosmos_frac"])
    sing3, cos3 = np.array(sing3), np.array(cos3)

    # Microstate distribution
    all_ms3 = []
    for d in dataset:
        all_ms3.extend(classify_segment_topographic(d["multi_ch"], km, cluster_to_state, fs))
    dist3 = Counter(all_ms3)
    total3 = sum(dist3.values())
    print("  Microstate distribution:")
    for st in ["A_frontal", "B_occipital", "C_right", "D_baseline"]:
        print(f"    {st}: {dist3.get(st,0)/total3*100:.1f}%")

    t3, p3 = stats.ttest_ind(sing3[y==1], sing3[y==0], equal_var=False)
    print(f"  Singularity t-test: t={t3:.3f} p={p3:.4f}")

    res3 = nested_model_test(y, delta, sing3, cos3, "Observer 3 (topographic k-means)")

    # Correlation of Observer 3 singularity with delta
    r_topo_delta, p_r = stats.pearsonr(sing3, delta)
    print(f"  r(QA_sing3, delta) = {r_topo_delta:.4f} (p={p_r:.4f})")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("NESTED MODEL COMPARISON")
    print(f"  {'Observer':<36}  {'R²(delta)':>9}  {'R²(+QA)':>9}  {'ΔR²':>8}  {'p(QA add)':>10}")
    print(f"  {'-'*36}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*10}")

    for r in [res1, res2, res3]:
        sig = "***" if r["p_qa_add"] < 0.001 else ("**" if r["p_qa_add"] < 0.01
              else ("*" if r["p_qa_add"] < 0.05 else "ns"))
        print(f"  {r['label']:<36}  {r['r2_delta']:>9.4f}  {r['r2_delta_qa']:>9.4f}  "
              f"{r['delta_r2']:>+8.4f}  {r['p_qa_add']:>8.4f} {sig}")

    print()
    print("=" * 70)
    print("VERDICT")
    print()

    for r in [res1, res2, res3]:
        if r["p_qa_add"] < 0.05:
            verdict = f"QA ADDS (ΔR²={r['delta_r2']:+.4f}) — empirical discriminant"
        else:
            verdict = f"QA DOES NOT ADD (ΔR²={r['delta_r2']:+.4f}) — interpretation layer"
        print(f"  {r['label'][:36]}: {verdict}")


if __name__ == "__main__":
    main()
