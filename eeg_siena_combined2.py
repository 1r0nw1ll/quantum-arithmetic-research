#!/usr/bin/env python3
# RT1_OBSERVER_FILE: spectral bandpass and power are observer projections of EEG signal
"""Siena Scalp EEG replication of CHB-MIT combined-2 protocol.

Pre-registration: docs/specs/EEG_SIENA_PREREGISTRATION.md
Dataset: PhysioNet siena-scalp-eeg v1.0.0 (doi:10.13026/5d4a-j060)
Protocol (fixed before data access):
  - Features: f0_sing_z (topographic Singularity fraction) +
              f2_sat_z  (multiband Satellite fraction), both signed z-score
  - Baseline: delta-band power
  - Split: 70/30 train/test stratified, SEED=42
  - Inclusion: >= 12 ictal windows
  - Aggregation: Fisher's method; primary endpoint Fisher p < 0.05
"""
import json, re, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch, butter, filtfilt
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

SIENA_ROOT = Path(__file__).parent / "archive/phase_artifacts/phase2_data/eeg/siena"
RESULTS_OUT = Path(__file__).parent / "eeg_siena_combined2_results.json"
PREREG_SHA_FILE = Path(__file__).parent / "docs/specs/EEG_SIENA_PREREGISTRATION.md"

# ── Constants (identical to CHB-MIT combined-2) ────────────────────────────────

WINDOW_SEC   = 10.0
BUFFER_SEC   = 300.0
SEED         = 42
TEST_FRAC    = 0.30
N_CHANNELS   = 23   # use first 23 channels (standard 10-20, avoids EKG ch)
MIN_ICTAL    = 12   # inclusion threshold
SUB_SEC      = 1.0

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 50),
}
BAND_LIST = list(BANDS.items())

# ── QA orbit helpers ──────────────────────────────────────────────────────────

def _orbit_period_m24(b0, e0, m=24):
    b, e = b0, e0
    for k in range(1, 300):
        b, e = e, (b + e - 1) % m + 1
        if b == b0 and e == e0:
            return k
    return -1

def orbit_fam(b, e, m=24):
    p = _orbit_period_m24(b, e, m)
    if p == 1:  return "Singularity"
    if p == 8:  return "Satellite"
    return "Cosmos"

MICROSTATE_STATES = {
    "A_frontal":   (8,  3),
    "B_occipital": (5, 16),
    "C_right":     (11, 19),
    "D_baseline":  (24, 24),
}
MICROSTATE_STATES_INT = {0: (8, 3), 1: (5, 16), 2: (11, 19), 3: (24, 24)}
TRANSITION_TABLE = {
    (s1, s2): orbit_fam(MICROSTATE_STATES[s1][0], MICROSTATE_STATES[s2][1], m=24)
    for s1 in MICROSTATE_STATES for s2 in MICROSTATE_STATES
}

# ── Siena annotation parser ───────────────────────────────────────────────────

_TIME_RE = re.compile(r'\b(\d{1,2})[.:](\d{2})[.:](\d{2})\b')

def _hms_to_sec(s):
    """Extract first H[H].MM.SS or H[H]:MM:SS pattern from a line, return seconds.
    Uses regex so trailing garbage (e.g. 'opure 11.40.43') and stray chars
    (e.g. '1 6.49.25' typo) are handled gracefully. Returns None if no match."""
    m = _TIME_RE.search(s)
    if not m:
        return None
    h, mn, sc = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return h * 3600 + mn * 60 + sc

def _norm_fname(s):
    """Lowercase + fix PNOx→PN0x typo in PN06 annotation."""
    return re.sub(r'pno(\d)', r'pn0\1', s.strip().lower())

def parse_siena_summary(path):
    """Return {edf_filename: [{start_s, end_s}, ...]} from Seizures-list-PNxx.txt.

    Handles dataset quirks:
      - 'Start time:' / 'End time:' as well as 'Seizure start/end time:'
      - Trailing garbage after timestamp ('opure', clinical notes, etc.)
      - Typo '1 6.49.25' — regex skips stray leading chars
      - 'PN01.edf' annotation name vs 'PN01-1.edf' on disk (resolved in extract_windows)
    """
    ann = {}
    cur_file = None
    reg_start = None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            m = re.match(r'File name:\s*(\S+\.edf)', line, re.I)
            if m:
                cur_file = _norm_fname(m.group(1))
                reg_start = None
                continue
            m = re.match(r'Registration start time:', line, re.I)
            if m and cur_file is not None:
                t = _hms_to_sec(line)
                if t is not None:
                    reg_start = t
                continue
            # Accept both 'Seizure start time:' and bare 'Start time:'
            m = re.match(r'(?:Seizure\s+)?start\s+time:', line, re.I)
            if m and cur_file is not None and reg_start is not None:
                t = _hms_to_sec(line)
                if t is not None:
                    offset = t - reg_start
                    if offset < 0:
                        offset += 86400
                    if cur_file not in ann:
                        ann[cur_file] = []
                    ann[cur_file].append({"start_s": offset, "end_s": None})
                continue
            m = re.match(r'(?:Seizure\s+)?end\s+time:', line, re.I)
            if m and cur_file is not None and reg_start is not None:
                t = _hms_to_sec(line)
                if t is not None:
                    offset = t - reg_start
                    if offset < 0:
                        offset += 86400
                    for a in reversed(ann.get(cur_file, [])):
                        if a["end_s"] is None:
                            a["end_s"] = offset
                            break
    return {k: [a for a in v if a["end_s"] is not None] for k, v in ann.items() if v}

# ── EDF reader ────────────────────────────────────────────────────────────────

def read_edf(path):
    import pyedflib
    with pyedflib.EdfReader(str(path)) as f:
        n_use = min(f.signals_in_file, N_CHANNELS)
        fs    = int(round(f.getSampleFrequency(0)))
        n_s   = f.getNSamples()[0]
        sig   = np.zeros((n_use, n_s), dtype=np.float32)
        for i in range(n_use):
            sig[i] = f.readSignal(i)
    if n_use < N_CHANNELS:
        sig = np.pad(sig, ((0, N_CHANNELS - n_use), (0, 0)))
    return sig, fs

# ── Window extraction ─────────────────────────────────────────────────────────

def extract_windows(patient_dir):
    summaries = list(patient_dir.glob("Seizures-list-*.txt"))
    if not summaries:
        return [], "no summary file"
    ann = parse_siena_summary(summaries[0])
    if not ann:
        return [], "no parsed seizures"

    all_ivs = defaultdict(list)
    for fname, seqs in ann.items():
        for s in seqs:
            all_ivs[fname].append((float(s["start_s"]), float(s["end_s"])))

    sei_wins, bas_wins = [], []
    rng = np.random.default_rng(SEED)

    for fname, seqs in ann.items():
        # Try exact normalised name first
        candidates = [p for p in patient_dir.glob("*.edf") if p.name.lower() == fname]
        if not candidates:
            # Annotation may say 'PN01.edf' when disk has 'PN01-1.edf' — prefix match
            stem = fname.replace(".edf", "")
            candidates = [p for p in sorted(patient_dir.glob("*.edf"))
                          if p.name.lower().startswith(stem)]
        if not candidates:
            continue
        edf = candidates[0]
        try:
            sig, fs = read_edf(edf)
        except Exception as e:
            print(f"  skip {edf.name}: {e}")
            continue
        total_s = sig.shape[1] / fs

        for s in seqs:
            onset  = float(s["start_s"])
            offset = min(float(s["end_s"]), total_s)  # clamp to file duration
            if offset <= onset:
                continue
            n_win = max(1, int((offset - onset) // WINDOW_SEC))
            for i in range(n_win):
                t0 = onset + i * WINDOW_SEC
                if t0 + WINDOW_SEC > total_s:
                    break
                s0, s1 = int(t0 * fs), int((t0 + WINDOW_SEC) * fs)
                sei_wins.append({"label": 1, "sig": sig[:, s0:s1], "fs": fs})

        ivs = all_ivs[fname]
        earliest = min(iv[0] for iv in ivs)
        latest   = max(iv[1] for iv in ivs)
        added = 0
        for pos in np.arange(WINDOW_SEC, earliest - BUFFER_SEC, WINDOW_SEC * 3):
            if added >= 5:
                break
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            bas_wins.append({"label": 0, "sig": sig[:, s0:s1], "fs": fs})
            added += 1
        for pos in np.arange(latest + BUFFER_SEC, total_s - WINDOW_SEC, WINDOW_SEC * 3):
            if added >= 10:
                break
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            bas_wins.append({"label": 0, "sig": sig[:, s0:s1], "fs": fs})
            added += 1

    if not sei_wins or not bas_wins:
        return [], f"no windows (sei={len(sei_wins)}, bas={len(bas_wins)})"
    n = min(len(sei_wins), len(bas_wins))
    idx_s = rng.choice(len(sei_wins), n, replace=False)
    idx_b = rng.choice(len(bas_wins), n, replace=False)
    return ([sei_wins[i] for i in idx_s] + [bas_wins[i] for i in idx_b]), "ok"

# ── Signal processing (observer projections) ──────────────────────────────────

def bandpass(sig, fs, low, high):
    nyq = fs / 2.0
    lo, hi = max(low / nyq, 1e-3), min(high / nyq, 0.999)
    if lo >= hi:
        return sig
    b, a = butter(4, [lo, hi], btype="band")
    return filtfilt(b, a, sig.astype(np.float64), axis=1)

def band_power(sig, fs, low, high):
    nperseg = min(2 * fs, sig.shape[1])
    rows = []
    for ch in range(sig.shape[0]):
        freqs, Pxx = welch(sig[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs >= low) & (freqs <= high)
        rows.append(float(np.mean(Pxx[mask])) if mask.any() else 0.0)
    return np.array(rows)

def delta_power_scalar(sig, fs):
    return float(band_power(sig, fs, 1, 4).mean())

# ── Microstate / feature extraction ──────────────────────────────────────────

def topo_fv(mc):
    rms  = np.sqrt(np.mean(mc.astype(np.float64) * mc.astype(np.float64), axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms

def fit_topo_km(windows):
    all_fvs = []
    for w in windows:
        sub_n = int(SUB_SEC * w["fs"])
        step  = max(1, sub_n // 2)
        for s in range(0, w["sig"].shape[1] - sub_n + 1, step):
            all_fvs.append(topo_fv(w["sig"][:, s: s + sub_n]))
    km    = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(np.array(all_fvs))
    norms = np.linalg.norm(km.cluster_centers_, axis=1)
    rank  = np.argsort(norms)
    names = ["D_baseline", "C_right", "B_occipital", "A_frontal"]
    order = [""] * 4
    for i, cid in enumerate(rank):
        order[cid] = names[i]
    return km, order

def f0_sing_frac(sig, fs, km, state_order):
    sub_n  = int(SUB_SEC * fs)
    step   = max(1, sub_n // 2)
    states = [state_order[int(km.predict(topo_fv(sig[:, s:s+sub_n]).reshape(1, -1))[0])]
              for s in range(0, sig.shape[1] - sub_n + 1, step)]
    if len(states) < 2:
        return 0.0
    orbits = [TRANSITION_TABLE[(states[i], states[i+1])] for i in range(len(states) - 1)]
    return orbits.count("Singularity") / len(orbits)

def band_topo_fv(sig, fs, low, high):
    filt = bandpass(sig, fs, low, high)
    rms  = np.sqrt(np.mean(filt * filt, axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms

def fit_band_kms(windows):
    """Return list of (km, order) per band. order[cluster_id] → rank 0-3 by RMS norm."""
    result = []
    for (_, (lo, hi)) in BAND_LIST:
        fvs   = np.array([band_topo_fv(w["sig"], w["fs"], lo, hi) for w in windows])
        km    = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(fvs)
        norms = np.linalg.norm(km.cluster_centers_, axis=1)
        rank  = np.argsort(norms)
        order = np.empty(4, dtype=int)
        for i, cid in enumerate(rank):
            order[cid] = i
        result.append((km, order))
    return result

def f2_sat_frac(sig, fs, kms_orders):
    """Satellite fraction across adjacent-band microstate transitions.
    Identical logic to CHB-MIT combined-2: rank-ordered KMeans label →
    MICROSTATE_STATES_INT → orbit_fam on cross-band (b_i, e_{i+1}) pairs."""
    assignments = [
        int(order[int(km.predict(band_topo_fv(sig, fs, lo, hi).reshape(1, -1))[0])])
        for (_, (lo, hi)), (km, order) in zip(BAND_LIST, kms_orders)
    ]
    orbits = [
        orbit_fam(MICROSTATE_STATES_INT[assignments[i]][0],
                  MICROSTATE_STATES_INT[assignments[i + 1]][1], m=24)
        for i in range(len(assignments) - 1)
    ]
    return orbits.count("Satellite") / len(orbits) if orbits else 0.0

# ── Logistic regression evaluation ───────────────────────────────────────────

def _null_ll(y):
    p = np.clip(y.mean(), 1e-10, 1 - 1e-10)
    return float(y.sum() * np.log(p) + (1 - y).sum() * np.log(1 - p))

def _model_ll(lr, X, y):
    probs = np.clip(lr.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))

def eval_delta_r2(feats, labels):
    idx = np.arange(len(labels))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC,
                                      stratify=labels, random_state=SEED)
    y_tr, y_te = labels[tr_idx], labels[te_idx]
    ll_null = _null_ll(y_te)

    lr_base = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_base.fit(feats[tr_idx, :1], y_tr)
    ll_base = _model_ll(lr_base, feats[te_idx, :1], y_te)
    r2_base = 1.0 - ll_base / ll_null if ll_null else 0.0

    lr_aug = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_aug.fit(feats[tr_idx], y_tr)
    ll_aug  = _model_ll(lr_aug, feats[te_idx], y_te)
    r2_aug  = 1.0 - ll_aug / ll_null if ll_null else 0.0

    lrs = max(0.0, 2.0 * (ll_aug - ll_base))
    p   = float(chi2.sf(lrs, df=feats.shape[1] - 1))
    return float(r2_aug - r2_base), float(p)

def fishers(ps):
    ps   = [max(1e-15, p) for p in ps]
    stat = -2.0 * sum(np.log(p) for p in ps)
    return float(stat), float(chi2.sf(stat, 2 * len(ps)))

# ── Per-patient analysis ──────────────────────────────────────────────────────

def analyse_patient(patient_dir, min_ictal=MIN_ICTAL):
    name = patient_dir.name
    windows, reason = extract_windows(patient_dir)
    if not windows:
        return {"patient": name, "status": f"skip: {reason}"}

    ictal_n = sum(1 for w in windows if w["label"] == 1)
    if ictal_n < min_ictal:
        return {"patient": name, "status": f"skip: only {ictal_n} ictal windows"}

    labels = np.array([w["label"] for w in windows])
    idx    = np.arange(len(labels))
    tr_idx, _ = train_test_split(idx, test_size=TEST_FRAC,
                                 stratify=labels, random_state=SEED)
    train_wins = [windows[i] for i in tr_idx]
    interictal_train = [w for w in train_wins if w["label"] == 0]

    # Fit microstate models on training-interictal
    km_topo, state_order = fit_topo_km(interictal_train)
    kms_orders           = fit_band_kms(interictal_train)

    # Extract raw features for all windows
    f0_raw = np.array([f0_sing_frac(w["sig"], w["fs"], km_topo, state_order)
                       for w in windows])
    f2_raw = np.array([f2_sat_frac(w["sig"], w["fs"], kms_orders)
                       for w in windows])

    # Per-patient z-score using training-interictal mean/std
    f0_tr_int = f0_raw[tr_idx][labels[tr_idx] == 0]
    f2_tr_int = f2_raw[tr_idx][labels[tr_idx] == 0]

    def zscore_from(vals, mu, sig):
        return (vals - mu) / sig if sig > 1e-9 else np.zeros_like(vals)

    f0_z = zscore_from(f0_raw, f0_tr_int.mean(), f0_tr_int.std())
    f2_z = zscore_from(f2_raw, f2_tr_int.mean(), f2_tr_int.std())

    # Delta-band baseline feature
    delta = np.array([delta_power_scalar(w["sig"], w["fs"]) for w in windows])
    feats = np.column_stack([delta, f0_z, f2_z])

    dr2, p = eval_delta_r2(feats, labels)

    # Diagnostic: ictal z-score displacement
    ictal_mask = labels == 1
    f0_ictal_z = float(f0_z[ictal_mask].mean()) if ictal_mask.any() else 0.0
    f2_ictal_z = float(f2_z[ictal_mask].mean()) if ictal_mask.any() else 0.0

    return {
        "patient":     name,
        "status":      "ok",
        "n_windows":   len(windows),
        "n_ictal":     int(ictal_n),
        "delta_r2":    round(dr2, 4),
        "p_lr":        round(p, 4),
        "f0_ictal_z":  round(f0_ictal_z, 2),
        "f2_ictal_z":  round(f2_ictal_z, 2),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Record pre-registration git SHA
    import subprocess
    try:
        prereg_sha = subprocess.check_output(
            ["git", "log", "--format=%H", "-1", "--", str(PREREG_SHA_FILE)],
            cwd=str(PREREG_SHA_FILE.parent.parent),
        ).decode().strip()
    except Exception:
        prereg_sha = "unknown"

    patient_dirs = sorted(SIENA_ROOT.glob("PN*/"))
    if not patient_dirs:
        print(f"ERROR: no patient directories found under {SIENA_ROOT}")
        print("Run:  bash eeg_siena_download.sh")
        sys.exit(1)

    # Run analysis at both thresholds; cache per-patient work
    results = []
    for pd in patient_dirs:
        print(f"\n--- {pd.name} ---")
        r = analyse_patient(pd, min_ictal=MIN_ICTAL)
        # Also compute at threshold=6 if it passed threshold=6 but not 12
        r6 = analyse_patient(pd, min_ictal=6)
        r["result_t6"] = {k: v for k, v in r6.items()} if r6["status"] == "ok" else None
        results.append(r)
        if r["status"] == "ok":
            print(f"  ΔR²={r['delta_r2']:+.3f}  p={r['p_lr']:.4f}  "
                  f"f0_z={r['f0_ictal_z']:+.1f}  f2_z={r['f2_ictal_z']:+.1f}")
        elif r6["status"] == "ok":
            print(f"  (t6) ΔR²={r6['delta_r2']:+.3f}  p={r6['p_lr']:.4f}  "
                  f"f0_z={r6['f0_ictal_z']:+.1f}  f2_z={r6['f2_ictal_z']:+.1f}")
        else:
            print(f"  {r['status']}")

    def summarise(label, pool, prereg):
        print(f"\n{'='*55}")
        print(f"{label}  —  Included: {len(pool)}/{len(results)} patients")
        if not pool:
            return None
        dr2s = [r["delta_r2"] for r in pool]
        ps   = [r["p_lr"]     for r in pool]
        mean_dr2 = np.mean(dr2s)
        se_dr2   = np.std(dr2s, ddof=1) / np.sqrt(len(dr2s))
        chi2_stat, fisher_p = fishers(ps)
        n_pos = sum(1 for d in dr2s if d > 0)
        print(f"Mean ΔR²: {mean_dr2:+.3f}  SE: {se_dr2:.3f}")
        print(f"Fisher χ²: {chi2_stat:.1f}  p = {fisher_p:.2e}  (df={2*len(pool)})")
        print(f"Positive ΔR²: {n_pos}/{len(pool)}")
        if fisher_p < 0.05 and n_pos > len(pool) / 2:
            verdict = "REPLICATION_PASS"
        elif fisher_p < 0.05:
            verdict = "SIGNIFICANT_REVERSED_POLARITY"
        elif n_pos > len(pool) / 2:
            verdict = "TREND_ONLY"
        else:
            verdict = "NON_REPLICATION"
        print(f"Verdict: {verdict}")
        return {
            "label":           label,
            "prereg":          prereg,
            "n_included":      len(pool),
            "n_total":         len(results),
            "mean_delta_r2":   round(float(mean_dr2), 4),
            "se_delta_r2":     round(float(se_dr2), 4),
            "fisher_chi2":     round(float(chi2_stat), 2),
            "fisher_p":        float(fisher_p),
            "n_positive_dr2":  n_pos,
            "verdict":         verdict,
        }

    included_t12 = [r for r in results if r["status"] == "ok"]
    included_t6  = [r["result_t6"] for r in results if r.get("result_t6") is not None]

    s12 = summarise("PRE-REGISTERED (threshold=12)", included_t12, prereg=True)
    s6  = summarise("POST-HOC sensitivity (threshold=6, declared)", included_t6, prereg=False)

    payload = {
        "prereg_sha":        prereg_sha,
        "dataset":           "siena-scalp-eeg/1.0.0",
        "preregistered":     s12,
        "posthoc_t6":        s6,
        "per_patient":       results,
    }
    with open(RESULTS_OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nResults written to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
