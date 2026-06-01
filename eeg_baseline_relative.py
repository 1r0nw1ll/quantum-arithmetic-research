#!/usr/bin/env python3
"""
eeg_baseline_relative.py — Baseline-relative orbit deviation features

Key insight from structural analysis: the orbit family direction is patient-specific
(chb09 is Satellite at rest, transitions to Cosmos during seizure; chb13 does the
reverse). The 4-tuple averaging used in the replication script is patient-agnostic
and averages over all k=9 centroids — it cannot capture per-window shifts relative
to each patient's resting state.

Fix: for each window, assign it to its nearest centroid (k-means), read off that
centroid's (b,e) pair, and compute deviations from the training-interictal mean.

Four models compared:
  A: delta only (spectral baseline)
  B: delta + averaged 4-tuple    (current pre-registered protocol)
  C: delta + baseline-relative   (new: per-window orbit deviation from rest)
  D: delta + 4-tuple + deviation (combined)

All models use 70/30 stratified split, seed=42, L2 logistic regression.
Theorem NT compliant: k-means fitted on all windows without seizure labels.
"""

import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pyedflib

warnings.filterwarnings("ignore")

CHBMIT_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
WINDOW_SEC  = 10.0
BUFFER_SEC  = 300.0
K_MEANS     = 9
SEED        = 42
TEST_FRAC   = 0.30
N_CHANNELS  = 23


# ── QA orbit map ──────────────────────────────────────────────────────────────

def _build_orbit_map(m: int = 9) -> dict[tuple, str]:
    def period(b0, e0):
        b, e = b0, e0
        for k in range(1, 100):
            b, e = e, (b + e - 1) % m + 1
            if b == b0 and e == e0:
                return k
        return -1
    return {(b, e): (
        "Singularity" if period(b, e) == 1 else
        "Satellite"   if period(b, e) == 8 else
        "Cosmos"
    ) for b in range(1, m + 1) for e in range(1, m + 1)}

ORBIT_MAP = _build_orbit_map(9)


# ── EDF + annotation loading ──────────────────────────────────────────────────

def read_edf(path: Path) -> tuple[np.ndarray, int]:
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


def parse_summary(path: Path) -> dict[str, list[dict]]:
    ann, cur = defaultdict(list), None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m:
                cur = m.group(1).lower(); continue
            if cur is None:
                continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                ann[cur].append({"start_s": int(m.group(1)), "end_s": None}); continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                for a in reversed(ann.get(cur, [])):
                    if a["end_s"] is None:
                        a["end_s"] = int(m.group(1)); break
    return {k: [a for a in v if a["end_s"] is not None] for k, v in ann.items() if v}


def psd_mean(sig: np.ndarray, fs: int) -> np.ndarray:
    nperseg = min(2 * fs, sig.shape[1])
    return np.mean([welch(sig[ch].astype(np.float64), fs=fs, nperseg=nperseg)[1]
                    for ch in range(sig.shape[0])], axis=0)


def delta_power(sig: np.ndarray, fs: int) -> float:
    nperseg = min(2 * fs, sig.shape[1])
    total = 0.0
    for ch in range(sig.shape[0]):
        freqs, Pxx = welch(sig[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs >= 1.0) & (freqs <= 4.0)
        total += float(np.mean(Pxx[mask])) if mask.any() else 0.0
    return total / sig.shape[0]


def extract_windows(patient_dir: Path) -> list[dict]:
    summaries = list(patient_dir.glob("*-summary.txt"))
    if not summaries:
        return []
    ann = parse_summary(summaries[0])
    if not ann:
        return []

    sei_wins, bas_wins = [], []
    all_ivs: dict[str, list[tuple]] = defaultdict(list)
    for fname, seqs in ann.items():
        for s in seqs:
            all_ivs[fname].append((float(s["start_s"]), float(s["end_s"])))

    for fname, seqs in ann.items():
        edf = patient_dir / fname
        if not edf.exists():
            continue
        try:
            sig, fs = read_edf(edf)
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}"); continue

        total_s = sig.shape[1] / fs
        for s in seqs:
            onset, offset = float(s["start_s"]), float(s["end_s"])
            for i in range(max(1, int((offset - onset) // WINDOW_SEC))):
                t0 = onset + i * WINDOW_SEC
                if t0 + WINDOW_SEC > total_s: break
                s0, s1 = int(t0 * fs), int((t0 + WINDOW_SEC) * fs)
                sei_wins.append({"label": 1, "signals": sig[:, s0:s1], "fs": fs})

        ivs = all_ivs[fname]
        earliest, latest = min(iv[0] for iv in ivs), max(iv[1] for iv in ivs)
        added = 0
        for pos in np.arange(WINDOW_SEC, earliest - BUFFER_SEC, WINDOW_SEC * 3):
            if added >= 4: break
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            bas_wins.append({"label": 0, "signals": sig[:, s0:s1], "fs": fs})
            added += 1
        for pos in np.arange(latest + BUFFER_SEC, total_s - WINDOW_SEC, WINDOW_SEC * 3):
            if added >= 8: break
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            bas_wins.append({"label": 0, "signals": sig[:, s0:s1], "fs": fs})
            added += 1

    if not sei_wins or not bas_wins:
        return []
    rng = np.random.default_rng(SEED)
    n = min(len(sei_wins), len(bas_wins))
    return ([sei_wins[i] for i in rng.choice(len(sei_wins), n, replace=False)] +
            [bas_wins[i] for i in rng.choice(len(bas_wins), n, replace=False)])


# ── Feature extraction ────────────────────────────────────────────────────────

def per_window_qa(windows: list[dict], km: KMeans, pca: PCA) -> np.ndarray:
    """
    For each window: assign to nearest centroid → (b,e) → orbit family.
    Returns float array [n, 4]: [b, e, cosmos_indicator, satellite_indicator].
    """
    feats = []
    for w in windows:
        psd = psd_mean(w["signals"], w["fs"])
        cid = int(km.predict(psd.reshape(1, -1))[0])
        c1c2 = pca.transform(km.cluster_centers_[cid].reshape(1, -1))[0]
        b = int(np.floor(np.abs(c1c2[0])) % 9) + 1
        e = int(np.floor(np.abs(c1c2[1])) % 9) + 1
        fam = ORBIT_MAP[(b, e)]
        feats.append([
            float(b),
            float(e),
            1.0 if fam == "Cosmos"    else 0.0,
            1.0 if fam == "Satellite" else 0.0,
        ])
    return np.array(feats)


def averaged_4tuple(windows: list[dict], km: KMeans, pca: PCA) -> np.ndarray:
    """
    Original protocol: average (b,e,d,a) across all k=9 centroids for each window.
    Returns [n, 4].
    """
    centroids_2d = pca.transform(km.cluster_centers_)
    c1, c2 = centroids_2d[:, 0], centroids_2d[:, 1]
    b = (np.floor(np.abs(c1)).astype(int) % 9) + 1
    e = (np.floor(np.abs(c2)).astype(int) % 9) + 1
    d = ((b + e - 1) % 9) + 1
    a = ((b + 2 * e - 1) % 9) + 1
    # Same for every window (fixed centroids)
    mean_tuple = np.array([b.mean(), e.mean(), d.mean(), a.mean()])
    return np.tile(mean_tuple, (len(windows), 1))


# ── Logistic regression helpers ───────────────────────────────────────────────

def _null_ll(y: np.ndarray) -> float:
    p = np.clip(y.mean(), 1e-10, 1 - 1e-10)
    return float(y.sum() * np.log(p) + (1 - y).sum() * np.log(1 - p))


def _model_ll(lr: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(lr.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def fit_eval(X_tr, y_tr, X_te, y_te, ll_null, n_qa_features) -> dict:
    lr = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr.fit(X_tr, y_tr)
    ll = _model_ll(lr, X_te, y_te)
    r2 = 1.0 - ll / ll_null if ll_null != 0 else 0.0
    lr_stat = max(0.0, 2.0 * (ll - _model_ll(
        LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000).fit(
            X_tr[:, :1], y_tr),
        X_te[:, :1], y_te)))
    p = float(chi2.sf(lr_stat, df=n_qa_features))
    return {"r2": float(r2), "lr_stat": float(lr_stat), "p": float(p)}


# ── Per-patient analysis ──────────────────────────────────────────────────────

def analyse_patient(pid: str, patient_dir: Path) -> dict | None:
    print(f"\n  [{pid}] Loading windows...")
    windows = extract_windows(patient_dir)
    if not windows:
        return None

    n_sei = sum(1 for w in windows if w["label"] == 1)
    n_bas = sum(1 for w in windows if w["label"] == 0)
    if n_sei < 4 or n_bas < 4:
        print(f"  [{pid}] Insufficient data — skip")
        return None
    print(f"  [{pid}] {n_sei} seizure + {n_bas} baseline")

    labels  = np.array([w["label"] for w in windows])
    idx     = np.arange(len(windows))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC,
                                      stratify=labels, random_state=SEED)
    y_tr, y_te = labels[tr_idx], labels[te_idx]

    # Observer: fit on ALL windows, no labels used (Theorem NT)
    psds = np.array([psd_mean(w["signals"], w["fs"]) for w in windows])
    pca  = PCA(n_components=2, random_state=SEED).fit(psds)
    km   = KMeans(n_clusters=K_MEANS, random_state=SEED, n_init=10).fit(psds)

    # Delta baseline feature
    delta = np.array([delta_power(w["signals"], w["fs"]) for w in windows])
    d_tr, d_te = delta[tr_idx].reshape(-1, 1), delta[te_idx].reshape(-1, 1)
    ll_null = _null_ll(y_te)

    # ── Model A: delta only ───────────────────────────────────────────────────
    lr_A = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_A.fit(d_tr, y_tr)
    ll_A = _model_ll(lr_A, d_te, y_te)
    r2_A = 1.0 - ll_A / ll_null if ll_null != 0 else 0.0

    # ── Model B: delta + averaged 4-tuple (pre-registered) ───────────────────
    avg4 = averaged_4tuple(windows, km, pca)
    X_tr_B = np.hstack([d_tr, avg4[tr_idx]])
    X_te_B = np.hstack([d_te, avg4[te_idx]])
    lr_B = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_B.fit(X_tr_B, y_tr)
    ll_B   = _model_ll(lr_B, X_te_B, y_te)
    r2_B   = 1.0 - ll_B / ll_null if ll_null != 0 else 0.0
    lrs_B  = max(0.0, 2.0 * (ll_B - ll_A))
    p_B    = float(chi2.sf(lrs_B, df=4))

    # ── Model C: delta + baseline-relative orbit deviation ────────────────────
    pw = per_window_qa(windows, km, pca)  # [n, 4]: b, e, cosmos, satellite

    # Baseline = mean of training interictal windows
    tr_inter_mask = (labels[tr_idx] == 0)
    if tr_inter_mask.sum() == 0:
        print(f"  [{pid}] No training interictal windows — skip")
        return None
    baseline = pw[tr_idx][tr_inter_mask].mean(axis=0)  # [4]

    # Deviation: per-window value minus patient's own interictal mean
    pw_dev = pw - baseline  # [n, 4]

    X_tr_C = np.hstack([d_tr, pw_dev[tr_idx]])
    X_te_C = np.hstack([d_te, pw_dev[te_idx]])
    lr_C = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_C.fit(X_tr_C, y_tr)
    ll_C  = _model_ll(lr_C, X_te_C, y_te)
    r2_C  = 1.0 - ll_C / ll_null if ll_null != 0 else 0.0
    lrs_C = max(0.0, 2.0 * (ll_C - ll_A))
    p_C   = float(chi2.sf(lrs_C, df=4))

    # ── Model D: delta + 4-tuple + deviation (combined) ───────────────────────
    X_tr_D = np.hstack([d_tr, avg4[tr_idx], pw_dev[tr_idx]])
    X_te_D = np.hstack([d_te, avg4[te_idx], pw_dev[te_idx]])
    lr_D = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_D.fit(X_tr_D, y_tr)
    ll_D  = _model_ll(lr_D, X_te_D, y_te)
    r2_D  = 1.0 - ll_D / ll_null if ll_null != 0 else 0.0
    lrs_D = max(0.0, 2.0 * (ll_D - ll_A))
    p_D   = float(chi2.sf(lrs_D, df=8))

    # Baseline per-window orbit fracs (informational)
    b_b = float(baseline[0])
    b_e = float(baseline[1])
    b_cosmos = float(baseline[2])
    b_sat    = float(baseline[3])

    # Mean deviation in ictal windows
    te_ictal = te_idx[y_te == 1]
    te_inter = te_idx[y_te == 0]
    ic_dev_cosmos = float(pw_dev[te_ictal, 2].mean()) if len(te_ictal) else 0.0
    ic_dev_sat    = float(pw_dev[te_ictal, 3].mean()) if len(te_ictal) else 0.0
    in_dev_cosmos = float(pw_dev[te_inter, 2].mean()) if len(te_inter) else 0.0
    in_dev_sat    = float(pw_dev[te_inter, 3].mean()) if len(te_inter) else 0.0

    print(f"  [{pid}] Baseline: cosmos={b_cosmos:.3f}  satellite={b_sat:.3f}")
    print(f"  [{pid}] Ictal deviation: Δcosmos={ic_dev_cosmos:+.3f}  Δsatellite={ic_dev_sat:+.3f}")
    print(f"  [{pid}] ΔR²: A(delta)={r2_A:.4f}  "
          f"B(+4tuple)={r2_B-r2_A:+.4f}  "
          f"C(+dev)={r2_C-r2_A:+.4f}  "
          f"D(+both)={r2_D-r2_A:+.4f}")

    return {
        "patient":          pid,
        "n_sei":            int(n_sei),
        "n_bas":            int(n_bas),
        "baseline_cosmos":  round(b_cosmos, 4),
        "baseline_sat":     round(b_sat, 4),
        "ictal_dev_cosmos": round(ic_dev_cosmos, 4),
        "ictal_dev_sat":    round(ic_dev_sat, 4),
        "r2_A":             round(r2_A, 5),
        "delta_r2_B":       round(r2_B - r2_A, 5),
        "delta_r2_C":       round(r2_C - r2_A, 5),
        "delta_r2_D":       round(r2_D - r2_A, 5),
        "p_B":              round(p_B, 6),
        "p_C":              round(p_C, 6),
        "p_D":              round(p_D, 6),
    }


def fishers(ps: list[float]) -> tuple[float, float]:
    ps = [max(1e-15, p) for p in ps]
    stat = -2.0 * sum(np.log(p) for p in ps)
    return float(stat), float(chi2.sf(stat, 2 * len(ps)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EEG Baseline-Relative Orbit Features — Model Comparison")
    print("  A: delta only")
    print("  B: delta + averaged 4-tuple   (pre-registered protocol)")
    print("  C: delta + per-window orbit deviation from interictal baseline")
    print("  D: delta + 4-tuple + deviation (combined)")
    print("=" * 72)

    patient_dirs = sorted(CHBMIT_ROOT.glob("chb*/"))
    ready = [pd for pd in patient_dirs
             if list(pd.glob("*-summary.txt")) and not list(pd.glob("*.tmp"))]
    print(f"\nPatients: {[p.name for p in ready]}\n")

    results = []
    for pdir in ready:
        res = analyse_patient(pdir.name, pdir)
        if res:
            results.append(res)

    if not results:
        print("No results."); return

    n = len(results)
    print("\n" + "=" * 72)
    print("PER-PATIENT RESULTS")
    print(f"  {'Pat':<6}  {'baseline':>8}  {'dev_ictal':>9}  "
          f"{'ΔR²_B':>7}  {'ΔR²_C':>7}  {'ΔR²_D':>7}  "
          f"{'p_B':>8}  {'p_C':>8}  {'p_D':>8}")
    print(f"  {'':6}  {'cos|sat':>8}  {'Δcos|Δsat':>9}  "
          f"{'4tuple':>7}  {'dev':>7}  {'both':>7}  "
          f"{'':8}  {'':8}  {'':8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}"
          f"  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in results:
        bcs = f"{r['baseline_cosmos']:.2f}|{r['baseline_sat']:.2f}"
        dev = f"{r['ictal_dev_cosmos']:+.2f}|{r['ictal_dev_sat']:+.2f}"
        print(f"  {r['patient']:<6}  {bcs:>8}  {dev:>9}  "
              f"  {r['delta_r2_B']:>+6.4f}  {r['delta_r2_C']:>+6.4f}"
              f"  {r['delta_r2_D']:>+6.4f}"
              f"  {r['p_B']:>8.4f}  {r['p_C']:>8.4f}  {r['p_D']:>8.4f}")

    print()
    for label, key_dr2, key_p in [
        ("B (4-tuple)",  "delta_r2_B", "p_B"),
        ("C (deviation)", "delta_r2_C", "p_C"),
        ("D (combined)", "delta_r2_D", "p_D"),
    ]:
        vals = [r[key_dr2] for r in results]
        ps   = [r[key_p]   for r in results]
        mean = float(np.mean(vals))
        se   = float(np.std(vals) / np.sqrt(n))
        n_sig = sum(1 for p in ps if p < 0.05)
        n_pos = sum(1 for v in vals if v > 0)
        chi2_f, p_fish = fishers(ps)
        print(f"  Model {label}:")
        print(f"    mean ΔR²={mean:+.4f} ± {se:.4f}   "
              f"pos={n_pos}/{n}  sig={n_sig}/{n}  "
              f"Fisher χ²={chi2_f:.1f}  p={p_fish:.3e}")

    print()
    print("WINNER: model with best mean ΔR²:")
    best = max(["delta_r2_B", "delta_r2_C", "delta_r2_D"],
               key=lambda k: np.mean([r[k] for r in results]))
    labels_map = {"delta_r2_B": "B (4-tuple)",
                  "delta_r2_C": "C (deviation)",
                  "delta_r2_D": "D (combined)"}
    best_val = float(np.mean([r[best] for r in results]))
    print(f"    {labels_map[best]} — mean ΔR²={best_val:+.4f}")

    out = {
        "patients": results,
        "aggregate": {m: {
            "mean_delta_r2": round(float(np.mean([r[f"delta_r2_{m}"] for r in results])), 5),
            "se":            round(float(np.std([r[f"delta_r2_{m}"] for r in results]) /
                                         np.sqrt(n)), 5),
            "n_positive":    sum(1 for r in results if r[f"delta_r2_{m}"] > 0),
            "n_significant": sum(1 for r in results if r[f"p_{m}"] < 0.05),
            "fisher_p":      round(fishers([r[f"p_{m}"] for r in results])[1], 8),
        } for m in ["B", "C", "D"]},
    }
    out_path = Path("results/eeg_baseline_relative.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
