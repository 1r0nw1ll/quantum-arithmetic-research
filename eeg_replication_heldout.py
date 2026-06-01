#!/usr/bin/env python3
"""
eeg_replication_heldout.py — Pre-registered replication on held-out CHB-MIT patients

Pre-committed protocol (from papers/ready-for-submission/qa-eeg-seizure/paper.tex):
  - k=9 centroids, random seed=42
  - 70/30 stratified split
  - L2 logistic regression (sklearn defaults)
  - McFadden pseudo-R² as primary outcome
  - LR test per patient, Fisher's method for aggregation

Success criterion: mean ΔR² > 0.05 AND at least 7/13 patients significant (p < 0.05).

Held-out patients (never analysed in original study):
  chb08, chb09, chb10, chb12, chb13, chb14, chb15, chb16, chb17, chb19, chb21, chb22, chb23

Observer projection (Theorem NT compliant):
  EEG signal → 23-ch PSD (Welch) → k-means (k=9) → PCA → (c1,c2) → mod-9 → (b,e,d,a)
  Seizure labels NEVER touch the discretisation map.
"""

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pyedflib

warnings.filterwarnings("ignore")

CHBMIT_ROOT = Path(__file__).parent / "archive/phase_artifacts/phase2_data/eeg/chbmit"
BASE_URL = "https://physionet.org/files/chbmit/1.0.0"

HELD_OUT = [
    "chb08", "chb09", "chb10",
    "chb12", "chb13", "chb14", "chb15", "chb16", "chb17",
    "chb19",
    "chb21", "chb22", "chb23",
]

WINDOW_SEC = 10.0
BUFFER_SEC = 300.0
N_BASELINE = 8
K_MEANS = 9
SEED = 42
TEST_FRAC = 0.30
FS_EXPECTED = 256
N_CHANNELS = 23


# ── EDF reader ────────────────────────────────────────────────────────────────

def read_edf(path: Path) -> tuple[np.ndarray, float]:
    """Read EDF, return (signals [n_ch, n_samples], fs). Clips to first N_CHANNELS."""
    with pyedflib.EdfReader(str(path)) as f:
        n_ch = f.signals_in_file
        fs = f.getSampleFrequency(0)
        n_use = min(n_ch, N_CHANNELS)
        n_samp = f.getNSamples()[0]
        signals = np.zeros((n_use, n_samp), dtype=np.float32)
        for i in range(n_use):
            signals[i] = f.readSignal(i)
    return signals, float(fs)


# ── Summary parser ────────────────────────────────────────────────────────────

def parse_summary(path: Path) -> dict[str, list[dict]]:
    """Parse *-summary.txt → {edf_filename: [{start_s, end_s}, ...]}"""
    ann: dict[str, list[dict]] = defaultdict(list)
    cur = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m:
                cur = m.group(1).lower()
                continue
            if cur is None:
                continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                ann[cur].append({"start_s": int(m.group(1)), "end_s": None})
                continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                for a in reversed(ann.get(cur, [])):
                    if a["end_s"] is None:
                        a["end_s"] = int(m.group(1))
                        break
    return {k: [a for a in v if a["end_s"] is not None]
            for k, v in ann.items() if v}


# ── Window extraction ─────────────────────────────────────────────────────────

def extract_windows(patient_dir: Path) -> list[dict]:
    """
    Return list of {label: int, signals: np.ndarray [n_ch, n_samp]}.
    label=1 ictal, label=0 interictal. Balanced by undersampling majority.
    """
    summaries = list(patient_dir.glob("*-summary.txt"))
    if not summaries:
        return []
    ann = parse_summary(summaries[0])
    if not ann:
        return []

    seizure_windows = []
    baseline_windows = []

    all_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for fname, seqs in ann.items():
        for s in seqs:
            all_intervals[fname].append((float(s["start_s"]), float(s["end_s"])))

    for fname, seqs in ann.items():
        edf_path = patient_dir / fname
        if not edf_path.exists():
            continue
        try:
            sig, fs = read_edf(edf_path)
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}")
            continue

        n_ch, n_samp = sig.shape
        if n_ch < N_CHANNELS:
            sig = np.pad(sig, ((0, N_CHANNELS - n_ch), (0, 0)))
        fs = int(round(fs))
        win = int(WINDOW_SEC * fs)
        total_s = n_samp / fs

        for s in seqs:
            onset, offset = float(s["start_s"]), float(s["end_s"])
            n_ic = max(1, int((offset - onset) // WINDOW_SEC))
            for i in range(n_ic):
                t0 = onset + i * WINDOW_SEC
                if t0 + WINDOW_SEC > total_s:
                    break
                s0, s1 = int(t0 * fs), int((t0 + WINDOW_SEC) * fs)
                seizure_windows.append({"label": 1, "signals": sig[:, s0:s1], "fs": fs})

        ivs = all_intervals[fname]
        earliest = min(iv[0] for iv in ivs)
        latest = max(iv[1] for iv in ivs)
        added = 0

        pos = WINDOW_SEC
        while pos + WINDOW_SEC <= earliest - BUFFER_SEC and added < N_BASELINE // 2:
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            baseline_windows.append({"label": 0, "signals": sig[:, s0:s1], "fs": fs})
            added += 1
            pos += WINDOW_SEC * 3

        pos = latest + BUFFER_SEC
        while pos + WINDOW_SEC <= total_s and added < N_BASELINE:
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            baseline_windows.append({"label": 0, "signals": sig[:, s0:s1], "fs": fs})
            added += 1
            pos += WINDOW_SEC * 3

    if not seizure_windows or not baseline_windows:
        return []

    # Balance by undersampling majority
    rng = np.random.default_rng(SEED)
    n = min(len(seizure_windows), len(baseline_windows))
    sei = rng.choice(len(seizure_windows), n, replace=False).tolist()
    bas = rng.choice(len(baseline_windows), n, replace=False).tolist()
    return [seizure_windows[i] for i in sei] + [baseline_windows[i] for i in bas]


# ── Observer projection (Theorem NT) ─────────────────────────────────────────

def psd_matrix(signals: np.ndarray, fs: int) -> np.ndarray:
    """
    Compute channel×frequency PSD matrix using Welch (2s segments).
    Returns array [n_ch, n_freq].
    """
    rows = []
    nperseg = min(2 * fs, signals.shape[1])
    for ch in range(signals.shape[0]):
        _, Pxx = welch(signals[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        rows.append(Pxx)
    return np.array(rows)  # [n_ch, n_freq]


def delta_power(signals: np.ndarray, fs: int) -> float:
    """Mean delta-band (1-4 Hz) power across channels."""
    nperseg = min(2 * fs, signals.shape[1])
    total = 0.0
    for ch in range(signals.shape[0]):
        freqs, Pxx = welch(signals[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs >= 1.0) & (freqs <= 4.0)
        total += float(np.mean(Pxx[mask])) if mask.any() else 0.0
    return total / signals.shape[0]


def window_to_qa_tuple(psd_mat: np.ndarray, km: KMeans, pca: PCA) -> np.ndarray:
    """
    Map a channel×frequency PSD matrix to an averaged (b,e,d,a) 4-tuple.

    Steps:
      1. km.cluster_centers_: [k=9, n_freq_features] — already fitted
      2. Project each centroid through pca → (c1, c2)
      3. b = (floor|c1| mod 9) + 1, e = (floor|c2| mod 9) + 1
      4. d = (b+e-1) mod 9 + 1, a = (b+2e-1) mod 9 + 1
      5. Average (b,e,d,a) across k=9 centroids
    """
    # Project centroids (already computed at fit time)
    centroids_2d = pca.transform(km.cluster_centers_)  # [k, 2]
    c1, c2 = centroids_2d[:, 0], centroids_2d[:, 1]
    b = (np.floor(np.abs(c1)).astype(int) % 9) + 1
    e = (np.floor(np.abs(c2)).astype(int) % 9) + 1
    d = ((b + e - 1) % 9) + 1
    a = ((b + 2 * e - 1) % 9) + 1
    return np.array([b.mean(), e.mean(), d.mean(), a.mean()])


def fit_observer(windows: list[dict]) -> tuple[KMeans, PCA]:
    """
    Fit k-means (k=9) and PCA (2 components) on PSD matrices.
    No seizure labels used — pure unsupervised observer projection.
    """
    mats = []
    for w in windows:
        mat = psd_matrix(w["signals"], w["fs"])  # [n_ch, n_freq]
        mats.append(mat.mean(axis=0))  # collapse channels → [n_freq]
    X = np.array(mats)

    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(X)

    km = KMeans(n_clusters=K_MEANS, random_state=SEED, n_init=10)
    km.fit(X)

    return km, pca


# ── Logistic regression + McFadden R² ────────────────────────────────────────

def _log_likelihood(lr: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(lr.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))


def _null_ll(y: np.ndarray) -> float:
    p = y.mean()
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(y.sum() * np.log(p) + (1 - y).sum() * np.log(1 - p))


def mcfadden_r2(ll_model: float, ll_null: float) -> float:
    return 1.0 - ll_model / ll_null if ll_null != 0 else 0.0


def analyse_patient(pid: str, patient_dir: Path) -> dict | None:
    print(f"\n  [{pid}] Extracting windows...")
    windows = extract_windows(patient_dir)
    if not windows:
        print(f"  [{pid}] No data — skip")
        return None

    n_sei = sum(1 for w in windows if w["label"] == 1)
    n_bas = sum(1 for w in windows if w["label"] == 0)
    print(f"  [{pid}] {n_sei} seizure + {n_bas} baseline windows")

    if n_sei < 4 or n_bas < 4:
        print(f"  [{pid}] Insufficient data — skip")
        return None

    # Observer projection — fit on ALL windows, no labels used
    print(f"  [{pid}] Fitting observer (k=9 k-means + PCA)...")
    km, pca = fit_observer(windows)

    # Build feature matrix
    labels = np.array([w["label"] for w in windows])
    delta_feats = np.array([delta_power(w["signals"], w["fs"]) for w in windows])
    qa_feats = np.array([window_to_qa_tuple(psd_matrix(w["signals"], w["fs"]), km, pca)
                         for w in windows])

    # 70/30 stratified split
    idx = np.arange(len(windows))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC, stratify=labels,
                                      random_state=SEED)

    y_tr, y_te = labels[tr_idx], labels[te_idx]
    d_tr, d_te = delta_feats[tr_idx].reshape(-1, 1), delta_feats[te_idx].reshape(-1, 1)
    q_tr, q_te = qa_feats[tr_idx], qa_feats[te_idx]

    # Baseline model: delta only
    lr_base = LogisticRegression(penalty="l2", random_state=SEED, max_iter=1000)
    lr_base.fit(d_tr, y_tr)
    ll_base = _log_likelihood(lr_base, d_te, y_te)
    ll_null = _null_ll(y_te)
    r2_base = mcfadden_r2(ll_base, ll_null)

    # Augmented model: delta + QA 4-tuple
    X_tr_aug = np.hstack([d_tr, q_tr])
    X_te_aug = np.hstack([d_te, q_te])
    lr_aug = LogisticRegression(penalty="l2", random_state=SEED, max_iter=1000)
    lr_aug.fit(X_tr_aug, y_tr)
    ll_aug = _log_likelihood(lr_aug, X_te_aug, y_te)
    r2_aug = mcfadden_r2(ll_aug, ll_null)

    delta_r2 = r2_aug - r2_base

    # LR test: χ²(df=4) since QA adds 4 features
    lr_stat = max(0.0, 2.0 * (ll_aug - ll_base))
    p_val = float(chi2.sf(lr_stat, df=4))

    print(f"  [{pid}] ΔR²={delta_r2:+.4f}  p={p_val:.4e}  "
          f"R²_base={r2_base:.4f}  R²_aug={r2_aug:.4f}")

    return {
        "patient": pid,
        "n_sei": int(n_sei),
        "n_bas": int(n_bas),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "r2_base": float(r2_base),
        "r2_aug": float(r2_aug),
        "delta_r2": float(delta_r2),
        "lr_stat": float(lr_stat),
        "p_val": float(p_val),
    }


def fishers_method(p_values: list[float]) -> tuple[float, float]:
    p_clip = [max(1e-15, p) for p in p_values]
    stat = -2.0 * sum(np.log(p) for p in p_clip)
    df = 2 * len(p_values)
    return float(stat), float(chi2.sf(stat, df))


# ── Downloader ────────────────────────────────────────────────────────────────

def _parse_summary_files(summary_path: Path) -> dict:
    seizure_files, all_files = [], []
    cur, n_sei = None, 0
    with open(summary_path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m:
                cur = m.group(1).lower()
                n_sei = 0
                all_files.append(cur)
                continue
            m = re.match(r"Number of Seizures in File:\s+(\d+)", line, re.I)
            if m and cur:
                n_sei = int(m.group(1))
                if n_sei > 0:
                    seizure_files.append(cur)
    return {"seizure_files": seizure_files, "all_files": all_files}


def download_patient(pid: str) -> bool:
    """Download summary + seizure EDFs for one patient. Returns True if ready."""
    from urllib.request import urlretrieve
    from urllib.error import URLError, HTTPError

    pdir = CHBMIT_ROOT / pid
    pdir.mkdir(parents=True, exist_ok=True)

    summary_name = f"{pid}-summary.txt"
    summary_url = f"{BASE_URL}/{pid}/{summary_name}"
    summary_path = pdir / summary_name

    if not summary_path.exists():
        print(f"  [{pid}] Downloading summary...")
        try:
            urlretrieve(summary_url, summary_path)
        except (URLError, HTTPError) as e:
            print(f"  [{pid}] Summary download failed: {e}")
            return False

    info = _parse_summary_files(summary_path)
    if not info["seizure_files"]:
        print(f"  [{pid}] No seizure files in summary — skip")
        return False

    # Pick interictal files: 2 no-seizure EDFs spread across recording
    no_sei = [f for f in info["all_files"] if f not in info["seizure_files"]]
    step = max(1, len(no_sei) // 3)
    interictal = [no_sei[i] for i in range(step, len(no_sei), step)][:2]

    files_to_get = list(dict.fromkeys(info["seizure_files"] + interictal))
    print(f"  [{pid}] Need {len(files_to_get)} EDFs "
          f"({len(info['seizure_files'])} seizure + {len(interictal)} interictal)")

    for fname in files_to_get:
        dest = pdir / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"    [EXISTS] {fname}")
            continue
        url = f"{BASE_URL}/{pid}/{fname}"
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        print(f"    Downloading {fname} ... ", end="", flush=True)
        try:
            urlretrieve(url, tmp)
            tmp.rename(dest)
            print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")
        except (URLError, HTTPError) as e:
            print(f"FAILED: {e}")
            if tmp.exists():
                tmp.unlink()

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CHB-MIT Held-Out Replication — Observer 3 (k=9 topographic)")
    print("Pre-registered protocol: k=9, seed=42, 70/30 split, L2 LR")
    print("Success criterion: mean ΔR² > 0.05, ≥ 7/13 patients p < 0.05")
    print("=" * 70)
    print(f"\nHeld-out patients ({len(HELD_OUT)}): {', '.join(HELD_OUT)}")
    print()

    # Phase 1: download
    print("Phase 1: downloading held-out patient data")
    print("-" * 70)
    for pid in HELD_OUT:
        download_patient(pid)

    # Phase 2: analyse
    print("\nPhase 2: running analysis")
    print("-" * 70)
    results = []
    for pid in HELD_OUT:
        pdir = CHBMIT_ROOT / pid
        if not pdir.exists():
            print(f"  [{pid}] directory missing — skip")
            continue
        res = analyse_patient(pid, pdir)
        if res is not None:
            results.append(res)

    if not results:
        print("\nNo usable patients — check data download.")
        return

    # Phase 3: aggregate
    print("\n" + "=" * 70)
    print("PER-PATIENT RESULTS")
    print(f"  {'Patient':<8}  {'N_sei':>5}  {'N_bas':>5}  "
          f"{'R²_base':>8}  {'R²_aug':>7}  {'ΔR²':>8}  {'p':>10}  {'sig':>4}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*4}")
    for r in results:
        sig = "*" if r["p_val"] < 0.05 else "ns"
        print(f"  {r['patient']:<8}  {r['n_sei']:>5}  {r['n_bas']:>5}  "
              f"  {r['r2_base']:>7.4f}  {r['r2_aug']:>7.4f}  {r['delta_r2']:>+8.4f}"
              f"  {r['p_val']:>10.3e}  {sig:>4}")

    n = len(results)
    mean_dr2 = float(np.mean([r["delta_r2"] for r in results]))
    se_dr2 = float(np.std([r["delta_r2"] for r in results]) / np.sqrt(n))
    n_sig = sum(1 for r in results if r["p_val"] < 0.05)
    n_pos = sum(1 for r in results if r["delta_r2"] > 0)
    chi2_fish, p_fish = fishers_method([r["p_val"] for r in results])

    print(f"\n  Mean ΔR²: {mean_dr2:+.4f} ± {se_dr2:.4f} (SE)")
    print(f"  Positive ΔR²: {n_pos}/{n}")
    print(f"  Significant (p<0.05): {n_sig}/{n}")
    print(f"  Fisher: χ²={chi2_fish:.2f}, df={2*n}, p={p_fish:.3e}")

    print("\n" + "=" * 70)
    print("REPLICATION VERDICT")
    criterion_dr2 = mean_dr2 > 0.05
    criterion_sig = n_sig >= 7
    success = criterion_dr2 and criterion_sig
    print(f"  mean ΔR² > 0.05: {'PASS' if criterion_dr2 else 'FAIL'} "
          f"(actual: {mean_dr2:+.4f})")
    print(f"  ≥ 7/13 significant: {'PASS' if criterion_sig else 'FAIL'} "
          f"(actual: {n_sig}/{n})")
    print(f"\n  OVERALL: {'REPLICATION SUCCESS' if success else 'REPLICATION FAIL'}")
    print("=" * 70)

    # Save
    out = {
        "protocol": {
            "k_means": K_MEANS,
            "seed": SEED,
            "test_frac": TEST_FRAC,
            "penalty": "l2",
            "outcome": "McFadden_pseudo_R2",
            "lr_test_df": 4,
            "success_criterion": "mean_delta_r2 > 0.05 AND n_significant >= 7",
        },
        "patients": results,
        "aggregate": {
            "n_patients": n,
            "mean_delta_r2": mean_dr2,
            "se_delta_r2": se_dr2,
            "n_positive": n_pos,
            "n_significant": n_sig,
            "fishers_chi2": chi2_fish,
            "fishers_df": 2 * n,
            "fishers_p": p_fish,
            "criterion_delta_r2_pass": criterion_dr2,
            "criterion_n_sig_pass": criterion_sig,
            "replication_success": success,
        }
    }
    out_path = Path("results/eeg_replication_heldout.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
