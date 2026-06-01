#!/usr/bin/env python3
"""
eeg_transition_orbit.py — Correct implementation of the transition-based orbit observer

The original ΔR²=+0.21 result uses:
  - k=4 topographic k-means (channel RMS amplitude, not PSD)
  - 1-second sub-windows (step=0.5s, ~19 transitions per 10s window)
  - Transition encoding: b = MICROSTATE_STATES[state_t][0],
                         e = MICROSTATE_STATES[state_{t+1}][1]
  - m=24 orbit families: Singularity (D→D), Satellite (A→B, A→D, D→B), Cosmos (rest)

This script adds baseline-relative deviation features on top of the original.

Four models compared:
  A: delta only
  B: delta + [singularity_frac, cosmos_frac]     (original protocol, from transitions)
  C: delta + deviation from interictal baseline   (baseline-relative)
  D: delta + transition fracs + deviation          (combined)
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pyedflib

warnings.filterwarnings("ignore")

CHBMIT_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
WINDOW_SEC  = 10.0
SUB_SEC     = 1.0
BUFFER_SEC  = 300.0
K_MEANS     = 4
SEED        = 42
TEST_FRAC   = 0.30
N_CHANNELS  = 23

# State alphabet — fixed (b,e) pairs at m=24
# Chosen so that A→B, A→D, D→B transitions are Satellite; D→D is Singularity.
MICROSTATE_STATES = {
    "A_frontal":   (8,  3),
    "B_occipital": (5, 16),
    "C_right":     (11, 19),
    "D_baseline":  (24, 24),
}


# ── Orbit family at m=24 ──────────────────────────────────────────────────────

def _orbit_period(b0, e0, m=24):
    b, e = b0, e0
    for k in range(1, 300):
        b, e = e, (b + e - 1) % m + 1
        if b == b0 and e == e0:
            return k
    return -1

def orbit_family_m24(b, e):
    p = _orbit_period(b, e)
    return "Singularity" if p == 1 else "Satellite" if p == 8 else "Cosmos"

# Pre-compute transition table
TRANSITION_TABLE = {
    (s1, s2): orbit_family_m24(MICROSTATE_STATES[s1][0], MICROSTATE_STATES[s2][1])
    for s1 in MICROSTATE_STATES for s2 in MICROSTATE_STATES
}

# Sanity check
assert sum(1 for f in TRANSITION_TABLE.values() if f == "Singularity") == 1
assert sum(1 for f in TRANSITION_TABLE.values() if f == "Satellite") == 3


# ── EDF loading ───────────────────────────────────────────────────────────────

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


# ── Topographic feature vector ────────────────────────────────────────────────

def topo_fv(multi_ch: np.ndarray) -> np.ndarray:
    """RMS per channel, L2-normalised. Shape: (n_ch,)."""
    rms  = np.sqrt(np.mean(multi_ch.astype(np.float64) ** 2, axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms


# ── Per-window transition orbit fractions ─────────────────────────────────────

def window_orbit_fracs(multi_ch: np.ndarray, fs: int,
                       km: KMeans, state_order: list[str]) -> dict:
    """
    Classify 10s window into 1s sub-windows (step=0.5s),
    compute transition orbit fracs from consecutive state pairs.
    Returns {'singularity': float, 'satellite': float, 'cosmos': float, 'n_transitions': int}.
    """
    sub_n   = int(SUB_SEC * fs)
    step    = max(1, sub_n // 2)
    total   = multi_ch.shape[1]
    states  = []

    for s in range(0, total - sub_n + 1, step):
        fv  = topo_fv(multi_ch[:, s: s + sub_n]).reshape(1, -1)
        cid = int(km.predict(fv)[0])
        states.append(state_order[cid])

    if len(states) < 2:
        return {"singularity": 0.0, "satellite": 0.0, "cosmos": 1.0, "n_transitions": 0}

    orbits = [TRANSITION_TABLE[(states[i], states[i + 1])]
              for i in range(len(states) - 1)]
    n = len(orbits)
    return {
        "singularity":    orbits.count("Singularity") / n,
        "satellite":      orbits.count("Satellite")   / n,
        "cosmos":         orbits.count("Cosmos")       / n,
        "n_transitions":  n,
    }


def delta_power(sig: np.ndarray, fs: int) -> float:
    nperseg = min(2 * fs, sig.shape[1])
    total = 0.0
    for ch in range(sig.shape[0]):
        freqs, Pxx = welch(sig[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs >= 1.0) & (freqs <= 4.0)
        total += float(np.mean(Pxx[mask])) if mask.any() else 0.0
    return total / sig.shape[0]


# ── Window extraction ─────────────────────────────────────────────────────────

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

        ivs  = all_ivs[fname]
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


# ── K-means fit + state assignment ────────────────────────────────────────────

def fit_topographic_km(windows: list[dict]) -> tuple[KMeans, list[str]]:
    """
    Fit k=4 k-means on ALL sub-windows of ALL windows (no labels).
    Assign clusters to states by centroid RMS rank:
      lowest → D_baseline, ..., highest → A_frontal.
    """
    all_fvs = []
    for w in windows:
        sub_n = int(SUB_SEC * w["fs"])
        step  = max(1, sub_n // 2)
        mc    = w["signals"]
        for s in range(0, mc.shape[1] - sub_n + 1, step):
            all_fvs.append(topo_fv(mc[:, s: s + sub_n]))

    X  = np.array(all_fvs)
    km = KMeans(n_clusters=K_MEANS, random_state=SEED, n_init=10).fit(X)

    # Rank centroids by L2 norm (proxy for overall amplitude)
    norms      = np.linalg.norm(km.cluster_centers_, axis=1)
    rank_order = np.argsort(norms)   # ascending: weakest → strongest
    state_names = ["D_baseline", "C_right", "B_occipital", "A_frontal"]
    # state_order[cluster_id] = state_name
    state_order = [""] * K_MEANS
    for rank, cid in enumerate(rank_order):
        state_order[cid] = state_names[rank]

    return km, state_order


# ── LR helpers ────────────────────────────────────────────────────────────────

def _null_ll(y):
    p = np.clip(y.mean(), 1e-10, 1 - 1e-10)
    return float(y.sum() * np.log(p) + (1 - y).sum() * np.log(1 - p))

def _model_ll(lr, X, y):
    probs = np.clip(lr.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))

def _lr():
    return LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)

def eval_model(X_tr, y_tr, X_te, y_te, ll_null, ll_delta, n_qa):
    model = _lr().fit(X_tr, y_tr)
    ll    = _model_ll(model, X_te, y_te)
    r2    = 1.0 - ll / ll_null if ll_null != 0 else 0.0
    lrs   = max(0.0, 2.0 * (ll - ll_delta))
    p     = float(chi2.sf(lrs, df=n_qa))
    return float(r2), float(p)


# ── Per-patient analysis ──────────────────────────────────────────────────────

def analyse_patient(pid: str, pdir: Path) -> dict | None:
    print(f"\n  [{pid}] Loading windows...")
    windows = extract_windows(pdir)
    if not windows:
        return None

    labels  = np.array([w["label"] for w in windows])
    n_sei   = int((labels == 1).sum())
    n_bas   = int((labels == 0).sum())
    if n_sei < 4 or n_bas < 4:
        print(f"  [{pid}] Insufficient data — skip")
        return None
    print(f"  [{pid}] {n_sei} seizure + {n_bas} baseline windows")

    # Observer: fit k-means on all sub-windows, no labels (Theorem NT)
    print(f"  [{pid}] Fitting k=4 topographic k-means...")
    km, state_order = fit_topographic_km(windows)
    print(f"  [{pid}] Cluster→state: {dict(enumerate(state_order))}")

    # Extract features for every window
    print(f"  [{pid}] Extracting transition orbit features...")
    delta_arr  = np.array([delta_power(w["signals"], w["fs"]) for w in windows])
    orb_arr    = np.array([
        [f["singularity"], f["satellite"], f["cosmos"]]
        for w in windows
        for f in [window_orbit_fracs(w["signals"], w["fs"], km, state_order)]
    ])  # [n, 3]: singularity, satellite, cosmos

    # 70/30 stratified split
    idx = np.arange(len(windows))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC,
                                      stratify=labels, random_state=SEED)
    y_tr, y_te = labels[tr_idx], labels[te_idx]

    d_tr = delta_arr[tr_idx].reshape(-1, 1)
    d_te = delta_arr[te_idx].reshape(-1, 1)
    o_tr = orb_arr[tr_idx]    # [n_tr, 3]
    o_te = orb_arr[te_idx]    # [n_te, 3]

    # Baseline-relative: subtract training-interictal mean
    inter_mask  = (labels[tr_idx] == 0)
    if inter_mask.sum() == 0:
        return None
    baseline    = o_tr[inter_mask].mean(axis=0)  # [3]
    dev_tr      = o_tr - baseline
    dev_te      = o_te - baseline

    ll_null  = _null_ll(y_te)

    # Model A: delta only
    lr_A = _lr().fit(d_tr, y_tr)
    ll_A = _model_ll(lr_A, d_te, y_te)
    r2_A = 1.0 - ll_A / ll_null if ll_null != 0 else 0.0

    # Model B: delta + [singularity_frac, cosmos_frac] — original protocol (2 QA features)
    X_tr_B = np.hstack([d_tr, o_tr[:, [0, 2]]])   # singularity + cosmos
    X_te_B = np.hstack([d_te, o_te[:, [0, 2]]])
    r2_B, p_B = eval_model(X_tr_B, y_tr, X_te_B, y_te, ll_null, ll_A, n_qa=2)

    # Model C: delta + deviation [sing_dev, sat_dev, cos_dev] (3 QA features)
    X_tr_C = np.hstack([d_tr, dev_tr])
    X_te_C = np.hstack([d_te, dev_te])
    r2_C, p_C = eval_model(X_tr_C, y_tr, X_te_C, y_te, ll_null, ll_A, n_qa=3)

    # Model D: delta + orbit fracs + deviation (5 QA features)
    X_tr_D = np.hstack([d_tr, o_tr, dev_tr])
    X_te_D = np.hstack([d_te, o_te, dev_te])
    r2_D, p_D = eval_model(X_tr_D, y_tr, X_te_D, y_te, ll_null, ll_A, n_qa=5)

    # Mean fracs per class
    ic_idx = te_idx[y_te == 1]
    in_idx = te_idx[y_te == 0]
    ic_orb = orb_arr[ic_idx].mean(axis=0) if len(ic_idx) else np.zeros(3)
    in_orb = orb_arr[in_idx].mean(axis=0) if len(in_idx) else np.zeros(3)

    print(f"  [{pid}] Transition fracs — ictal:      sing={ic_orb[0]:.3f}  "
          f"sat={ic_orb[1]:.3f}  cos={ic_orb[2]:.3f}")
    print(f"  [{pid}] Transition fracs — interictal: sing={in_orb[0]:.3f}  "
          f"sat={in_orb[1]:.3f}  cos={in_orb[2]:.3f}")
    print(f"  [{pid}] ΔR²: B(orig)={r2_B-r2_A:+.4f}  "
          f"C(dev)={r2_C-r2_A:+.4f}  D(both)={r2_D-r2_A:+.4f}")

    return {
        "patient":       pid,
        "n_sei":         n_sei,
        "n_bas":         n_bas,
        "baseline_sing": round(float(baseline[0]), 4),
        "baseline_sat":  round(float(baseline[1]), 4),
        "baseline_cos":  round(float(baseline[2]), 4),
        "ic_sing":       round(float(ic_orb[0]), 4),
        "ic_sat":        round(float(ic_orb[1]), 4),
        "ic_cos":        round(float(ic_orb[2]), 4),
        "in_sing":       round(float(in_orb[0]), 4),
        "in_sat":        round(float(in_orb[1]), 4),
        "in_cos":        round(float(in_orb[2]), 4),
        "r2_A":          round(r2_A, 5),
        "delta_r2_B":    round(r2_B - r2_A, 5),
        "delta_r2_C":    round(r2_C - r2_A, 5),
        "delta_r2_D":    round(r2_D - r2_A, 5),
        "p_B":           round(p_B, 6),
        "p_C":           round(p_C, 6),
        "p_D":           round(p_D, 6),
    }


def fishers(ps):
    ps   = [max(1e-15, p) for p in ps]
    stat = -2.0 * sum(np.log(p) for p in ps)
    return float(stat), float(chi2.sf(stat, 2 * len(ps)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 74)
    print("EEG Transition Orbit Analysis — Correct Implementation")
    print("  k=4 topographic k-means, 1s sub-windows, transition (b,e) at m=24")
    print("  B: original (singularity_frac + cosmos_frac)")
    print("  C: baseline-relative deviation (per-patient normalised)")
    print("  D: combined")
    print("=" * 74)

    ready = [pd for pd in sorted(CHBMIT_ROOT.glob("chb*/"))
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
    print("\n" + "=" * 74)
    print("PER-PATIENT RESULTS")
    print(f"  {'Pat':<6}  {'ictal':>18}  {'interictal':>18}  "
          f"{'ΔR²_B':>7}  {'ΔR²_C':>7}  {'ΔR²_D':>7}  {'p_C':>8}")
    print(f"  {'':6}  {'sing|sat|cos':>18}  {'sing|sat|cos':>18}  "
          f"{'orig':>7}  {'dev':>7}  {'both':>7}  {'':8}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*18}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")

    for r in results:
        ic  = f"{r['ic_sing']:.2f}|{r['ic_sat']:.2f}|{r['ic_cos']:.2f}"
        inn = f"{r['in_sing']:.2f}|{r['in_sat']:.2f}|{r['in_cos']:.2f}"
        print(f"  {r['patient']:<6}  {ic:>18}  {inn:>18}  "
              f"  {r['delta_r2_B']:>+6.4f}  {r['delta_r2_C']:>+6.4f}"
              f"  {r['delta_r2_D']:>+6.4f}  {r['p_C']:>8.4f}")

    print()
    for label, key_dr2, key_p in [("B (orig)",  "delta_r2_B", "p_B"),
                                   ("C (dev)",   "delta_r2_C", "p_C"),
                                   ("D (both)",  "delta_r2_D", "p_D")]:
        vals  = [r[key_dr2] for r in results]
        ps    = [r[key_p]   for r in results]
        mean  = float(np.mean(vals))
        se    = float(np.std(vals) / np.sqrt(n))
        n_sig = sum(1 for p in ps if p < 0.05)
        n_pos = sum(1 for v in vals if v > 0)
        chi2_f, p_fish = fishers(ps)
        print(f"  Model {label}:  mean ΔR²={mean:+.4f}±{se:.4f}  "
              f"pos={n_pos}/{n}  sig={n_sig}/{n}  Fisher p={p_fish:.3e}")

    out = {
        "patients": results,
        "aggregate": {m: {
            "mean_delta_r2": round(float(np.mean([r[f"delta_r2_{m}"] for r in results])), 5),
            "n_positive":    sum(1 for r in results if r[f"delta_r2_{m}"] > 0),
            "n_significant": sum(1 for r in results if r[f"p_{m}"] < 0.05),
            "fisher_p":      round(fishers([r[f"p_{m}"] for r in results])[1], 8),
        } for m in ["B", "C", "D"]},
    }
    out_path = Path("results/eeg_transition_orbit.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
