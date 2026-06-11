#!/usr/bin/env python3
QA_COMPLIANCE = "EEG seizure: deep structural analysis of QA orbit patterns; integer orbit counts; observer projections only"
"""
eeg_structure_analysis.py — Deep structural analysis of QA orbit patterns in EEG

Goes beyond ΔR² to understand WHERE in QA state space the ictal/interictal
distinction lives.

Analyses:
  1. Orbit family fractions per window (Cosmos/Satellite/Singularity)
     — instead of averaging the 4-tuple, count which families each centroid lands in
  2. (b,e) joint distribution heatmap: ictal vs interictal 9×9 grid
  3. Pre-ictal dynamics: interictal → pre-ictal (30-120s before) → ictal trajectory
  4. Cross-patient orbit consistency

Runs on whatever patients are already downloaded.
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu
import pyedflib

warnings.filterwarnings("ignore")

CHBMIT_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
WINDOW_SEC = 10.0
BUFFER_SEC = 300.0
PRE_ICTAL_SEC = 120.0   # pre-ictal window: 30–120s before seizure onset
K_MEANS = 9
SEED = 42
N_CHANNELS = 23


# ── QA orbit classification ────────────────────────────────────────────────────

def _build_orbit_map(m: int = 9) -> dict[tuple, str]:
    """
    Classify every (b,e) ∈ {1..m}² by orbit family under T(b,e)=(e,(b+e-1)%m+1).
    Returns dict {(b,e): 'Cosmos'|'Satellite'|'Singularity'}.
    """
    def orbit_period(b0, e0):
        b, e = b0, e0
        for k in range(1, 100):
            b, e = e, (b + e - 1) % m + 1
            if b == b0 and e == e0:
                return k
        return -1

    families = {}
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            p = orbit_period(b, e)
            if p == 1:
                families[(b, e)] = "Singularity"
            elif p == 8:
                families[(b, e)] = "Satellite"
            else:
                families[(b, e)] = "Cosmos"
    return families

ORBIT_MAP = _build_orbit_map(9)

# Verify counts
from collections import Counter
_counts = Counter(ORBIT_MAP.values())
assert _counts["Singularity"] == 1, f"Expected 1 Singularity, got {_counts['Singularity']}"
assert _counts["Satellite"] == 8, f"Expected 8 Satellite, got {_counts['Satellite']}"
assert _counts["Cosmos"] == 72, f"Expected 72 Cosmos, got {_counts['Cosmos']}"


def centroids_to_orbit_fracs(centroids_2d: np.ndarray) -> dict:
    """
    Map k centroids (already PCA-projected to 2D) to (b,e) pairs → orbit families.
    Returns {family: fraction, 'b_vals': list, 'e_vals': list}.
    """
    c1, c2 = centroids_2d[:, 0], centroids_2d[:, 1]
    b_arr = (np.floor(np.abs(c1)).astype(int) % 9) + 1
    e_arr = (np.floor(np.abs(c2)).astype(int) % 9) + 1

    families = [ORBIT_MAP[(b, e)] for b, e in zip(b_arr.tolist(), e_arr.tolist())]
    k = len(families)
    return {
        "Cosmos":       families.count("Cosmos") / k,
        "Satellite":    families.count("Satellite") / k,
        "Singularity":  families.count("Singularity") / k,
        "b_vals":       b_arr.tolist(),
        "e_vals":       e_arr.tolist(),
        "pairs":        list(zip(b_arr.tolist(), e_arr.tolist())),
    }


# ── EDF + annotation loading (same as replication script) ────────────────────

def read_edf(path: Path) -> tuple[np.ndarray, float]:
    with pyedflib.EdfReader(str(path)) as f:
        n_ch = f.signals_in_file
        fs = f.getSampleFrequency(0)
        n_use = min(n_ch, N_CHANNELS)
        n_samp = f.getNSamples()[0]
        signals = np.zeros((n_use, n_samp), dtype=np.float32)
        for i in range(n_use):
            signals[i] = f.readSignal(i)
    if n_use < N_CHANNELS:
        signals = np.pad(signals, ((0, N_CHANNELS - n_use), (0, 0)))
    return signals, float(fs)


def parse_summary(path: Path) -> dict[str, list[dict]]:
    ann: dict[str, list[dict]] = defaultdict(list)
    cur = None
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


def psd_mean(signals: np.ndarray, fs: int) -> np.ndarray:
    """Mean PSD across channels → [n_freq]."""
    nperseg = min(2 * fs, signals.shape[1])
    rows = []
    for ch in range(signals.shape[0]):
        _, Pxx = welch(signals[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        rows.append(Pxx)
    return np.mean(rows, axis=0)


def delta_power(signals: np.ndarray, fs: int) -> float:
    nperseg = min(2 * fs, signals.shape[1])
    total = 0.0
    for ch in range(signals.shape[0]):
        freqs, Pxx = welch(signals[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs >= 1.0) & (freqs <= 4.0)
        total += float(np.mean(Pxx[mask])) if mask.any() else 0.0
    return total / signals.shape[0]


# ── Per-patient loader: 3 classes (interictal / pre-ictal / ictal) ────────────

def load_patient_3class(patient_dir: Path) -> list[dict]:
    """
    Returns windows with label: 'ictal' | 'pre_ictal' | 'interictal'.
    pre_ictal = 30–120s before seizure onset.
    """
    summaries = list(patient_dir.glob("*-summary.txt"))
    if not summaries:
        return []
    ann = parse_summary(summaries[0])
    if not ann:
        return []

    windows = []
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

        fs = int(round(fs))
        win = int(WINDOW_SEC * fs)
        total_s = sig.shape[1] / fs

        for s in seqs:
            onset, offset = float(s["start_s"]), float(s["end_s"])

            # Ictal windows
            n_ic = max(1, int((offset - onset) // WINDOW_SEC))
            for i in range(n_ic):
                t0 = onset + i * WINDOW_SEC
                if t0 + WINDOW_SEC > total_s:
                    break
                s0, s1 = int(t0 * fs), int((t0 + WINDOW_SEC) * fs)
                windows.append({
                    "label": "ictal",
                    "signals": sig[:, s0:s1],
                    "fs": fs,
                    "t_start": t0,
                    "t_onset": onset,
                    "source": fname,
                })

            # Pre-ictal: 30–120s before onset
            for t0 in np.arange(max(0, onset - PRE_ICTAL_SEC),
                                  max(0, onset - 30.0),
                                  WINDOW_SEC):
                if t0 + WINDOW_SEC > total_s:
                    break
                s0, s1 = int(t0 * fs), int((t0 + WINDOW_SEC) * fs)
                windows.append({
                    "label": "pre_ictal",
                    "signals": sig[:, s0:s1],
                    "fs": fs,
                    "t_start": t0,
                    "t_onset": onset,
                    "source": fname,
                })

        # Interictal: >300s from any seizure
        ivs = all_intervals[fname]
        earliest = min(iv[0] for iv in ivs)
        latest = max(iv[1] for iv in ivs)

        pos = WINDOW_SEC
        added = 0
        while pos + WINDOW_SEC <= earliest - BUFFER_SEC and added < 8:
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            windows.append({
                "label": "interictal",
                "signals": sig[:, s0:s1],
                "fs": fs,
                "t_start": pos,
                "source": fname,
            })
            added += 1
            pos += WINDOW_SEC * 3

        pos = latest + BUFFER_SEC
        while pos + WINDOW_SEC <= total_s and added < 12:
            s0, s1 = int(pos * fs), int((pos + WINDOW_SEC) * fs)
            windows.append({
                "label": "interictal",
                "signals": sig[:, s0:s1],
                "fs": fs,
                "t_start": pos,
                "source": fname,
            })
            added += 1
            pos += WINDOW_SEC * 3

    return windows


# ── Per-patient orbit analysis ────────────────────────────────────────────────

def analyse_patient(pid: str, patient_dir: Path) -> dict | None:
    print(f"\n  [{pid}] Loading 3-class windows...")
    windows = load_patient_3class(patient_dir)
    if not windows:
        return None

    counts = Counter(w["label"] for w in windows)
    print(f"  [{pid}] {counts}")
    if counts.get("ictal", 0) < 4:
        print(f"  [{pid}] Too few ictal windows — skip")
        return None

    # Fit PCA + k-means on all windows (no labels)
    print(f"  [{pid}] Fitting observer...")
    psds = np.array([psd_mean(w["signals"], w["fs"]) for w in windows])
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(psds)
    km = KMeans(n_clusters=K_MEANS, random_state=SEED, n_init=10)
    km.fit(psds)

    # Project centroids once (fixed for all windows)
    centroids_2d = pca.transform(km.cluster_centers_)  # [9, 2]
    orbit_fracs = centroids_to_orbit_fracs(centroids_2d)

    # Per-window features
    records = []
    for w in windows:
        psd = psd_mean(w["signals"], w["fs"])
        # Each window gets the centroid orbit fracs (fixed per patient)
        # PLUS: which cluster this window belongs to
        cluster_id = int(km.predict(psd.reshape(1, -1))[0])
        cluster_psd = km.cluster_centers_[cluster_id]
        c1c2 = pca.transform(cluster_psd.reshape(1, -1))[0]
        b = int(np.floor(np.abs(c1c2[0])) % 9) + 1
        e = int(np.floor(np.abs(c1c2[1])) % 9) + 1
        family = ORBIT_MAP[(b, e)]

        records.append({
            "label":     w["label"],
            "cluster":   cluster_id,
            "b": b, "e": e,
            "family":    family,
            "delta":     delta_power(w["signals"], w["fs"]),
            "cosmos_frac":      orbit_fracs["Cosmos"],
            "satellite_frac":   orbit_fracs["Satellite"],
            "singularity_frac": orbit_fracs["Singularity"],
            "source": w.get("source", ""),
        })

    # ── Analysis 1: Orbit family enrichment per label ─────────────────────────
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    enrichment = {}
    for family in ["Cosmos", "Satellite", "Singularity"]:
        for label in ["ictal", "pre_ictal", "interictal"]:
            grp = by_label[label]
            if not grp:
                continue
            frac = sum(1 for r in grp if r["family"] == family) / len(grp)
            enrichment[f"{family}_{label}"] = round(frac, 4)

    # ── Analysis 2: (b,e) heatmap — ictal vs interictal ──────────────────────
    heatmap_ictal = np.zeros((9, 9))
    heatmap_inter = np.zeros((9, 9))
    for r in records:
        b, e = r["b"] - 1, r["e"] - 1  # 0-indexed
        if r["label"] == "ictal":
            heatmap_ictal[b, e] += 1
        elif r["label"] == "interictal":
            heatmap_inter[b, e] += 1

    # Normalize to fractions
    if heatmap_ictal.sum() > 0:
        heatmap_ictal /= heatmap_ictal.sum()
    if heatmap_inter.sum() > 0:
        heatmap_inter /= heatmap_inter.sum()
    heatmap_diff = heatmap_ictal - heatmap_inter  # positive = enriched in ictal

    # ── Analysis 3: Family fractions — statistical tests ──────────────────────
    tests = {}
    ictal_families  = [r["family"] for r in by_label["ictal"]]
    inter_families  = [r["family"] for r in by_label["interictal"]]
    preic_families  = [r["family"] for r in by_label.get("pre_ictal", [])]

    for family in ["Cosmos", "Satellite", "Singularity"]:
        # Delta power as binary feature: does this family have higher/lower delta?
        ic_delta  = [r["delta"] for r in by_label["ictal"] if r["family"] == family]
        int_delta = [r["delta"] for r in by_label["interictal"] if r["family"] == family]

        # Fraction test: is this family over/under-represented in ictal?
        n_ic_fam  = ictal_families.count(family)
        n_int_fam = inter_families.count(family)
        n_ic_tot  = len(ictal_families)
        n_int_tot = len(inter_families)

        if n_ic_tot > 0 and n_int_tot > 0:
            # Chi-squared: family vs not-family × ictal vs interictal
            ct = np.array([[n_ic_fam, n_ic_tot - n_ic_fam],
                           [n_int_fam, n_int_tot - n_int_fam]])
            if ct.min() >= 0 and ct.sum() > 0:
                try:
                    chi2_stat, p_chi2, _, _ = chi2_contingency(ct)
                except Exception:
                    chi2_stat, p_chi2 = np.nan, 1.0
            else:
                chi2_stat, p_chi2 = np.nan, 1.0

            tests[family] = {
                "frac_ictal":  round(n_ic_fam / n_ic_tot, 4),
                "frac_inter":  round(n_int_fam / n_int_tot, 4),
                "chi2":        round(float(chi2_stat), 4) if not np.isnan(chi2_stat) else None,
                "p":           round(float(p_chi2), 6),
                "direction":   "enriched" if n_ic_fam / max(n_ic_tot, 1) > n_int_fam / max(n_int_tot, 1) else "depleted",
            }

    # ── Analysis 4: Pre-ictal trajectory ──────────────────────────────────────
    preictal_trajectory = {}
    if by_label.get("pre_ictal"):
        for family in ["Cosmos", "Satellite", "Singularity"]:
            ic_frac   = enrichment.get(f"{family}_ictal", None)
            pre_frac  = enrichment.get(f"{family}_pre_ictal", None)
            int_frac  = enrichment.get(f"{family}_interictal", None)
            preictal_trajectory[family] = {
                "interictal": int_frac,
                "pre_ictal":  pre_frac,
                "ictal":      ic_frac,
            }

    # ── Analysis 5: Delta power vs orbit family ────────────────────────────────
    delta_by_family = {}
    for family in ["Cosmos", "Satellite", "Singularity"]:
        ic_d  = [r["delta"] for r in by_label["ictal"]       if r["family"] == family]
        int_d = [r["delta"] for r in by_label["interictal"]  if r["family"] == family]
        delta_by_family[family] = {
            "n_ictal": len(ic_d),
            "n_inter": len(int_d),
            "mean_delta_ictal":  round(float(np.mean(ic_d)),  4) if ic_d  else None,
            "mean_delta_inter":  round(float(np.mean(int_d)), 4) if int_d else None,
        }

    # Top (b,e) cells enriched in ictal
    flat_diff = [(float(heatmap_diff[b, e]), b + 1, e + 1,
                  ORBIT_MAP[(b + 1, e + 1)])
                 for b in range(9) for e in range(9)]
    flat_diff.sort(reverse=True)
    top_ictal_cells  = [{"b": b, "e": e, "diff": d, "family": fam}
                        for d, b, e, fam in flat_diff[:10]]
    top_inter_cells  = [{"b": b, "e": e, "diff": -d, "family": fam}
                        for d, b, e, fam in flat_diff[-10:]]

    print(f"  [{pid}] Orbit family tests:")
    for fam, t in tests.items():
        print(f"    {fam:12s} ictal={t['frac_ictal']:.3f}  inter={t['frac_inter']:.3f}"
              f"  χ²={t['chi2']}  p={t['p']:.4f}  ({t['direction']})")

    return {
        "patient":              pid,
        "n_ictal":              len(by_label["ictal"]),
        "n_pre_ictal":          len(by_label.get("pre_ictal", [])),
        "n_interictal":         len(by_label["interictal"]),
        "centroid_orbit_fracs": {k: v for k, v in orbit_fracs.items()
                                 if k in ("Cosmos", "Satellite", "Singularity")},
        "window_family_fracs":  enrichment,
        "family_tests":         tests,
        "preictal_trajectory":  preictal_trajectory,
        "delta_by_family":      delta_by_family,
        "top_ictal_cells":      top_ictal_cells,
        "top_inter_cells":      top_inter_cells,
        "heatmap_diff":         heatmap_diff.tolist(),
    }


# ── Cross-patient aggregation ─────────────────────────────────────────────────

def summarise(results: list[dict]) -> dict:
    """Aggregate cross-patient patterns."""
    families = ["Cosmos", "Satellite", "Singularity"]

    # Consistent direction test: how many patients show enrichment in each family?
    direction_counts = {f: {"enriched": 0, "depleted": 0} for f in families}
    p_values_by_family = {f: [] for f in families}

    for r in results:
        for fam in families:
            t = r["family_tests"].get(fam, {})
            if t:
                d = t["direction"]
                direction_counts[fam][d] += 1
                p_values_by_family[fam].append(t["p"])

    # Fisher's method per family
    def fishers(ps):
        from scipy.stats import chi2
        ps = [max(1e-15, p) for p in ps]
        stat = -2 * sum(np.log(p) for p in ps)
        return float(stat), float(chi2.sf(stat, 2 * len(ps)))

    cross_patient = {}
    for fam in families:
        ps = p_values_by_family[fam]
        if ps:
            chi2_stat, p_fish = fishers(ps)
            cross_patient[fam] = {
                "n_enriched":  direction_counts[fam]["enriched"],
                "n_depleted":  direction_counts[fam]["depleted"],
                "n_patients":  len(ps),
                "fisher_chi2": round(chi2_stat, 3),
                "fisher_p":    round(p_fish, 6),
            }

    # Mean fractions ictal vs interictal
    mean_fracs = {}
    for fam in families:
        ic_fracs  = [r["window_family_fracs"].get(f"{fam}_ictal",  0) for r in results]
        int_fracs = [r["window_family_fracs"].get(f"{fam}_interictal", 0) for r in results]
        mean_fracs[fam] = {
            "mean_ictal": round(float(np.mean(ic_fracs)),  4),
            "mean_inter": round(float(np.mean(int_fracs)), 4),
            "delta":      round(float(np.mean(ic_fracs)) - float(np.mean(int_fracs)), 4),
        }

    # Aggregate (b,e) heatmap
    agg_diff = np.zeros((9, 9))
    for r in results:
        agg_diff += np.array(r["heatmap_diff"])
    agg_diff /= len(results)
    flat = [(float(agg_diff[b, e]), b + 1, e + 1, ORBIT_MAP[(b + 1, e + 1)])
            for b in range(9) for e in range(9)]
    flat.sort(reverse=True)
    top_cells = [{"b": b, "e": e, "mean_diff": round(d, 5), "family": fam}
                 for d, b, e, fam in flat[:15]]

    return {
        "n_patients":    len(results),
        "cross_patient": cross_patient,
        "mean_fracs":    mean_fracs,
        "top_ictal_cells": top_cells,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EEG Structural Analysis — Orbit Family Enrichment")
    print("Three-class: interictal / pre-ictal / ictal")
    print("=" * 70)

    patient_dirs = sorted(CHBMIT_ROOT.glob("chb*/"))
    # Only fully downloaded (no .tmp files)
    ready = []
    for pd in patient_dirs:
        if list(pd.glob("*.tmp")):
            print(f"  {pd.name}: still downloading — skip")
            continue
        if list(pd.glob("*-summary.txt")):
            ready.append(pd)

    print(f"\nReady patients: {[p.name for p in ready]}")

    results = []
    for pdir in ready:
        pid = pdir.name
        res = analyse_patient(pid, pdir)
        if res:
            results.append(res)

    if not results:
        print("\nNo results — check data.")
        return

    # Cross-patient summary
    summary = summarise(results)

    print("\n" + "=" * 70)
    print("CROSS-PATIENT ORBIT FAMILY ENRICHMENT")
    print(f"  {'Family':12s}  {'Mean_ictal':>10}  {'Mean_inter':>10}  "
          f"{'Delta':>7}  {'N_enrich':>8}  {'Fisher_p':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*10}")
    for fam in ["Cosmos", "Satellite", "Singularity"]:
        mf = summary["mean_fracs"][fam]
        cp = summary["cross_patient"].get(fam, {})
        n_enr = cp.get("n_enriched", "?")
        n_tot = cp.get("n_patients", "?")
        fp    = cp.get("fisher_p", None)
        print(f"  {fam:12s}  {mf['mean_ictal']:>10.4f}  {mf['mean_inter']:>10.4f}  "
              f"  {mf['delta']:>+6.4f}  {str(n_enr)+'/'+str(n_tot):>8}  "
              f"  {fp:.4e}" if fp is not None else
              f"  {fam:12s}  {mf['mean_ictal']:>10.4f}  {mf['mean_inter']:>10.4f}  "
              f"  {mf['delta']:>+6.4f}")

    print()
    print("TOP (b,e) CELLS ENRICHED IN ICTAL (averaged across patients):")
    print(f"  {'b':>3}  {'e':>3}  {'mean_diff':>10}  {'family':12s}")
    for cell in summary["top_ictal_cells"][:10]:
        print(f"  {cell['b']:>3}  {cell['e']:>3}  {cell['mean_diff']:>+10.5f}  {cell['family']}")

    print()
    print("PRE-ICTAL TRAJECTORY (mean fraction across patients):")
    for fam in ["Cosmos", "Satellite", "Singularity"]:
        traj_vals = []
        for r in results:
            t = r["preictal_trajectory"].get(fam, {})
            traj_vals.append(t)
        inter_m = np.mean([t.get("interictal") or 0 for t in traj_vals])
        pre_m   = np.mean([t.get("pre_ictal")  or 0 for t in traj_vals])
        ic_m    = np.mean([t.get("ictal")       or 0 for t in traj_vals])
        print(f"  {fam:12s}  interictal={inter_m:.3f}  →  pre_ictal={pre_m:.3f}"
              f"  →  ictal={ic_m:.3f}  (Δ={ic_m-inter_m:+.3f})")

    # Save
    out = {
        "patients":  results,
        "summary":   summary,
    }
    out_path = Path("results/eeg_structure_analysis.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
