#!/usr/bin/env python3
QA_COMPLIANCE = "EEG seizure: pre-ictal trajectory with transition orbit features; k=4 topographic transitions; integer (b,e) in {1..24}^2"
"""
eeg_preictal_trajectory.py — Pre-ictal trajectory with correct transition orbit features

Same observer as eeg_transition_orbit.py (k=4 channel-RMS, 1s sub-windows,
transition (b,e) at m=24), but with 4 time bins:
  interictal   : >300s from any seizure
  pre3          : 90-120s before onset
  pre2          : 60-90s before onset
  pre1          : 30-60s before onset
  ictal         : within seizure

Per-patient: mean [singularity, satellite, cosmos] fraction in each bin.
Cross-patient: test whether the interictal→ictal gradient is monotonic.
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr, ttest_ind
from sklearn.cluster import KMeans
import pyedflib

warnings.filterwarnings("ignore")

CHBMIT_ROOT = Path("archive/phase_artifacts/phase2_data/eeg/chbmit")
WINDOW_SEC  = 10.0
SUB_SEC     = 1.0
BUFFER_SEC  = 300.0
K_MEANS     = 4
SEED        = 42
N_CHANNELS  = 23

MICROSTATE_STATES = {
    "A_frontal":   (8,  3),
    "B_occipital": (5, 16),
    "C_right":     (11, 19),
    "D_baseline":  (24, 24),
}

# Pre-ictal bins (seconds before onset)
PRE_BINS = [
    ("pre3", 90, 120),
    ("pre2", 60,  90),
    ("pre1", 30,  60),
]
ALL_LABELS = ["interictal", "pre3", "pre2", "pre1", "ictal"]


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

TRANSITION_TABLE = {
    (s1, s2): orbit_family_m24(MICROSTATE_STATES[s1][0], MICROSTATE_STATES[s2][1])
    for s1 in MICROSTATE_STATES for s2 in MICROSTATE_STATES
}


def read_edf(path):
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

def parse_summary(path):
    ann, cur = defaultdict(list), None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m:
                cur = m.group(1).lower(); continue
            if cur is None: continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                ann[cur].append({"start_s": int(m.group(1)), "end_s": None}); continue
            m = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)\s+second", line, re.I)
            if m:
                for a in reversed(ann.get(cur, [])):
                    if a["end_s"] is None:
                        a["end_s"] = int(m.group(1)); break
    return {k: [a for a in v if a["end_s"] is not None] for k, v in ann.items() if v}

def topo_fv(mc):
    rms  = np.sqrt(np.mean(mc.astype(np.float64) ** 2, axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms

def window_orbit_fracs(mc, fs, km, state_order):
    sub_n  = int(SUB_SEC * fs)
    step   = max(1, sub_n // 2)
    states = []
    for s in range(0, mc.shape[1] - sub_n + 1, step):
        cid = int(km.predict(topo_fv(mc[:, s:s+sub_n]).reshape(1,-1))[0])
        states.append(state_order[cid])
    if len(states) < 2:
        return np.array([0.0, 0.0, 1.0])
    orbits = [TRANSITION_TABLE[(states[i], states[i+1])] for i in range(len(states)-1)]
    n = len(orbits)
    return np.array([
        orbits.count("Singularity") / n,
        orbits.count("Satellite")   / n,
        orbits.count("Cosmos")      / n,
    ])


def load_all_windows(patient_dir):
    """Load windows with 5-class labels including pre-ictal bins."""
    summaries = list(patient_dir.glob("*-summary.txt"))
    if not summaries: return []
    ann = parse_summary(summaries[0])
    if not ann: return []

    windows = []
    all_ivs = defaultdict(list)
    for fname, seqs in ann.items():
        for s in seqs:
            all_ivs[fname].append((float(s["start_s"]), float(s["end_s"])))

    for fname, seqs in ann.items():
        edf = patient_dir / fname
        if not edf.exists(): continue
        try:
            sig, fs = read_edf(edf)
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}"); continue

        total_s = sig.shape[1] / fs
        ivs = all_ivs[fname]

        for s in seqs:
            onset, offset = float(s["start_s"]), float(s["end_s"])

            # Ictal
            for i in range(max(1, int((offset - onset) // WINDOW_SEC))):
                t0 = onset + i * WINDOW_SEC
                if t0 + WINDOW_SEC > total_s: break
                s0, s1 = int(t0*fs), int((t0+WINDOW_SEC)*fs)
                windows.append({"label": "ictal", "signals": sig[:,s0:s1], "fs": fs,
                                 "t_onset": onset})

            # Pre-ictal bins
            for (bin_name, lo, hi) in PRE_BINS:
                for t0 in np.arange(onset - hi, onset - lo, WINDOW_SEC):
                    if t0 < 0 or t0 + WINDOW_SEC > total_s: continue
                    # Ensure not inside another seizure
                    in_other = any(
                        not (iv[1] + 30 < t0 or t0 + WINDOW_SEC < iv[0] - 30)
                        for iv in ivs if not (iv[0] == onset)
                    )
                    if in_other: continue
                    s0, s1 = int(t0*fs), int((t0+WINDOW_SEC)*fs)
                    windows.append({"label": bin_name, "signals": sig[:,s0:s1], "fs": fs,
                                    "t_onset": onset})

        # Interictal: >300s from all seizures
        earliest = min(iv[0] for iv in ivs)
        latest   = max(iv[1] for iv in ivs)
        added = 0
        for pos in np.arange(WINDOW_SEC, earliest - BUFFER_SEC, WINDOW_SEC * 3):
            if added >= 6: break
            s0, s1 = int(pos*fs), int((pos+WINDOW_SEC)*fs)
            windows.append({"label": "interictal", "signals": sig[:,s0:s1], "fs": fs})
            added += 1
        for pos in np.arange(latest + BUFFER_SEC, total_s - WINDOW_SEC, WINDOW_SEC * 3):
            if added >= 10: break
            s0, s1 = int(pos*fs), int((pos+WINDOW_SEC)*fs)
            windows.append({"label": "interictal", "signals": sig[:,s0:s1], "fs": fs})
            added += 1

    return windows


def fit_km(windows):
    all_fvs = []
    for w in windows:
        sub_n = int(SUB_SEC * w["fs"])
        step  = max(1, sub_n // 2)
        mc    = w["signals"]
        for s in range(0, mc.shape[1] - sub_n + 1, step):
            all_fvs.append(topo_fv(mc[:, s:s+sub_n]))
    X  = np.array(all_fvs)
    km = KMeans(n_clusters=K_MEANS, random_state=SEED, n_init=10).fit(X)
    norms = np.linalg.norm(km.cluster_centers_, axis=1)
    rank  = np.argsort(norms)
    names = ["D_baseline", "C_right", "B_occipital", "A_frontal"]
    order = [""] * K_MEANS
    for i, cid in enumerate(rank):
        order[cid] = names[i]
    return km, order


def analyse_patient(pid, pdir):
    print(f"  [{pid}] Loading...")
    windows = load_all_windows(pdir)
    if not windows: return None

    from collections import Counter
    cnts = Counter(w["label"] for w in windows)
    print(f"  [{pid}] {dict(cnts)}")
    if cnts.get("ictal", 0) < 4: return None

    km, state_order = fit_km(windows)

    # Compute orbit fracs for every window
    fracs = np.array([window_orbit_fracs(w["signals"], w["fs"], km, state_order)
                      for w in windows])   # [n, 3]: sing, sat, cos
    labels = [w["label"] for w in windows]

    # Mean fracs per bin
    result = {"patient": pid}
    for lbl in ALL_LABELS:
        idx = [i for i,l in enumerate(labels) if l == lbl]
        if idx:
            m = fracs[idx].mean(axis=0)
            result[lbl] = {"sing": round(float(m[0]),4),
                           "sat":  round(float(m[1]),4),
                           "cos":  round(float(m[2]),4),
                           "n":    len(idx)}
        else:
            result[lbl] = None

    # Per-seizure: track each onset separately to get finer trajectory
    # Group pre-ictal windows by t_onset
    by_onset = defaultdict(lambda: defaultdict(list))
    for i, w in enumerate(windows):
        if "t_onset" in w:
            by_onset[w["t_onset"]][w["label"]].append(fracs[i])
    # Interictal: one pool
    inter_idx = [i for i,l in enumerate(labels) if l == "interictal"]
    inter_mean = fracs[inter_idx].mean(axis=0) if inter_idx else np.array([0,0,1])

    # Gradient test for singularity: does it monotonically decline?
    sing_seq = []
    for lbl in ALL_LABELS:
        idx = [i for i,l in enumerate(labels) if l == lbl]
        if idx:
            sing_seq.append(fracs[idx, 0].mean())
        else:
            sing_seq.append(None)

    # Spearman rho across bins where data exists
    valid = [(i, v) for i, v in enumerate(sing_seq) if v is not None]
    if len(valid) >= 3:
        xs, ys = zip(*valid)
        rho, p_rho = spearmanr(xs, ys)
        # Negative rho = monotone decline (correct direction)
        result["spearman_rho"]  = round(float(rho), 4)
        result["spearman_p"]    = round(float(p_rho), 4)
        result["monotone_decline"] = (rho < -0.5)
    else:
        result["spearman_rho"] = None
        result["spearman_p"]   = None
        result["monotone_decline"] = None

    # t-test: interictal vs ictal singularity
    inter_sing = fracs[inter_idx, 0] if inter_idx else np.array([])
    ictal_idx  = [i for i,l in enumerate(labels) if l == "ictal"]
    ictal_sing = fracs[ictal_idx, 0] if ictal_idx else np.array([])
    if len(inter_sing) >= 2 and len(ictal_sing) >= 2:
        t, p = ttest_ind(inter_sing, ictal_sing, equal_var=False)
        result["ttest_t"] = round(float(t), 4)
        result["ttest_p"] = round(float(p), 6)
    else:
        result["ttest_t"] = None
        result["ttest_p"] = None

    result["sing_seq"] = [round(v, 4) if v is not None else None for v in sing_seq]
    result["n_total"]  = len(windows)

    # Print per-patient table
    print(f"  [{pid}] Singularity trajectory:")
    for lbl, v in zip(ALL_LABELS, sing_seq):
        n = cnts.get(lbl, 0)
        bar = "█" * int((v or 0) * 30)
        print(f"    {lbl:12s} ({n:3d}w)  {v:.4f}  {bar}" if v is not None
              else f"    {lbl:12s}  (no data)")
    if result["spearman_rho"] is not None:
        direction = "DECLINE" if result["monotone_decline"] else "no monotone decline"
        print(f"    Spearman ρ={result['spearman_rho']:.3f}  p={result['spearman_p']:.4f}  → {direction}")
    if result["ttest_p"] is not None:
        print(f"    t-test (inter vs ictal): t={result['ttest_t']:.2f}  p={result['ttest_p']:.4f}")

    return result


def main():
    print("=" * 68)
    print("EEG Pre-Ictal Trajectory — Transition Orbit (k=4, m=24)")
    print("Bins: interictal | pre3(90-120s) | pre2(60-90s) | pre1(30-60s) | ictal")
    print("=" * 68 + "\n")

    ready = [pd for pd in sorted(CHBMIT_ROOT.glob("chb*/"))
             if list(pd.glob("*-summary.txt")) and not list(pd.glob("*.tmp"))]
    print(f"Patients: {[p.name for p in ready]}\n")

    results = []
    for pdir in ready:
        res = analyse_patient(pdir.name, pdir)
        if res:
            results.append(res)

    if not results:
        print("No results."); return

    # ── Cross-patient summary ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("CROSS-PATIENT SINGULARITY TRAJECTORY")
    print(f"  {'Patient':<8}  " + "  ".join(f"{l:>10}" for l in ALL_LABELS) +
          f"  {'ρ':>6}  {'trend':>12}")
    print(f"  {'-'*8}  " + "  ".join(f"{'-'*10}" for _ in ALL_LABELS) +
          f"  {'-'*6}  {'-'*12}")

    n_decline = 0
    for r in results:
        row = []
        for lbl in ALL_LABELS:
            d = r.get(lbl)
            row.append(f"{d['sing']:.4f}({d['n']:3d})" if d else "  ---     ")
        rho = r.get("spearman_rho")
        trend = "DECLINE" if r.get("monotone_decline") else ("flat/rise" if rho is not None else "n/a")
        if r.get("monotone_decline"):
            n_decline += 1
        rho_s = f"{rho:+.3f}" if rho is not None else "  n/a"
        print(f"  {r['patient']:<8}  " + "  ".join(row) + f"  {rho_s:>6}  {trend:>12}")

    # Mean trajectory across patients
    print(f"\n  {'Mean':8}  ", end="")
    for lbl in ALL_LABELS:
        vals = [r[lbl]["sing"] for r in results if r.get(lbl)]
        if vals:
            print(f"  {np.mean(vals):.4f}({len(vals):3d})", end="")
        else:
            print(f"  ---     ", end="")
    print()

    print(f"\n  Patients with monotone singularity decline: {n_decline}/{len(results)}")

    # ── All three families ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("MEAN TRANSITION FRACS BY BIN (cross-patient)")
    print(f"  {'Bin':12s}  {'sing':>8}  {'sat':>8}  {'cos':>8}  {'N_patients':>10}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for lbl in ALL_LABELS:
        sings = [r[lbl]["sing"] for r in results if r.get(lbl)]
        sats  = [r[lbl]["sat"]  for r in results if r.get(lbl)]
        coss  = [r[lbl]["cos"]  for r in results if r.get(lbl)]
        if not sings: continue
        print(f"  {lbl:12s}  {np.mean(sings):8.4f}  {np.mean(sats):8.4f}"
              f"  {np.mean(coss):8.4f}  {len(sings):>10}")

    # Save
    out_path = Path("results/eeg_preictal_trajectory.json")
    out_path.parent.mkdir(exist_ok=True)
    def _ser(o):
        if isinstance(o, (np.bool_, bool)): return bool(o)
        if isinstance(o, (np.integer,)):    return int(o)
        if isinstance(o, (np.floating,)):   return float(o)
        raise TypeError(type(o))
    out_path.write_text(json.dumps(results, indent=2, default=_ser))
    print(f"\n  Saved: {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
