#!/usr/bin/env python3
"""
54_qa_timeseries_invariants.py — QA Z[φ]/Z[√2] two-structure in solar activity
================================================================================

PRE-REGISTRATION (written BEFORE running any analysis on out-of-sample data):

  Domain: Solar activity (monthly international sunspot number, SILSO)
  Data: Royal Observatory of Belgium, 1749–present
  In-sample (parameter fit): 1749–1974
  Out-of-sample (hypothesis test): 1975–present (modern monitoring era)

  Context: Session 2026-06-03 established that QA dynamics live at the
  intersection of two quadratic forms:
    f(b,e) = b² + be − e²   (Z[φ] invariant, constant along each QA orbit)
    g(b,e) = b² − 2e²       (Z[√2] / Pell — male g>0, female g<0, boundary g=0)

  This is the first experiment testing whether these algebraic invariants carry
  predictive signal in a real, well-understood time series (11-year Schwabe cycle).

HYPOTHESES (pre-registered, thresholds fixed BEFORE examining out-of-sample data):

  H1 (Orbit-Cycle Correlation):
    The rolling 12-month Cosmos-fraction correlates with the rolling 12-month
    smoothed sunspot number (solar cycle phase proxy). Prediction: r > 0.
    Cosmos orbits = dynamically active long-period orbits; higher solar activity
    should push the system into more distinct orbit classes → higher Cosmos fraction.
    Test: Pearson r on rolling windows over out-of-sample.
    PASS threshold: |r| > 0.15, p < 0.01

  H2 (Z[φ] f-Stability predicts low variance):
    Within a 12-month rolling window, define f-stability as the fraction of
    consecutive (b_t, e_t) → (b_{t+1}, e_{t+1}) pairs where
    |f(b_t,e_t) − f(b_{t+1},e_{t+1})| = 0 (Z[φ] class unchanged).
    Windows with f-stability ≥ 0.5 predict lower next-12-month sunspot variance
    (system on a stable orbit segment).
    Test: Mann-Whitney U on next-window variance, stable vs. unstable.
    PASS threshold: p < 0.05, direction: stable_var < unstable_var

  H3 (Z[√2] g-sign → direction prediction):
    g(b_t, e_t) = b_t² − 2·e_t²
    g > 0 (male, above Pell line): predict s[t+3] > s[t]   (sunspot rise)
    g < 0 (female, below Pell line): predict s[t+3] < s[t]  (sunspot fall)
    g = 0: no prediction.
    Test: binomial test on g≠0 subset vs. 50% chance (two-tailed).
    PASS threshold: p < 0.01
    Baseline comparison: naive momentum (s[t]>s[t-3] → predict rise).

Surrogates (N=500): State sequence randomly shuffled, hypotheses recomputed.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=sunspot_microstate, state_alphabet=solar_activity_quantile"

import sys, os, json
import numpy as np
import urllib.request
from sklearn.cluster import KMeans
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qa_orbit_rules import orbit_family

# ── Parameters (fixed before running) ─────────────────────────────────────────
MODULUS    = 24
K          = 6      # K-means clusters
SPLIT_YEAR = 1975   # in-sample < 1975, out-of-sample >= 1975
ROLL_WIN   = 12     # rolling window (months) for H1/H2
FH         = 3      # forecast horizon (months) for H3
N_SURR     = 500
np.random.seed(42)

# CMAP: sorted cluster index (0=lowest sunspot, 5=highest) → QA state in {1..24}
# Spread through all regions of the mod-24 state space; multiples of 8 → satellite.
CMAP_SORTED = {0: 4, 1: 8, 2: 12, 3: 16, 4: 20, 5: 24}


# ── Precomputed orbit/f/g tables (O(m²) once, O(1) per lookup) ────────────────

def _build_tables(m: int):
    fam_int = {"cosmos": 0, "satellite": 1, "singularity": 2}
    orb = np.zeros((m + 1, m + 1), dtype=np.int8)
    f_t = np.zeros((m + 1, m + 1), dtype=np.int64)
    g_t = np.zeros((m + 1, m + 1), dtype=np.int64)
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            orb[b, e] = fam_int[orbit_family(b, e, m)]
            f_t[b, e] = b * b + b * e - e * e   # norm_f
            g_t[b, e] = b * b - 2 * e * e        # Z[√2] Pell
    return orb, f_t, g_t


# ── Data loading ───────────────────────────────────────────────────────────────

def load_silso_monthly():
    url = "https://www.sidc.be/silso/INFO/snmtotcsv.php"
    r = urllib.request.urlopen(url, timeout=30)
    raw = r.read().decode("latin-1")
    rows = []
    for line in raw.strip().split("\n"):
        parts = line.strip().split(";")
        if len(parts) < 4:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            sn    = float(parts[3])
        except ValueError:
            continue
        rows.append((year, month, max(0.0, sn)))
    return rows


def fit_quantizer(sn_train):
    X  = np.array(sn_train).reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    km.fit(X)
    order     = np.argsort(km.cluster_centers_.flatten())  # sorted asc
    inv_order = np.argsort(order)                          # orig → sorted
    cmap = {int(orig): CMAP_SORTED[int(si)] for orig, si in enumerate(inv_order)}
    return km, cmap


def quantize(sn_arr, km, cmap):
    labels = km.predict(sn_arr.reshape(-1, 1))
    return np.array([cmap[int(l)] for l in labels], dtype=np.int32)


# ── Rolling features ───────────────────────────────────────────────────────────

def rolling_features(states, sn_vals, orb_tab, f_tab, test_mask, roll_win, fh):
    """
    Returns parallel arrays indexed by time t (1 ≤ t < n):
      cosmos_frac  : rolling roll_win-month cosmos fraction ending at t
      sn_mean      : rolling roll_win-month sunspot mean ending at t
      sn_var_next  : variance of sn_vals[t:t+roll_win] (forward)
      f_stable     : f-stability flag (≥0.5 of consecutive f pairs equal in window)
      g            : g(b_t, e_t) = states[t]² - 2*states[t-1]²
      sn_now       : sn_vals[t]
      sn_fh        : sn_vals[t+fh] if t+fh < n else nan
    """
    n = len(states)
    cosmos_frac = np.full(n, np.nan)
    sn_mean     = np.full(n, np.nan)
    sn_var_next = np.full(n, np.nan)
    f_stable    = np.full(n, np.nan)
    g_arr       = np.zeros(n, dtype=np.int64)
    sn_fh_arr   = np.full(n, np.nan)

    for t in range(1, n):
        b = int(states[t])
        e = int(states[t - 1])
        g_arr[t] = b * b - 2 * e * e

        if t + fh < n:
            sn_fh_arr[t] = float(sn_vals[t + fh])

        if t < roll_win:
            continue

        # Rolling window [t-roll_win, t)
        w_states = states[t - roll_win: t]
        w_sn     = sn_vals[t - roll_win: t]
        w_orbs   = np.array(
            [orb_tab[int(w_states[i]), int(w_states[i - 1])]
             for i in range(1, len(w_states))],
            dtype=np.int8,
        )
        w_f = np.array(
            [f_tab[int(w_states[i]), int(w_states[i - 1])]
             for i in range(1, len(w_states))],
            dtype=np.int64,
        )
        cosmos_frac[t] = float(np.mean(w_orbs == 0))  # 0=cosmos
        sn_mean[t]     = float(np.mean(w_sn))
        f_stable[t]    = float(np.mean(np.diff(w_f) == 0)) if len(w_f) > 1 else 0.0

        if t + roll_win <= n:
            sn_var_next[t] = float(np.var(sn_vals[t: t + roll_win]))

    return cosmos_frac, sn_mean, sn_var_next, f_stable, g_arr, sn_fh_arr


# ── Hypothesis tests ───────────────────────────────────────────────────────────

def test_h1(cosmos_frac, sn_mean, test_mask):
    mask = test_mask & np.isfinite(cosmos_frac) & np.isfinite(sn_mean)
    if mask.sum() < 10:
        return np.nan, np.nan, False
    r, p = stats.pearsonr(cosmos_frac[mask], sn_mean[mask])
    return float(r), float(p), bool(abs(r) > 0.15 and p < 0.01)


def test_h2(sn_var_next, f_stable, test_mask):
    mask = test_mask & np.isfinite(sn_var_next) & np.isfinite(f_stable)
    stable   = sn_var_next[mask & (f_stable >= 0.5)]
    unstable = sn_var_next[mask & (f_stable < 0.5)]
    if len(stable) < 5 or len(unstable) < 5:
        return np.nan, np.nan, np.nan, len(stable), len(unstable), False
    mw, p = stats.mannwhitneyu(stable, unstable, alternative="less")
    return (float(np.median(stable)), float(np.median(unstable)),
            float(p), int(len(stable)), int(len(unstable)),
            bool(p < 0.05 and np.median(stable) < np.median(unstable)))


def test_h3(g_arr, sn_vals, sn_fh_arr, test_mask, fh):
    mask = test_mask & (g_arr != 0) & np.isfinite(sn_fh_arr)
    if mask.sum() < 20:
        return np.nan, 0, 0, np.nan, False

    g_sub  = g_arr[mask]
    sn_sub = sn_vals[mask]
    fh_sub = sn_fh_arr[mask]

    pred_up   = g_sub > 0
    actual_up = fh_sub > sn_sub
    correct   = int(np.sum(pred_up == actual_up))
    total     = int(len(g_sub))
    acc       = correct / total
    binom     = stats.binomtest(correct, total, 0.5, alternative="two-sided")
    return float(acc), correct, total, float(binom.pvalue), bool(binom.pvalue < 0.01)


def momentum_baseline(sn_vals, test_mask, fh):
    """Naive momentum: s[t]>s[t-fh] → predict s[t+fh]>s[t]."""
    n = len(sn_vals)
    correct = total = 0
    for t in range(fh, n - fh):
        if not test_mask[t]:
            continue
        pred_up   = sn_vals[t] > sn_vals[t - fh]
        actual_up = sn_vals[t + fh] > sn_vals[t]
        if pred_up is not None:
            correct += int(pred_up == actual_up)
            total   += 1
    return correct / total if total else np.nan, total


# ── Surrogate null ─────────────────────────────────────────────────────────────

def run_surrogates(states, sn_vals, orb_tab, f_tab, test_mask, roll_win, fh, n_surr):
    rng = np.random.default_rng(99)
    surr_r_h1  = []
    surr_acc_h3 = []

    for _ in range(n_surr):
        shuf = rng.permuted(states)
        cf, sm, sv, fs, g_s, sfh = rolling_features(
            shuf, sn_vals, orb_tab, f_tab, test_mask, roll_win, fh
        )
        r_s, _, _ = test_h1(cf, sm, test_mask)
        acc_s, _, _, _, _ = test_h3(g_s, sn_vals, sfh, test_mask, fh)
        if np.isfinite(r_s):
            surr_r_h1.append(r_s)
        if np.isfinite(acc_s):
            surr_acc_h3.append(acc_s)

    return np.array(surr_r_h1), np.array(surr_acc_h3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Experiment 54: QA Z[φ]/Z[√2] Two-Structure in Solar Activity")
    print("=" * 70)

    # Precompute tables
    print("Building orbit/f/g tables for mod-24…")
    orb_tab, f_tab, _ = _build_tables(MODULUS)

    # Load data
    print("Downloading SILSO monthly sunspot data…")
    rows = load_silso_monthly()
    print(f"  {len(rows)} months  ({rows[0][0]}-{rows[-1][0]})")

    years    = np.array([r[0] for r in rows])
    months   = np.array([r[1] for r in rows])
    sn_vals  = np.array([r[2] for r in rows])

    # Split
    train_mask = years < SPLIT_YEAR
    test_mask  = years >= SPLIT_YEAR
    print(f"  In-sample (train): {train_mask.sum()} months  (<{SPLIT_YEAR})")
    print(f"  Out-of-sample:     {test_mask.sum()} months (≥{SPLIT_YEAR})")

    # Fit K-means on training data only
    km, cmap = fit_quantizer(sn_vals[train_mask])
    cents = sorted(km.cluster_centers_.flatten())
    print(f"  K={K} centroids: {[round(c,1) for c in cents]}")
    print(f"  CMAP (sorted→state): {cmap}")

    # Quantize all months
    states = quantize(sn_vals, km, cmap)
    n = len(states)

    # Rolling features
    print("\nComputing rolling features…")
    cosmos_frac, sn_mean, sn_var_next, f_stable, g_arr, sn_fh_arr = \
        rolling_features(states, sn_vals, orb_tab, f_tab, test_mask, ROLL_WIN, FH)

    # Orbit distribution on test set
    orb_dist = {}
    test_t = np.where(test_mask)[0]
    for t in test_t[test_t > 0]:
        fam = ["cosmos", "satellite", "singularity"][int(orb_tab[int(states[t]), int(states[t - 1])])]
        orb_dist[fam] = orb_dist.get(fam, 0) + 1
    total_test_pairs = sum(orb_dist.values())
    print("\nOrbit distribution (test set):")
    for fam in ["cosmos", "satellite", "singularity"]:
        cnt = orb_dist.get(fam, 0)
        print(f"  {fam}: {cnt}  ({100*cnt/total_test_pairs:.1f}%)")

    # ── H1 ─────────────────────────────────────────────────────────────────────
    print("\n── H1: Cosmos-fraction ↔ sunspot count (Pearson r) ──")
    r_h1, p_h1, pass_h1 = test_h1(cosmos_frac, sn_mean, test_mask)
    print(f"  r = {r_h1:.4f},  p = {p_h1:.4e}  →  {'PASS ✓' if pass_h1 else 'FAIL'}")

    # ── H2 ─────────────────────────────────────────────────────────────────────
    print("\n── H2: Z[φ] f-stability → lower next-window variance ──")
    med_s, med_u, p_h2, n_s, n_u, pass_h2 = test_h2(sn_var_next, f_stable, test_mask)
    print(f"  stable windows (n={n_s}):   median next-var = {med_s:.1f}")
    print(f"  unstable windows (n={n_u}): median next-var = {med_u:.1f}")
    print(f"  Mann-Whitney p = {p_h2:.4e}  →  {'PASS ✓' if pass_h2 else 'FAIL'}")

    # ── H3 ─────────────────────────────────────────────────────────────────────
    print(f"\n── H3: Z[√2] g-sign → direction ({FH}-month horizon) ──")
    acc_h3, corr_h3, tot_h3, p_h3, pass_h3 = test_h3(
        g_arr, sn_vals, sn_fh_arr, test_mask, FH
    )
    print(f"  Accuracy = {acc_h3:.4f}  ({corr_h3}/{tot_h3}),  p = {p_h3:.4e}"
          f"  →  {'PASS ✓' if pass_h3 else 'FAIL'}")
    mom_acc, mom_n = momentum_baseline(sn_vals, test_mask, FH)
    print(f"  Momentum baseline: {mom_acc:.4f}  (n={mom_n})")

    # ── Surrogates ─────────────────────────────────────────────────────────────
    print(f"\n── Surrogates (N={N_SURR}, shuffled state sequence) ──")
    surr_r, surr_acc = run_surrogates(
        states, sn_vals, orb_tab, f_tab, test_mask, ROLL_WIN, FH, N_SURR
    )
    rank_p_h1 = float(np.mean(np.abs(surr_r) >= abs(r_h1))) if len(surr_r) else np.nan
    rank_p_h3 = float(np.mean(surr_acc >= acc_h3)) if len(surr_acc) else np.nan
    print(f"  H1  rank-p = {rank_p_h1:.3f}  (real |r|={abs(r_h1):.4f},"
          f" surr mean={np.mean(np.abs(surr_r)):.4f})")
    print(f"  H3  rank-p = {rank_p_h3:.3f}  (real acc={acc_h3:.4f},"
          f" surr mean={np.mean(surr_acc):.4f})")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "domain": "solar_activity_sunspot",
        "data": "SILSO monthly international sunspot number",
        "n_months_train": int(train_mask.sum()),
        "n_months_test":  int(test_mask.sum()),
        "params": {"modulus": MODULUS, "K": K, "split_year": SPLIT_YEAR,
                   "roll_win": ROLL_WIN, "FH": FH},
        "cmap": {str(k): v for k, v in cmap.items()},
        "orbit_distribution_test": orb_dist,
        "H1": {
            "test": "Pearson r (cosmos-frac vs sunspot rolling 12m)",
            "r": float(r_h1), "p": float(p_h1),
            "threshold": "|r|>0.15, p<0.01", "pass": bool(pass_h1),
            "surr_rank_p": float(rank_p_h1),
        },
        "H2": {
            "test": "Mann-Whitney U (next-window variance, f-stable vs f-unstable)",
            "median_stable_var": float(med_s) if med_s is not None else None,
            "median_unstable_var": float(med_u) if med_u is not None else None,
            "n_stable": int(n_s), "n_unstable": int(n_u),
            "p": float(p_h2), "threshold": "p<0.05, direction: stable<unstable",
            "pass": bool(pass_h2),
        },
        "H3": {
            "test": f"Binomial (g-sign direction, FH={FH} months)",
            "accuracy": float(acc_h3), "n_correct": int(corr_h3),
            "n_total": int(tot_h3), "p": float(p_h3),
            "threshold": "p<0.01", "pass": bool(pass_h3),
            "momentum_baseline_acc": float(mom_acc),
            "surr_rank_p": float(rank_p_h3),
        },
        "n_hypotheses_pass": int(pass_h1) + int(pass_h2) + int(pass_h3),
    }

    outfile = os.path.join(HERE, "54_qa_timeseries_invariants_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {outfile}")

    passed = results["n_hypotheses_pass"]
    print(f"\n{'='*70}")
    print(f"Summary: {passed}/3 hypotheses PASS")
    print(f"  H1 (Orbit-cycle correlation): {'PASS' if pass_h1 else 'FAIL'}")
    print(f"  H2 (Z[φ] f-stability):        {'PASS' if pass_h2 else 'FAIL'}")
    print(f"  H3 (Z[√2] g-sign direction):  {'PASS' if pass_h3 else 'FAIL'}")
    return results


if __name__ == "__main__":
    main()
