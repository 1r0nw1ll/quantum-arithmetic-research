#!/usr/bin/env python3
"""
55_qa_synchronous_harmonics.py — Multi-scale QA orbit synchrony in solar activity
===================================================================================

PRE-REGISTRATION (written BEFORE running any analysis on out-of-sample data):

  Domain: Solar activity (SILSO monthly sunspot, 1749–present)
  In-sample (parameter fit): 1749–1974
  Out-of-sample (hypothesis test): 1975–present

  Context: Exp 54 confirmed Z[φ] f-stability as a variance indicator (p=4.7e-19).
  QA mod-24 has two natural periods: Satellite=8, Cosmos=24 (ratio 3:1).
  This experiment tests whether those periods create synchronous structure when
  the SAME time series is analysed simultaneously at three temporal lags.

  Three simultaneous scales at each time t:
    S1  = orbit_family(state[t],   state[t-1])   — monthly lag
    S3  = orbit_family(state[t],   state[t-3])   — quarterly lag
    S12 = orbit_family(state[t],   state[t-12])  — annual lag

  Key mathematical fact (traced in experiment design):
    The satellite orbit (period 8) is closed under all lags: lag-k from any
    satellite pair lands on another satellite pair for all k ∈ {1,...,7}.
    The cosmos orbit (period 24) is closed under lag-k for k ∈ {1,...,23}.
    → If the physical system genuinely visits satellite states for ≥12 consecutive
      months (one full Schwabe sub-phase), ALL THREE scales should read "satellite"
      simultaneously. This synchrony is falsifiable.

HYPOTHESES (pre-registered, thresholds fixed BEFORE examining out-of-sample):

  H1 (Scale dependence):
    The joint distribution of (S1, S3) is not independent.
    Test: chi-square test on (S1, S3) 3×3 contingency table vs. product-of-
    marginals null.
    PASS threshold: p < 0.01

  H2 (Satellite synchrony cascade):
    P(S3=Satellite | S1=Satellite) > P(S3=Satellite)
    AND
    P(S12=Satellite | S3=Satellite) > P(S12=Satellite)
    Both enrichments must hold.
    Test: Fisher exact on each conditional; enrichment ratio reported.
    PASS threshold: both one-sided p < 0.05, both ratios > 1

  H3 (3:1 harmonic run-length signature):
    During annual-Satellite windows (S12=Satellite), identify runs of consecutive
    S1=Satellite months. The mean run length should cluster near 8 (one satellite
    period) rather than near 1 (exponential decay null) or 3 (random-lag null).
    Test: t-test, mean run length vs. 1 (p<0.01); report whether mean is closer
    to 8 than to 1.
    PASS threshold: mean > 4 (midpoint of 1..8), p < 0.01

  H4 (Three-scale synchrony):
    The fraction of months where S1==S3==S12 exceeds what independence predicts.
    Expected under independence: P(agree) = Σ_fam P(S1=fam)*P(S3=fam)*P(S12=fam)
    Test: binomial test (observed agree-count vs. expected-count).
    PASS threshold: p < 0.01

Surrogates (N=500): State sequence shuffled; all four hypotheses recomputed.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=sunspot_multiscale, state_alphabet=solar_activity_quantile"

import sys, os, json
import numpy as np
import urllib.request
from sklearn.cluster import KMeans
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qa_orbit_rules import orbit_family

# ── Parameters ─────────────────────────────────────────────────────────────────
MODULUS    = 24
K          = 6
SPLIT_YEAR = 1975
N_SURR     = 500
LAGS       = [1, 3, 12]          # monthly, quarterly, annual
CMAP_SORTED = {0: 4, 1: 8, 2: 12, 3: 16, 4: 20, 5: 24}
FAM_INT    = {"cosmos": 0, "satellite": 1, "singularity": 2}
INT_FAM    = {0: "cosmos", 1: "satellite", 2: "singularity"}
np.random.seed(42)


# ── Precomputed orbit table ─────────────────────────────────────────────────────

def _build_orbit_table(m: int) -> np.ndarray:
    tbl = np.zeros((m + 1, m + 1), dtype=np.int8)
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            tbl[b, e] = FAM_INT[orbit_family(b, e, m)]
    return tbl


# ── Data loading / quantization (identical to exp 54) ──────────────────────────

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
            year, month, sn = int(parts[0]), int(parts[1]), float(parts[3])
        except ValueError:
            continue
        rows.append((year, month, max(0.0, sn)))
    return rows


def fit_quantizer(sn_train):
    X  = np.array(sn_train).reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    km.fit(X)
    order     = np.argsort(km.cluster_centers_.flatten())
    inv_order = np.argsort(order)
    cmap = {int(orig): CMAP_SORTED[int(si)] for orig, si in enumerate(inv_order)}
    return km, cmap


def quantize(sn_arr, km, cmap):
    labels = km.predict(sn_arr.reshape(-1, 1))
    return np.array([cmap[int(l)] for l in labels], dtype=np.int32)


# ── Multi-scale orbit assignment ────────────────────────────────────────────────

def multi_scale_orbits(states: np.ndarray, orb_tbl: np.ndarray, lags: list[int]):
    """
    Returns dict lag → array of orbit-int (length n, nan-coded as -1 for t<lag).
    orbit-int: 0=cosmos, 1=satellite, 2=singularity, -1=undefined (t < lag).
    """
    n = len(states)
    result = {}
    for lag in lags:
        arr = np.full(n, -1, dtype=np.int8)
        for t in range(lag, n):
            b = int(states[t])
            e = int(states[t - lag])
            arr[t] = orb_tbl[b, e]
        result[lag] = arr
    return result


# ── Hypothesis tests ────────────────────────────────────────────────────────────

def test_h1(s1, s3, test_mask):
    """Chi-square test: joint (S1, S3) distribution vs independence.
    Singularity is merged into cosmos (rare class; zero cells break chi2)."""
    valid = test_mask & (s1 >= 0) & (s3 >= 0)
    if valid.sum() < 30:
        return np.nan, False
    # 2-class: cosmos-or-singularity(0) vs satellite(1)
    s1b = (s1[valid] == 1).astype(int)
    s3b = (s3[valid] == 1).astype(int)
    contingency = np.zeros((2, 2), dtype=int)
    for i in range(2):
        for j in range(2):
            contingency[i, j] = int(np.sum((s1b == i) & (s3b == j)))
    if contingency.min() == 0:
        return 1.0, False
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return float(p), bool(p < 0.01)


def test_h2(s1, s3, s12, test_mask):
    """Fisher exact: satellite cascade across scales."""
    valid13  = test_mask & (s1 >= 0) & (s3 >= 0)
    valid312 = test_mask & (s3 >= 0) & (s12 >= 0)

    # P(S3=sat | S1=sat) vs P(S3=sat | S1=cosmos)
    a = int(np.sum((s1[valid13] == 1) & (s3[valid13] == 1)))  # sat/sat
    b = int(np.sum((s1[valid13] == 1) & (s3[valid13] != 1)))  # sat/~sat
    c = int(np.sum((s1[valid13] == 0) & (s3[valid13] == 1)))  # cos/sat
    d = int(np.sum((s1[valid13] == 0) & (s3[valid13] != 1)))  # cos/~sat
    _, p13 = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    ratio13 = (a / (a + b)) / (c / (c + d)) if (a + b) > 0 and (c + d) > 0 else np.nan

    # P(S12=sat | S3=sat) vs P(S12=sat | S3=cosmos)
    a2 = int(np.sum((s3[valid312] == 1) & (s12[valid312] == 1)))
    b2 = int(np.sum((s3[valid312] == 1) & (s12[valid312] != 1)))
    c2 = int(np.sum((s3[valid312] == 0) & (s12[valid312] == 1)))
    d2 = int(np.sum((s3[valid312] == 0) & (s12[valid312] != 1)))
    _, p312 = stats.fisher_exact([[a2, b2], [c2, d2]], alternative="greater")
    ratio312 = (a2 / (a2 + b2)) / (c2 / (c2 + d2)) if (a2 + b2) > 0 and (c2 + d2) > 0 else np.nan

    pass_h2 = bool(p13 < 0.05 and p312 < 0.05 and
                   (np.isfinite(ratio13) and ratio13 > 1) and
                   (np.isfinite(ratio312) and ratio312 > 1))
    return float(p13), float(p312), float(ratio13), float(ratio312), pass_h2


def test_h3(s1, s12, test_mask):
    """Run-length analysis: S1=satellite runs during S12=satellite windows."""
    valid = test_mask & (s1 >= 0) & (s12 >= 0)
    in_annual_sat = valid & (s12 == 1)

    # Extract satellite run lengths in s1 restricted to annual-satellite periods
    runs = []
    in_run = False
    run_len = 0
    n = len(s1)
    for t in range(n):
        if not in_annual_sat[t]:
            if in_run and run_len > 0:
                runs.append(run_len)
            in_run = False
            run_len = 0
            continue
        if s1[t] == 1:  # monthly satellite
            in_run = True
            run_len += 1
        else:
            if in_run and run_len > 0:
                runs.append(run_len)
            in_run = False
            run_len = 0
    if in_run and run_len > 0:
        runs.append(run_len)

    if len(runs) < 5:
        return np.nan, np.nan, runs, False

    runs_arr = np.array(runs, dtype=float)
    mean_run = float(np.mean(runs_arr))
    t_stat, p_t = stats.ttest_1samp(runs_arr, popmean=1.0, alternative="greater")
    pass_h3 = bool(mean_run > 4.0 and p_t < 0.01)
    return mean_run, float(p_t), list(runs), pass_h3


def test_h4(s1, s3, s12, test_mask):
    """Three-scale synchrony: observed agree fraction vs product-of-marginals."""
    valid = test_mask & (s1 >= 0) & (s3 >= 0) & (s12 >= 0)
    n_valid = int(valid.sum())
    if n_valid < 30:
        return np.nan, np.nan, np.nan, False

    s1v, s3v, s12v = s1[valid], s3[valid], s12[valid]

    # Observed
    agree = int(np.sum((s1v == s3v) & (s3v == s12v)))
    obs_frac = agree / n_valid

    # Expected under independence
    p_expected = 0.0
    for fam in range(3):
        p1  = float(np.mean(s1v == fam))
        p3  = float(np.mean(s3v == fam))
        p12 = float(np.mean(s12v == fam))
        p_expected += p1 * p3 * p12
    expected_count = p_expected * n_valid

    binom = stats.binomtest(agree, n_valid, p_expected, alternative="greater")
    pass_h4 = bool(binom.pvalue < 0.01 and obs_frac > p_expected)
    return float(obs_frac), float(p_expected), float(binom.pvalue), pass_h4


# ── Surrogate null ─────────────────────────────────────────────────────────────

def run_surrogates(states, orb_tbl, test_mask, lags, n_surr):
    rng   = np.random.default_rng(99)
    null_p_h1   = []
    null_p13_h2 = []
    null_frac_h4 = []

    for _ in range(n_surr):
        shuf  = rng.permuted(states)
        ms    = multi_scale_orbits(shuf, orb_tbl, lags)
        s1s, s3s, s12s = ms[1], ms[3], ms[12]

        p1, _       = test_h1(s1s, s3s, test_mask)
        p13, _, _, _, _ = test_h2(s1s, s3s, s12s, test_mask)
        frac, _, _, _   = test_h4(s1s, s3s, s12s, test_mask)

        if np.isfinite(p1):   null_p_h1.append(p1)
        if np.isfinite(p13):  null_p13_h2.append(p13)
        if np.isfinite(frac): null_frac_h4.append(frac)

    return (np.array(null_p_h1), np.array(null_p13_h2), np.array(null_frac_h4))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Experiment 55: QA Multi-Scale Synchronous Harmonics (Solar Activity)")
    print("=" * 70)

    print("Building orbit table for mod-24…")
    orb_tbl = _build_orbit_table(MODULUS)

    print("Downloading SILSO monthly sunspot data…")
    rows = load_silso_monthly()
    years   = np.array([r[0] for r in rows])
    sn_vals = np.array([r[2] for r in rows])
    print(f"  {len(rows)} months  ({rows[0][0]}–{rows[-1][0]})")

    train_mask = years < SPLIT_YEAR
    test_mask  = years >= SPLIT_YEAR
    print(f"  In-sample: {train_mask.sum()} months  |  Out-of-sample: {test_mask.sum()} months")

    km, cmap = fit_quantizer(sn_vals[train_mask])
    states   = quantize(sn_vals, km, cmap)

    # Multi-scale orbit assignment
    print("\nComputing multi-scale orbit assignments…")
    ms = multi_scale_orbits(states, orb_tbl, LAGS)
    s1, s3, s12 = ms[1], ms[3], ms[12]

    # Distribution summary
    for lag, s in [(1, s1), (3, s3), (12, s12)]:
        valid = test_mask & (s >= 0)
        n_v   = valid.sum()
        if n_v == 0:
            continue
        sv = s[valid]
        print(f"\n  Lag-{lag:2d} orbit distribution (test, n={n_v}):")
        for fam in range(3):
            cnt = int(np.sum(sv == fam))
            print(f"    {INT_FAM[fam]:12s}: {cnt:4d}  ({100*cnt/n_v:.1f}%)")

    # ── H1 ─────────────────────────────────────────────────────────────────────
    print("\n── H1: Joint (S1,S3) vs independence (chi-square) ──")
    p_h1, pass_h1 = test_h1(s1, s3, test_mask)
    print(f"  p = {p_h1:.4e}  →  {'PASS ✓' if pass_h1 else 'FAIL'}")

    # Show contingency
    valid13 = test_mask & (s1 >= 0) & (s3 >= 0)
    print("  Contingency (S1 rows, S3 cols — cosmos/satellite/singularity):")
    for i in range(3):
        row = [int(np.sum((s1[valid13] == i) & (s3[valid13] == j))) for j in range(3)]
        print(f"    {INT_FAM[i]:12s}: {row}")

    # ── H2 ─────────────────────────────────────────────────────────────────────
    print("\n── H2: Satellite cascade across scales (Fisher exact) ──")
    p13, p312, r13, r312, pass_h2 = test_h2(s1, s3, s12, test_mask)
    print(f"  S1→S3  : p={p13:.4e},  enrichment ratio={r13:.3f}")
    print(f"  S3→S12 : p={p312:.4e},  enrichment ratio={r312:.3f}")
    print(f"  →  {'PASS ✓' if pass_h2 else 'FAIL'}")

    # ── H3 ─────────────────────────────────────────────────────────────────────
    print("\n── H3: Satellite run lengths during annual-Satellite windows ──")
    mean_run, p_h3, runs, pass_h3 = test_h3(s1, s12, test_mask)
    if runs:
        from collections import Counter
        run_dist = dict(sorted(Counter(runs).items()))
        print(f"  n_runs = {len(runs)},  mean_length = {mean_run:.2f}")
        print(f"  Run-length distribution: {run_dist}")
        print(f"  t-test (mean > 1): p = {p_h3:.4e}  →  {'PASS ✓' if pass_h3 else 'FAIL'}")
    else:
        print("  Insufficient runs")

    # ── H4 ─────────────────────────────────────────────────────────────────────
    print("\n── H4: Three-scale synchrony vs independence ──")
    obs_frac, exp_frac, p_h4, pass_h4 = test_h4(s1, s3, s12, test_mask)
    print(f"  Observed agree fraction = {obs_frac:.4f}")
    print(f"  Expected (independence) = {exp_frac:.4f}")
    print(f"  Lift = {obs_frac/exp_frac:.3f}x,  p = {p_h4:.4e}  →  {'PASS ✓' if pass_h4 else 'FAIL'}")

    # ── Surrogates ─────────────────────────────────────────────────────────────
    print(f"\n── Surrogates (N={N_SURR}) ──")
    null_p_h1, null_p13, null_frac_h4 = run_surrogates(
        states, orb_tbl, test_mask, LAGS, N_SURR
    )
    if len(null_p_h1):
        rank_h1 = float(np.mean(null_p_h1 <= p_h1))
        print(f"  H1  rank-p = {rank_h1:.3f}  (real p={p_h1:.2e}, surr median p={np.median(null_p_h1):.2e})")
    if len(null_p13):
        rank_h2 = float(np.mean(null_p13 <= p13))
        print(f"  H2  rank-p = {rank_h2:.3f}  (real p13={p13:.2e}, surr median={np.median(null_p13):.2e})")
    if len(null_frac_h4):
        rank_h4 = float(np.mean(null_frac_h4 >= obs_frac))
        print(f"  H4  rank-p = {rank_h4:.3f}  (real frac={obs_frac:.4f}, surr median={np.median(null_frac_h4):.4f})")

    # ── Save ───────────────────────────────────────────────────────────────────
    results = {
        "domain": "solar_activity_multiscale",
        "data": "SILSO monthly international sunspot number",
        "n_months_test": int(test_mask.sum()),
        "params": {"modulus": MODULUS, "K": K, "split_year": SPLIT_YEAR,
                   "lags": LAGS},
        "H1": {
            "test": "chi-square (S1,S3) joint vs independence",
            "p": float(p_h1), "threshold": "p<0.01", "pass": bool(pass_h1),
            "surr_rank_p": float(rank_h1) if len(null_p_h1) else None,
        },
        "H2": {
            "test": "Fisher exact satellite cascade",
            "p_S1_to_S3": float(p13), "enrichment_S1_to_S3": float(r13),
            "p_S3_to_S12": float(p312), "enrichment_S3_to_S12": float(r312),
            "threshold": "both p<0.05, both ratio>1", "pass": bool(pass_h2),
            "surr_rank_p_S1_S3": float(rank_h2) if len(null_p13) else None,
        },
        "H3": {
            "test": "t-test satellite run length > 1",
            "mean_run_length": float(mean_run) if mean_run is not None and np.isfinite(mean_run) else None,
            "p": float(p_h3) if p_h3 is not None and np.isfinite(p_h3) else None,
            "n_runs": len(runs),
            "threshold": "mean>4, p<0.01", "pass": bool(pass_h3),
        },
        "H4": {
            "test": "binomial synchrony vs independence",
            "observed_frac": float(obs_frac),
            "expected_frac": float(exp_frac),
            "lift": float(obs_frac / exp_frac) if exp_frac > 0 else None,
            "p": float(p_h4), "threshold": "p<0.01", "pass": bool(pass_h4),
            "surr_rank_p": float(rank_h4) if len(null_frac_h4) else None,
        },
        "n_hypotheses_pass": int(pass_h1) + int(pass_h2) + int(pass_h3) + int(pass_h4),
    }

    outfile = os.path.join(HERE, "55_qa_synchronous_harmonics_results.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {outfile}")

    passed = results["n_hypotheses_pass"]
    print(f"\n{'='*70}")
    print(f"Summary: {passed}/4 hypotheses PASS")
    print(f"  H1 (Scale dependence chi-sq):         {'PASS' if pass_h1 else 'FAIL'}")
    print(f"  H2 (Satellite cascade S1→S3→S12):     {'PASS' if pass_h2 else 'FAIL'}")
    print(f"  H3 (Run length ~ 8 in annual-sat):    {'PASS' if pass_h3 else 'FAIL'}")
    print(f"  H4 (Three-scale synchrony > indep):   {'PASS' if pass_h4 else 'FAIL'}")
    return results


if __name__ == "__main__":
    main()
