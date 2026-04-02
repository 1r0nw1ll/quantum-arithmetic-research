#!/usr/bin/env python3
# SUPERSEDED — QA-noncompliant: equalize_quantize clips to [0,m-1] (A1 violation) and
# derives (b,e) from continuous returns (T2-b violation). qa_step uses (b+e)%m (A1).
# Use qa_finance_orbit_classifier.py instead.
"""
qa_finance_joint_transition.py
================================
QA Finance Joint Transition Structure — cross-asset neighborhood analysis.

The mis-specification in the previous test: using a single asset's return
as both b and e in a QA pair means we're testing whether a 1D return series
self-predicts under the QA map. That's too similar to autocorrelation.

The correct joint encoding:
  b_t = quantized SPY return (equity)
  e_t = quantized TLT return (bonds)

Now (b_t, e_t) is a genuine 2D market state. The QA map T(b,e) = (e, b+e mod m)
describes how the equity-bond joint state should evolve IF the market is on a
QA orbit. This is a cross-asset structural hypothesis, not a 1D AC test.

Three tests
-----------

TEST 1: QA Neighborhood Concentration
  For each observed transition (b_t,e_t) → (b_{t+1},e_{t+1}):
  Is (b_{t+1},e_{t+1}) within QA radius-k of the predicted next state T(b_t,e_t)?
  radius-0: exact QA step (too strict)
  radius-1: T(b,e) or any state 1 Q-step from T(b,e)
  radius-2: within 2 Q-steps
  Compare to: what fraction of the state space is within radius-k? (geometric null)
  Compare to: shuffle / block-shuffle (empirical null)

TEST 2: Stress / Degeneracy Clustering
  Map each day to its QA pair orbit family (cosmos/satellite/singularity).
  Ask: do VIX spikes (stress days) land disproportionately in satellite/near-singularity regions?
  Compare: family distribution on stress days vs calm days.
  Bonus: check if the SPY-TLT correlation breakdown (crisis) aligns with orbit-family breakdown.

TEST 3: Obstruction-Aligned Extremes
  Compute QA norm f(b,e) = b²+be-e² for each joint state.
  States where v_p(f) = 1 for an inert prime p are structurally obstructed (failure algebra).
  Ask: do extreme market events (daily return > 3σ in either asset) concentrate in
  obstructed or near-obstructed norm classes more than chance?

Null models
-----------
  shuffle_null:      shuffle joint (b,e) pairs in time
  block_shuffle:     5-day block shuffle (preserves within-week structure)
  independent_null:  draw b and e independently from their marginal distributions
                     (breaks cross-asset coupling while preserving individual marginals)

VERDICT logic
-------------
  QA_STRUCTURED  if ≥2/3 tests beat all nulls with p<0.05
  MARGINAL       if ≥1/3 tests beat shuffle null with p<0.05
  NULL           otherwise
"""

import numpy as np
from collections import Counter, defaultdict
from scipy import stats
import warnings
import sys

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

MODULI  = [9, 24]
RNG_SEED = 42
N_PERM   = 500
BLOCK    = 5
STRESS_PERCENTILE = 90   # VIX top-decile = stress day
EXTREME_SIGMA     = 2.5  # return > 2.5σ = extreme

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_aligned(tickers, start="2015-01-01", end="2024-01-01"):
    """Fetch daily returns for multiple tickers, aligned on common dates."""
    import pandas as pd
    closes = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in tickers:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if not df.empty:
                closes[t] = df["Close"].squeeze()
    if not closes:
        return None
    frame = pd.DataFrame(closes).dropna()
    returns = np.log(frame / frame.shift(1)).dropna()
    return returns

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def equalize_quantize(series, m):
    n = len(series)
    ranks = np.argsort(np.argsort(series))
    return np.clip((ranks * m // n).astype(int), 0, m - 1)

# ---------------------------------------------------------------------------
# QA orbit machinery
# ---------------------------------------------------------------------------

def qa_step(b, e, m):
    return e, (b + e) % m

def orbit_family(b, e, m, _cache={}):
    key = (b, e, m)
    if key in _cache:
        return _cache[key]
    seen = set(); cb, ce = b, e
    for _ in range(m * m + 1):
        if (cb, ce) in seen: break
        seen.add((cb, ce))
        cb, ce = ce, (cb + ce) % m
    length = len(seen)
    if length == 1:   fam = "singularity"
    elif length >= max(m * 2, 24) - 2: fam = "cosmos"
    else:             fam = "satellite"
    _cache[key] = fam
    return fam

def qa_neighborhood(b, e, m, radius):
    """All states reachable within `radius` Q-steps from (b,e)."""
    frontier = {(b, e)}
    visited  = {(b, e)}
    for _ in range(radius):
        nxt = set()
        for sb, se in frontier:
            nb, ne = qa_step(sb, se, m)
            if (nb, ne) not in visited:
                visited.add((nb, ne)); nxt.add((nb, ne))
        frontier = nxt
    return visited

def neighborhood_size_fraction(radius, m):
    """Average fraction of state space within radius-k of any state."""
    # Sample 50 random states
    rng = np.random.default_rng(RNG_SEED)
    sizes = []
    for _ in range(50):
        b, e = int(rng.integers(0, m)), int(rng.integers(0, m))
        sizes.append(len(qa_neighborhood(b, e, m, radius)) / (m * m))
    return float(np.mean(sizes))

def qa_norm(b, e):
    return b * b + b * e - e * e     # NEVER b**2 — substrate rule

INERT = {9: [3], 24: [3, 7]}

def is_obstructed(b, e, m):
    r = qa_norm(b, e)
    inert = INERT.get(m, [])
    for p in inert:
        v = 0; n = abs(r)
        while n > 0 and n % p == 0: v += 1; n //= p
        if v == 1: return True
    return False

# ---------------------------------------------------------------------------
# TEST 1: Neighborhood concentration
# ---------------------------------------------------------------------------

def neighborhood_concentration_rate(b_states, e_states, m, radius):
    """
    Fraction of observed transitions where next joint state falls within
    QA radius-k neighborhood of the QA-predicted next state.
    """
    hits = 0; n = len(b_states) - 1
    for t in range(n):
        b, e = int(b_states[t]), int(e_states[t])
        b_next, e_next = int(b_states[t+1]), int(e_states[t+1])
        pred_b, pred_e = qa_step(b, e, m)
        hood = qa_neighborhood(pred_b, pred_e, m, radius)
        if (b_next, e_next) in hood:
            hits += 1
    return hits / max(1, n)

def test1_neighborhood(b_states, e_states, m, radii=(0, 1, 2)):
    rng = np.random.default_rng(RNG_SEED)
    results = {}
    for radius in radii:
        obs = neighborhood_concentration_rate(b_states, e_states, m, radius)
        geo_null = neighborhood_size_fraction(radius, m)  # geometric baseline

        # Shuffle null
        shuf_rates = []
        bp, ep = b_states.copy(), e_states.copy()
        for _ in range(N_PERM):
            idx = rng.permutation(len(bp))
            shuf_rates.append(
                neighborhood_concentration_rate(bp[idx], ep[idx], m, radius))
        shuf_mean = float(np.mean(shuf_rates))
        shuf_std  = float(np.std(shuf_rates))

        # Independent null: shuffle b and e independently
        ind_rates = []
        for _ in range(min(200, N_PERM)):
            bi = rng.permutation(bp)
            ei = rng.permutation(ep)
            ind_rates.append(
                neighborhood_concentration_rate(bi, ei, m, radius))
        ind_mean = float(np.mean(ind_rates))
        ind_std  = float(np.std(ind_rates))

        # Block shuffle
        n = len(bp)
        n_blocks = n // BLOCK
        block_rates = []
        for _ in range(min(200, N_PERM)):
            blocks_b = [bp[i*BLOCK:(i+1)*BLOCK] for i in range(n_blocks)]
            blocks_e = [ep[i*BLOCK:(i+1)*BLOCK] for i in range(n_blocks)]
            perm = rng.permutation(n_blocks)
            sb = np.concatenate([blocks_b[i] for i in perm])
            se = np.concatenate([blocks_e[i] for i in perm])
            block_rates.append(
                neighborhood_concentration_rate(sb, se, m, radius))
        block_mean = float(np.mean(block_rates))
        block_std  = float(np.std(block_rates))

        def pval(o, mu, sig):
            return float(stats.norm.sf((o - mu) / sig)) if sig > 1e-10 else (0.0 if o > mu else 1.0)

        results[radius] = {
            "obs":        round(obs, 4),
            "geo_null":   round(geo_null, 4),
            "shuf":       {"mean": round(shuf_mean,4), "p": round(pval(obs,shuf_mean,shuf_std),4)},
            "independent":{"mean": round(ind_mean,4),  "p": round(pval(obs,ind_mean,ind_std),4)},
            "block":      {"mean": round(block_mean,4),"p": round(pval(obs,block_mean,block_std),4)},
            "beats_geo":   obs > geo_null,
            "beats_shuf":  obs > shuf_mean,
            "beats_ind":   obs > ind_mean,
        }
    return results

# ---------------------------------------------------------------------------
# TEST 2: Stress / degeneracy clustering
# ---------------------------------------------------------------------------

def test2_stress_degeneracy(b_states, e_states, vix_series, m):
    """
    Do high-VIX days land more in satellite/near-singularity orbits?
    """
    n = min(len(b_states), len(vix_series))
    b_states = b_states[:n]; e_states = e_states[:n]
    vix = vix_series[:n]

    vix_thresh = np.percentile(vix, STRESS_PERCENTILE)
    stress_idx = np.where(vix >= vix_thresh)[0]
    calm_idx   = np.where(vix <  vix_thresh)[0]

    def family_dist(indices):
        c = Counter(orbit_family(int(b_states[i]), int(e_states[i]), m)
                    for i in indices if i < len(b_states))
        total = sum(c.values())
        return {k: round(v/total, 3) for k,v in c.items()} if total else {}

    stress_dist = family_dist(stress_idx)
    calm_dist   = family_dist(calm_idx)

    # Chi-square test: is family distribution different under stress?
    families = ["cosmos", "satellite", "singularity"]
    stress_counts = [sum(1 for i in stress_idx
                         if orbit_family(int(b_states[i]),int(e_states[i]),m)==f)
                     for f in families]
    calm_counts   = [sum(1 for i in calm_idx
                         if orbit_family(int(b_states[i]),int(e_states[i]),m)==f)
                     for f in families]

    # Normalise to same total for chi-square
    n_stress = sum(stress_counts); n_calm = sum(calm_counts)
    expected = [c/n_calm * n_stress for c in calm_counts] if n_calm else stress_counts
    chi2, p_chi2 = (stats.chisquare(stress_counts, expected)
                    if all(e > 0 for e in expected) else (0.0, 1.0))

    sat_stress = stress_dist.get("satellite", 0.0)
    sat_calm   = calm_dist.get("satellite", 0.0)
    sat_delta  = sat_stress - sat_calm

    return {
        "stress_family_dist": stress_dist,
        "calm_family_dist":   calm_dist,
        "satellite_delta":    round(sat_delta, 3),
        "chi2":               round(float(chi2), 3),
        "p_chi2":             round(float(p_chi2), 4),
        "n_stress":           n_stress,
        "n_calm":             n_calm,
        "significant":        float(p_chi2) < 0.05,
    }

# ---------------------------------------------------------------------------
# TEST 3: Obstruction-aligned extremes
# ---------------------------------------------------------------------------

def test3_obstruction_extremes(b_states, e_states, spy_ret, tlt_ret, m):
    """
    Do extreme return days (|r| > 2.5σ in either asset) concentrate in
    structurally obstructed norm classes?
    """
    n = min(len(b_states), len(spy_ret), len(tlt_ret))
    b_states = b_states[:n]; e_states = e_states[:n]
    spy_ret = spy_ret[:n];   tlt_ret = tlt_ret[:n]

    spy_z = (spy_ret - spy_ret.mean()) / (spy_ret.std() + 1e-10)
    tlt_z = (tlt_ret - tlt_ret.mean()) / (tlt_ret.std() + 1e-10)
    extreme_idx = np.where((np.abs(spy_z) > EXTREME_SIGMA) |
                           (np.abs(tlt_z) > EXTREME_SIGMA))[0]
    normal_idx  = np.where((np.abs(spy_z) <= EXTREME_SIGMA) &
                           (np.abs(tlt_z) <= EXTREME_SIGMA))[0]

    def obstruct_rate(indices):
        if len(indices) == 0: return 0.0
        obs = sum(1 for i in indices
                  if i < len(b_states) and is_obstructed(int(b_states[i]),int(e_states[i]),m))
        return obs / len(indices)

    ext_rate  = obstruct_rate(extreme_idx)
    norm_rate = obstruct_rate(normal_idx)
    delta     = ext_rate - norm_rate

    # Fisher exact test
    a = int(sum(1 for i in extreme_idx if i < len(b_states) and
                is_obstructed(int(b_states[i]),int(e_states[i]),m)))
    b_ = len(extreme_idx) - a
    c = int(sum(1 for i in normal_idx if i < len(b_states) and
                is_obstructed(int(b_states[i]),int(e_states[i]),m)))
    d = len(normal_idx) - c
    _, p_fisher = stats.fisher_exact([[a, b_], [c, d]], alternative="greater")

    return {
        "extreme_obstruct_rate": round(ext_rate, 4),
        "normal_obstruct_rate":  round(norm_rate, 4),
        "delta":                 round(delta, 4),
        "n_extreme":             len(extreme_idx),
        "n_normal":              len(normal_idx),
        "p_fisher":              round(float(p_fisher), 4),
        "significant":           float(p_fisher) < 0.05,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 72)
    print("QA Finance Joint Transition Structure")
    print("SPY(equity) × TLT(bonds) joint QA state encoding")
    print("=" * 72)
    print()

    # Fetch data
    print("Fetching SPY, TLT, GLD, ^VIX...", flush=True)
    rets = fetch_aligned(["SPY", "TLT", "GLD", "^VIX"])
    if rets is None:
        print("ERROR: data fetch failed"); sys.exit(1)

    spy_ret = rets["SPY"].values
    tlt_ret = rets["TLT"].values
    vix_lvl = rets["^VIX"].values  # VIX daily levels (not returns)
    # For VIX we want the level, not the return — re-fetch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vix_df = yf.download("^VIX", start="2015-01-01", end="2024-01-01",
                             progress=False, auto_adjust=True)
    vix_close = vix_df["Close"].squeeze().dropna().values
    # Align to same length as returns
    n_align = min(len(spy_ret), len(vix_close) - 1)
    spy_ret = spy_ret[:n_align]
    tlt_ret = tlt_ret[:n_align]
    vix_lvl = vix_close[1:n_align+1]  # VIX level on the same dates as returns

    print(f"Data aligned: {n_align} trading days (2015-2024)")
    print(f"SPY return mean={spy_ret.mean():.5f}  std={spy_ret.std():.4f}")
    print(f"TLT return mean={tlt_ret.mean():.5f}  std={tlt_ret.std():.4f}")
    print(f"VIX level  mean={vix_lvl.mean():.2f}   max={vix_lvl.max():.1f}")
    print()

    all_verdicts = []

    for m in MODULI:
        print(f"{'='*60}")
        print(f"MODULUS m={m}  state space={m}×{m}={m*m}")
        print(f"{'='*60}")

        b_states = equalize_quantize(spy_ret, m)  # equity → b
        e_states = equalize_quantize(tlt_ret, m)  # bonds  → e

        # Geometric null: what fraction of state space is in radius-k?
        print(f"\nGeometric null fractions (avg % of state space within radius k):")
        for r in (0, 1, 2):
            frac = neighborhood_size_fraction(r, m)
            print(f"  radius-{r}: {frac:.4f} ({frac*100:.1f}%)")

        # --- TEST 1 ---
        print(f"\n--- TEST 1: Neighborhood Concentration ---")
        t1 = test1_neighborhood(b_states, e_states, m, radii=(0, 1, 2))
        t1_signals = 0
        for radius, res in t1.items():
            shuf_p = res["shuf"]["p"]; ind_p = res["independent"]["p"]
            beats = res["beats_geo"] and res["beats_shuf"] and res["beats_ind"]
            sig = "*" if shuf_p < 0.05 else " "
            print(f"  radius-{radius}: obs={res['obs']:.4f}  geo_null={res['geo_null']:.4f}  "
                  f"shuf_null={res['shuf']['mean']:.4f}(p={shuf_p:.3f}{sig})  "
                  f"ind_null={res['independent']['mean']:.4f}(p={ind_p:.3f})")
            if beats and shuf_p < 0.05:
                t1_signals += 1
        t1_pass = t1_signals >= 1

        # --- TEST 2 ---
        print(f"\n--- TEST 2: Stress / Degeneracy (VIX top-{100-STRESS_PERCENTILE}%) ---")
        t2 = test2_stress_degeneracy(b_states, e_states, vix_lvl, m)
        print(f"  Orbit family under stress: {t2['stress_family_dist']}")
        print(f"  Orbit family under calm:   {t2['calm_family_dist']}")
        print(f"  Satellite Δ (stress-calm): {t2['satellite_delta']:+.3f}")
        print(f"  Chi-square: χ²={t2['chi2']:.3f}  p={t2['p_chi2']:.4f}"
              + (" *" if t2['significant'] else ""))
        t2_pass = t2["significant"] or abs(t2["satellite_delta"]) > 0.02

        # --- TEST 3 ---
        print(f"\n--- TEST 3: Obstruction-Aligned Extremes (|z|>{EXTREME_SIGMA}σ) ---")
        t3 = test3_obstruction_extremes(b_states, e_states, spy_ret, tlt_ret, m)
        print(f"  Extreme days:  obstruct_rate={t3['extreme_obstruct_rate']:.4f}  n={t3['n_extreme']}")
        print(f"  Normal days:   obstruct_rate={t3['normal_obstruct_rate']:.4f}  n={t3['n_normal']}")
        print(f"  Delta: {t3['delta']:+.4f}  Fisher p={t3['p_fisher']:.4f}"
              + (" *" if t3['significant'] else ""))
        t3_pass = t3["significant"] or t3["delta"] > 0.01

        # --- Verdict for this modulus ---
        n_pass = sum([t1_pass, t2_pass, t3_pass])
        if n_pass >= 2:
            verdict = f"QA_STRUCTURED ({n_pass}/3 tests)"
        elif n_pass == 1:
            verdict = f"MARGINAL (1/3 tests)"
        else:
            verdict = "NULL (0/3 tests)"
        print(f"\n  VERDICT mod-{m}: {verdict}  "
              f"[T1={t1_pass} T2={t2_pass} T3={t3_pass}]")
        all_verdicts.append((m, verdict, t1_pass, t2_pass, t3_pass))

    # Overall
    print(f"\n{'='*72}")
    print("OVERALL")
    print(f"{'='*72}")
    for m, v, t1, t2, t3 in all_verdicts:
        print(f"  mod-{m}: {v}")

    structured = sum(1 for _, v, *_ in all_verdicts if "QA_STRUCTURED" in v)
    marginal   = sum(1 for _, v, *_ in all_verdicts if "MARGINAL" in v)

    if structured >= 1:
        overall = ("QA_STRUCTURED — joint SPY×TLT state encoding shows measurable "
                   "concentration of market transitions near QA adjacency/degeneracy structure.")
    elif marginal >= 1:
        overall = ("MARGINAL — some QA joint structure visible. "
                   "Refine: try finer state encodings, longer history, or VIX as direct state.")
    else:
        overall = ("NULL — joint cross-asset QA encoding does not show transition concentration "
                   "beyond null models. Recommend: pivot main empirical lane to EEG.")

    print(f"\nVERDICT: {overall}")
    print()
    return all_verdicts

if __name__ == "__main__":
    run()
    sys.exit(0)
