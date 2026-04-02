#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
# SUPERSEDED — QA-noncompliant: orbit_family_mod uses (b+e)%m qa_step (A1: allows 0)
# and classifies by period heuristic instead of canonical (m//3)|b AND (m//3)|e rule.
# Use qa_finance_orbit_classifier.py instead.
"""
qa_finance_transition_structure.py
====================================
QA Finance Transition Structure Experiment.

Scientific question
-------------------
Do observed market state transitions concentrate along QA-legal (orbit-following)
transitions more than appropriate null models preserving autocorrelation?

This is a fundamentally different question from "does QA predict alpha."
It asks: is the geometry of market dynamics QA-structured?

State encoding
--------------
Daily log-return r_t is quantized into a QA state b_t ∈ {0..m-1}.
Consecutive states (b_{t-1}, b_t) form QA pair (b, e).
The next pair is (b_t, b_{t+1}).

A "QA-legal transition" is:
    (b_t, b_{t+1}) = T(b_{t-1}, b_t) = (b_t, (b_{t-1} + b_t) % m)
i.e., the orbit-following step.

Hypothesis A: QA-legal transition rate > 1/m (chance)
Hypothesis B: QA-legal rate in market > QA-legal rate in matched nulls
Hypothesis C: Satellite/singularity states concentrate during high-vol regimes
Hypothesis D: Transition matrix concentrates near QA-adjacency more than null

Null models
-----------
1. shuffle_null     — shuffle return sequence (destroys all structure)
2. phase_rand_null  — phase-randomize returns (preserves spectrum, kills order)
3. ar1_null         — AR(1) fit to returns (same AC, no QA structure)
4. block_shuffle    — shuffle in 5-day blocks (preserves short-term AC)

Assets tested: SPY, QQQ, GLD, TLT, BTC-USD (if available)
Moduli tested: m=9, m=24 (both documented in literature)

Output
------
Per-asset, per-modulus table of:
  qa_legal_rate vs null rates + p-value (permutation test)
  orbit family distribution during high-vol vs low-vol regimes
  failure algebra: which transitions are structurally obstructed?

VERDICT: QA_STRUCTURED if qa_legal_rate > all nulls with p < 0.05
         MARGINAL if qa_legal_rate > nulls but p >= 0.05
         NULL if not distinguishable from nulls
"""

import numpy as np
from collections import Counter, defaultdict
from scipy import stats
import sys

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

MODULI = [9, 24]
N_PERMUTATIONS = 1000
RNG_SEED = 42
BLOCK_SIZE = 5  # days for block shuffle

ASSETS = ["SPY", "QQQ", "GLD", "TLT"]
BTC_ASSET = "BTC-USD"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_returns(ticker: str, start="2015-01-01", end="2024-01-01") -> np.ndarray:
    """Fetch daily log returns. Returns array or None on failure."""
    if not HAS_YF:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 100:
            return None
        close = df["Close"].values.flatten()
        close = close[~np.isnan(close)]
        if len(close) < 100:
            return None
        returns = np.diff(np.log(close + 1e-10))
        return returns
    except Exception:
        return None

def synthetic_returns(n=2000, alpha=0.05, vol=0.01, seed=RNG_SEED) -> np.ndarray:
    """AR(1) log-return series — used as fallback if yfinance unavailable."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = alpha * r[i-1] + vol * rng.standard_normal()
    return r

# ---------------------------------------------------------------------------
# QA encoding
# ---------------------------------------------------------------------------

def equalize_quantize(returns: np.ndarray, m: int) -> np.ndarray:
    """Histogram-equalized quantization → uniform distribution over {0..m-1}."""
    n = len(returns)
    ranks = np.argsort(np.argsort(returns))
    states = (ranks * m // n).astype(int)
    return np.clip(states, 0, m - 1)

def qa_legal_transition(b: int, e: int, b_next: int, e_next: int, m: int) -> bool:
    """Is (b,e)→(b_next,e_next) a QA-legal orbit-following step?"""
    t_b, t_e = e, (b + e) % m
    return b_next == t_b and e_next == t_e

def orbit_family_mod(b: int, e: int, m: int) -> str:
    """Classify (b,e) orbit family via orbit length."""
    seen = set()
    cb, ce = b, e
    for _ in range(m * m + 1):
        if (cb, ce) in seen:
            break
        seen.add((cb, ce))
        cb, ce = ce, (cb + ce) % m
    length = len(seen)
    if length == 1:
        return "singularity"
    max_len = max(24, m * 2)  # cosmos orbits are longest
    if length >= max_len - 2:
        return "cosmos"
    return "satellite"

def precompute_families(m: int) -> dict:
    families = {}
    for b in range(m):
        for e in range(m):
            families[(b, e)] = orbit_family_mod(b, e, m)
    return families

# ---------------------------------------------------------------------------
# QA-legal transition rate
# ---------------------------------------------------------------------------

def qa_legal_rate(states: np.ndarray, m: int) -> float:
    """Fraction of consecutive (b,e)→(b',e') pairs that are QA-legal."""
    n = len(states)
    if n < 3:
        return 0.0
    b  = states[:-2]
    e  = states[1:-1]
    bp = states[1:-1]
    ep = states[2:]
    legal = np.sum((bp == e) & (ep == (b + e) % m))
    return float(legal) / (n - 2)

# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

def shuffle_null_rate(states: np.ndarray, m: int, n_perm: int = N_PERMUTATIONS,
                      seed: int = RNG_SEED) -> tuple:
    """Permutation null: shuffle states, compute mean and std of QA-legal rate."""
    rng = np.random.default_rng(seed)
    rates = []
    s = states.copy()
    for _ in range(n_perm):
        rng.shuffle(s)
        rates.append(qa_legal_rate(s, m))
    return float(np.mean(rates)), float(np.std(rates))

def phase_rand_rate(returns: np.ndarray, m: int, n_perm: int = 200,
                    seed: int = RNG_SEED) -> tuple:
    """Phase-randomized null: preserves power spectrum, destroys order."""
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_perm):
        fft = np.fft.rfft(returns)
        phases = rng.uniform(0, 2 * np.pi, len(fft))
        fft_rand = np.abs(fft) * np.exp(1j * phases)
        r_rand = np.fft.irfft(fft_rand, n=len(returns))
        states_rand = equalize_quantize(r_rand, m)
        rates.append(qa_legal_rate(states_rand, m))
    return float(np.mean(rates)), float(np.std(rates))

def ar1_null_rate(returns: np.ndarray, m: int, n_perm: int = 200,
                  seed: int = RNG_SEED) -> tuple:
    """AR(1) null: same lag-1 AC, no QA structure."""
    rng = np.random.default_rng(seed)
    alpha = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    vol = float(np.std(returns)) * np.sqrt(1 - alpha * alpha)
    rates = []
    n = len(returns)
    for _ in range(n_perm):
        r = np.zeros(n)
        r[0] = rng.standard_normal() * np.std(returns)
        for i in range(1, n):
            r[i] = alpha * r[i-1] + vol * rng.standard_normal()
        states = equalize_quantize(r, m)
        rates.append(qa_legal_rate(states, m))
    return float(np.mean(rates)), float(np.std(rates))

def block_shuffle_rate(states: np.ndarray, m: int, n_perm: int = 200,
                       block: int = BLOCK_SIZE, seed: int = RNG_SEED) -> tuple:
    """Block-shuffle null: preserves short-term AC within blocks."""
    rng = np.random.default_rng(seed)
    n = len(states)
    n_blocks = n // block
    rates = []
    for _ in range(n_perm):
        blocks = [states[i*block:(i+1)*block] for i in range(n_blocks)]
        rng.shuffle(blocks)
        shuffled = np.concatenate(blocks)
        rates.append(qa_legal_rate(shuffled, m))
    return float(np.mean(rates)), float(np.std(rates))

def permutation_pvalue(observed: float, null_mean: float, null_std: float) -> float:
    """One-sided p-value: P(null >= observed) assuming Normal null distribution."""
    if null_std < 1e-10:
        return 0.0 if observed > null_mean else 1.0
    z = (observed - null_mean) / null_std
    return float(stats.norm.sf(z))  # one-sided: P(Z >= z)

# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------

def regime_orbit_analysis(returns: np.ndarray, states: np.ndarray,
                           families: dict, m: int) -> dict:
    """
    Compare orbit family distribution in high-vol vs low-vol regimes.
    High-vol = top tercile of 20-day rolling volatility.
    """
    n = len(returns)
    roll_vol = np.array([np.std(returns[max(0,i-20):i+1]) for i in range(n)])
    vol_threshold = np.percentile(roll_vol, 67)

    high_vol_idx = np.where(roll_vol > vol_threshold)[0]
    low_vol_idx  = np.where(roll_vol <= vol_threshold)[0]

    def family_dist(indices):
        dist = Counter()
        for i in indices:
            if i + 1 < len(states):
                b, e = states[i], states[i+1]
                if (b, e) in families:
                    dist[families[(b, e)]] += 1
        total = sum(dist.values())
        return {k: round(v/total, 3) for k, v in dist.items()} if total else {}

    return {
        "high_vol": family_dist(high_vol_idx),
        "low_vol":  family_dist(low_vol_idx),
        "vol_threshold": round(float(vol_threshold), 5),
        "n_high": len(high_vol_idx),
        "n_low":  len(low_vol_idx),
    }

# ---------------------------------------------------------------------------
# Failure algebra: obstructed transitions
# ---------------------------------------------------------------------------

def failure_algebra_analysis(states: np.ndarray, m: int) -> dict:
    """
    Which transitions are structurally obstructed (inert prime divides norm)?
    Compare observed frequency of obstructed vs clear transitions.
    """
    INERT = {9: [3], 24: [3, 7]}.get(m, [])

    def v_p(n_val: int, p: int) -> int:
        if n_val == 0:
            return 999
        count = 0
        n_val = abs(n_val)
        while n_val % p == 0:
            count += 1; n_val //= p
        return count

    def is_obstructed(b: int, e: int) -> bool:
        r = b * b + b * e - e * e
        return any(v_p(r, p) == 1 for p in INERT)

    n = len(states)
    obstructed_count = 0
    clear_count = 0
    for i in range(n - 1):
        b, e = int(states[i]), int(states[i+1])
        if is_obstructed(b, e):
            obstructed_count += 1
        else:
            clear_count += 1

    total = obstructed_count + clear_count
    return {
        "obstructed_rate": round(obstructed_count / total, 4) if total else 0.0,
        "clear_rate":      round(clear_count / total, 4) if total else 0.0,
        "inert_primes":    INERT,
    }

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_asset(name: str, returns: np.ndarray, m: int) -> dict:
    chance = 1.0 / m
    states = equalize_quantize(returns, m)
    families = precompute_families(m)

    obs_rate = qa_legal_rate(states, m)

    # Nulls (reduced permutations for speed)
    shuf_mean, shuf_std   = shuffle_null_rate(states, m, n_perm=500)
    phase_mean, phase_std = phase_rand_rate(returns, m, n_perm=100)
    ar1_mean, ar1_std     = ar1_null_rate(returns, m, n_perm=100)
    block_mean, block_std = block_shuffle_rate(states, m, n_perm=100)

    p_shuf  = permutation_pvalue(obs_rate, shuf_mean, shuf_std)
    p_phase = permutation_pvalue(obs_rate, phase_mean, phase_std)
    p_ar1   = permutation_pvalue(obs_rate, ar1_mean, ar1_std)
    p_block = permutation_pvalue(obs_rate, block_mean, block_std)

    # Regime analysis
    regime = regime_orbit_analysis(returns, states, families, m)

    # Failure algebra
    fail = failure_algebra_analysis(states, m)

    # Verdict
    beats_all_nulls = (obs_rate > shuf_mean and obs_rate > phase_mean
                       and obs_rate > ar1_mean and obs_rate > block_mean)
    sig_vs_shuf = p_shuf < 0.05
    sig_vs_ar1  = p_ar1  < 0.05

    if beats_all_nulls and sig_vs_shuf and sig_vs_ar1:
        verdict = "QA_STRUCTURED"
    elif beats_all_nulls:
        verdict = "MARGINAL"
    elif obs_rate > chance:
        verdict = "ABOVE_CHANCE_ONLY"
    else:
        verdict = "NULL"

    return {
        "asset": name, "m": m, "n": len(returns),
        "obs_rate": round(obs_rate, 4),
        "chance": round(chance, 4),
        "nulls": {
            "shuffle":      {"mean": round(shuf_mean,4),  "p": round(p_shuf,4)},
            "phase_rand":   {"mean": round(phase_mean,4), "p": round(p_phase,4)},
            "ar1":          {"mean": round(ar1_mean,4),   "p": round(p_ar1,4)},
            "block_shuffle":{"mean": round(block_mean,4), "p": round(p_block,4)},
        },
        "regime": regime,
        "failure_algebra": fail,
        "verdict": verdict,
    }

def print_result(r: dict) -> None:
    m = r["m"]
    print(f"\n  {r['asset']} mod-{m}  (n={r['n']}):")
    print(f"    QA-legal rate: {r['obs_rate']:.4f}  chance={r['chance']:.4f}  "
          f"excess={r['obs_rate']-r['chance']:+.4f}")
    nulls = r["nulls"]
    for null_name, v in nulls.items():
        sig = "*" if v["p"] < 0.05 else " "
        print(f"    vs {null_name:<16}: null={v['mean']:.4f}  p={v['p']:.4f} {sig}")
    reg = r["regime"]
    hv = reg["high_vol"]
    lv = reg["low_vol"]
    sat_hi = hv.get("satellite", 0.0)
    sat_lo = lv.get("satellite", 0.0)
    print(f"    Satellite rate: high-vol={sat_hi:.3f}  low-vol={sat_lo:.3f}  "
          f"Δ={sat_hi-sat_lo:+.3f}")
    fa = r["failure_algebra"]
    print(f"    Obstructed transitions: {fa['obstructed_rate']:.3f}  "
          f"inert_primes={fa['inert_primes']}")
    print(f"    VERDICT: {r['verdict']}")

def run():
    print("=" * 72)
    print("QA Finance Transition Structure")
    print("=" * 72)
    print()

    all_results = []

    for ticker in ASSETS:
        print(f"Fetching {ticker}...", end=" ", flush=True)
        returns = fetch_returns(ticker)
        if returns is None:
            print("FAILED — using synthetic AR(1) fallback")
            returns = synthetic_returns()
        else:
            print(f"OK ({len(returns)} daily returns)")

        for m in MODULI:
            r = analyze_asset(ticker, returns, m)
            all_results.append(r)
            print_result(r)

    # Cross-asset summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for m in MODULI:
        mod_results = [r for r in all_results if r["m"] == m]
        structured = sum(1 for r in mod_results if r["verdict"] == "QA_STRUCTURED")
        marginal   = sum(1 for r in mod_results if r["verdict"] == "MARGINAL")
        null_count = sum(1 for r in mod_results if r["verdict"] in ("NULL","ABOVE_CHANCE_ONLY"))
        mean_excess = np.mean([r["obs_rate"] - r["chance"] for r in mod_results])
        print(f"  mod-{m}: QA_STRUCTURED={structured}  MARGINAL={marginal}  "
              f"NULL/ABOVE_CHANCE={null_count}  mean_excess={mean_excess:+.4f}")

    # Overall verdict
    total_structured = sum(1 for r in all_results if r["verdict"] == "QA_STRUCTURED")
    total = len(all_results)

    if total_structured >= total // 2:
        overall = (f"QA_STRUCTURED — {total_structured}/{total} asset×modulus pairs "
                   f"show QA-legal transition rate significantly above all null models.")
    elif any(r["verdict"] in ("QA_STRUCTURED","MARGINAL") for r in all_results):
        overall = (f"MARGINAL — some evidence of QA transition structure but not robust "
                   f"across all assets/moduli. Investigate strongest cases.")
    else:
        overall = ("NULL — market transitions do not concentrate along QA-legal paths "
                   "beyond null models. Finance track needs different state encoding.")

    print(f"\nVERDICT: {overall}")
    print()
    return all_results

if __name__ == "__main__":
    run()
    sys.exit(0)
