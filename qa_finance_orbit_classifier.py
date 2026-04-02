#!/usr/bin/env python3
"""
qa_finance_orbit_classifier.py — NT-Compliant Finance Orbit Classification
Track D | QA Finance

Research question: Does QA orbit family of the (SPY, TLT) joint market regime
predict next-day return direction at rates above the shuffle null?

Specifically: does the singularity orbit (structurally degenerate market state)
predict momentum extremes?

Architecture (Theorem NT compliant):

  [OBSERVER]  (SPY_return, TLT_return) → joint_regime ∈ {1,...,25}
                  declared a priori as 5×5 quintile grid — floats confined here
                  ↓ (boundary crossed once)
  [QA LAYER]  joint_regime → (b,e) from REGIME_STATES (int lookup)
                  → f(b,e) = b*b + b*e - e*e  (integer norm)
                  → orbit_family ∈ {singularity, satellite, cosmos}
                  ↓ (boundary crossed once)
  [PROJECTION] orbit sequence → next-day return comparison → null model test

Axiom compliance:
  A1: (b,e) from REGIME_STATES — all values in {1,...,MODULUS}, no zeros
  A2: coord_d = b+e, coord_a = b+2*e — derived only, not assigned directly
  T1: QA time = path step count k (integer)
  T2: continuous returns enter ONLY at observer boundary
  S1: b*b throughout, never b-squared
  S2: b,e are Python int in QA layer
"""

import numpy as np
import json
from pathlib import Path
from typing import NamedTuple

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

from qa_orbit_rules import norm_f, v3, orbit_family, qa_step

# ── QA_COMPLIANCE declaration ──────────────────────────────────────────────────

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "cert_family": "Track D finance — joint market regime orbit",
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "observer": "quintile_rank_joint_regime_classifier",
    "state_alphabet": "5x5 quintile grid → 25 joint regimes",
    "qa_layer_types": "int",
    "projection_types": "float",
}

# ── Discrete state alphabet ────────────────────────────────────────────────────
#
# State space: 5 SPY quintiles × 5 TLT quintiles = 25 joint market regimes.
# Quintiles declared a priori: Q1 (most bearish) through Q5 (most bullish).
#
# (b, e) assignment: SPY quintile → b, TLT quintile → e.
# Map quintile {1,2,3,4,5} → values spread across {1,...,24} to use
# the full orbit structure of mod-24 QA.
#
# Declared mapping (fixed — not derived from any signal):
#   Quintile 1 → 3   (low end — lower cosmos address)
#   Quintile 2 → 7
#   Quintile 3 → 12  (middle)
#   Quintile 4 → 18
#   Quintile 5 → 22  (high end — upper cosmos address)
#
# A1: all values in {1,...,24}. No zeros.

MODULUS = 24

QUINTILE_TO_STATE: dict[int, int] = {
    # Redesigned to activate full orbit structure (verified by orbit audit):
    # Q1=8, Q2=16, Q5=24 → multiples of 8 → satellite when crossed
    # (Q5, Q5) = (24,24) → singularity (unique fixed point)
    # Q3=12, Q4=18 → cosmos (not multiples of 8)
    1:  8,   # most bearish  → satellite-capable b/e value
    2: 16,   # bearish       → satellite-capable b/e value
    3: 12,   # neutral       → cosmos
    4: 18,   # bullish       → cosmos
    5: 24,   # most bullish  → singularity when b=e=24; satellite when paired with Q1/Q2
}

N_QUINTILES = 5
RNG_SEED = 42
N_PERMUTATIONS = 1000


def _assert_states_valid() -> None:
    """A1 gate: verify all declared state values are in {1,...,MODULUS}."""
    for q, v in QUINTILE_TO_STATE.items():
        assert 1 <= v <= MODULUS, f"Quintile {q}: value={v} violates A1"


# ── QA layer — integers only past this line ───────────────────────────────────

def regime_to_qa(spy_q: int, tlt_q: int) -> tuple[int, int]:
    """QA layer entry: map (SPY_quintile, TLT_quintile) → (b,e) integers."""
    b = int(QUINTILE_TO_STATE[spy_q])
    e = int(QUINTILE_TO_STATE[tlt_q])
    return b, e


def compute_orbit_sequence(spy_quintiles: list[int],
                           tlt_quintiles: list[int]) -> list[str]:
    """
    QA layer: convert paired quintile sequences → orbit family sequence.
    Integer arithmetic only — no floats here.
    """
    assert len(spy_quintiles) == len(tlt_quintiles)
    orbits = []
    for sq, tq in zip(spy_quintiles, tlt_quintiles):
        b, e = regime_to_qa(sq, tq)
        orbits.append(orbit_family(b, e))
    return orbits


def orbit_counts(orbits: list[str]) -> dict[str, int]:
    """QA layer: count orbit families (integer counts)."""
    counts: dict[str, int] = {"singularity": 0, "satellite": 0, "cosmos": 0}
    for o in orbits:
        if o in counts:
            counts[o] += 1
    return counts


# ── Observer layer — floats permitted ─────────────────────────────────────────

def compute_quintile_ranks(returns: np.ndarray) -> list[int]:
    """
    Observer: map continuous return series → quintile rank sequence {1,...,5}.

    Uses expanding-window quintile rank (no look-ahead: each day's quintile
    is computed relative to all returns up to that day).

    Returns list of Python int in {1,...,5}. Floats confined here.
    """
    n = len(returns)
    quintiles = []
    min_history = 20  # need at least 20 days to estimate quintiles reliably

    for i in range(n):
        if i < min_history:
            # Not enough history: assign middle quintile
            quintiles.append(3)
        else:
            history = returns[:i]
            boundaries = np.percentile(history, [20, 40, 60, 80])
            r = float(returns[i])
            if r <= boundaries[0]:
                q = 1
            elif r <= boundaries[1]:
                q = 2
            elif r <= boundaries[2]:
                q = 3
            elif r <= boundaries[3]:
                q = 4
            else:
                q = 5
            quintiles.append(q)

    return quintiles


def fetch_returns(tickers: list[str],
                  start: str = "2015-01-01",
                  end: str = "2024-01-01") -> dict[str, np.ndarray] | None:
    """
    Observer: fetch log returns for tickers. Floats permitted here.
    Returns dict of ticker → numpy array of log returns.
    """
    if not HAS_YF:
        return None

    import pandas as pd
    import warnings

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
    log_ret = np.log(frame / frame.shift(1)).dropna()

    return {t: log_ret[t].values for t in tickers if t in log_ret.columns}


def make_synthetic_returns(n: int = 2000,
                           seed: int = RNG_SEED) -> dict[str, np.ndarray]:
    """
    Observer (synthetic): generate correlated log return series for testing.
    Uses multivariate normal — floats confined here.
    """
    rng = np.random.default_rng(seed)
    # SPY-like: daily vol ~1%, slight positive drift
    # TLT-like: daily vol ~0.8%, slight negative correlation to SPY
    cov = np.array([[0.0001, -0.00005],
                    [-0.00005, 0.000064]])
    mean = np.array([0.0003, -0.0001])
    samples = rng.multivariate_normal(mean, cov, size=n)
    return {"SPY": samples[:, 0], "TLT": samples[:, 1]}


# ── Projection layer — floats permitted ───────────────────────────────────────

def orbit_fractions(orbits: list[str]) -> dict[str, float]:
    """Projection: orbit counts → fractions."""
    counts = orbit_counts(orbits)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def next_day_return_by_orbit(orbits: list[str],
                              spy_returns: np.ndarray) -> dict[str, dict]:
    """
    Projection: for each orbit family, compute next-day SPY return statistics.

    This is the core QA claim: orbit family at day t predicts SPY direction at t+1.
    """
    by_orbit: dict[str, list[float]] = {
        "singularity": [], "satellite": [], "cosmos": []
    }

    for i in range(len(orbits) - 1):
        orb = orbits[i]
        if orb in by_orbit:
            by_orbit[orb].append(float(spy_returns[i + 1]))

    result = {}
    for orb, returns in by_orbit.items():
        if len(returns) < 5:
            result[orb] = {"n": len(returns), "mean": None, "pos_rate": None}
            continue
        arr = np.array(returns)
        result[orb] = {
            "n": len(returns),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "pos_rate": float(np.mean(arr > 0)),
        }
    return result


def shuffle_null(spy_q: list[int], tlt_q: list[int],
                 spy_returns: np.ndarray,
                 n_perm: int = N_PERMUTATIONS,
                 seed: int = RNG_SEED) -> dict[str, list[float]]:
    """
    Projection: permutation null model.
    Shuffle (spy_q, tlt_q) pairs to break temporal structure.
    Returns distribution of pos_rate for each orbit under the null.
    """
    rng = np.random.default_rng(seed)
    null_pos_rates: dict[str, list[float]] = {
        "singularity": [], "satellite": [], "cosmos": []
    }

    pairs = list(zip(spy_q, tlt_q))
    for _ in range(n_perm):
        idx = rng.permutation(len(pairs))
        shuffled_spy_q = [pairs[i][0] for i in idx]
        shuffled_tlt_q = [pairs[i][1] for i in idx]
        orbits = compute_orbit_sequence(shuffled_spy_q, shuffled_tlt_q)
        stats = next_day_return_by_orbit(orbits, spy_returns)
        for orb in null_pos_rates:
            if stats[orb]["pos_rate"] is not None:
                null_pos_rates[orb].append(stats[orb]["pos_rate"])

    return null_pos_rates


def empirical_p_value(observed: float, null_distribution: list[float],
                      alternative: str = "greater") -> float:
    """Projection: empirical p-value from permutation null."""
    if not null_distribution:
        return float("nan")
    null = np.array(null_distribution)
    if alternative == "greater":
        return float(np.mean(null >= observed))
    elif alternative == "less":
        return float(np.mean(null <= observed))
    else:
        return float(np.mean(np.abs(null - np.mean(null)) >= abs(observed - np.mean(null))))


# ── Main experiment ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("QA FINANCE ORBIT CLASSIFIER — NT-Compliant Track D")
    print(f"Observer: {QA_COMPLIANCE['observer']}")
    print("=" * 70)
    print()

    # A1 gate at startup
    _assert_states_valid()

    # Print state alphabet and orbit assignments
    print("Declared state alphabet (A1 verified):")
    for q in range(1, N_QUINTILES + 1):
        b_val = QUINTILE_TO_STATE[q]
        print(f"  SPY/TLT quintile {q} → state value {b_val:2d}")
    print()
    print("Joint regime → orbit family (SPY_q, TLT_q):")
    for sq in range(1, N_QUINTILES + 1):
        for tq in range(1, N_QUINTILES + 1):
            b, e = regime_to_qa(sq, tq)
            orb = orbit_family(b, e)
            if orb == "singularity":
                print(f"  SPY_q={sq}, TLT_q={tq}  b={b:2d} e={e:2d}  → {orb} ***")
    print()

    # Fetch or generate returns (observer layer)
    print("Fetching market data (observer layer)...")
    returns = None
    if HAS_YF:
        returns = fetch_returns(["SPY", "TLT"])

    if returns is None:
        print("  yfinance unavailable — using synthetic returns")
        returns = make_synthetic_returns(n=2000)
    else:
        print(f"  SPY: {len(returns['SPY'])} days, TLT: {len(returns['TLT'])} days")

    # Align lengths
    n = min(len(returns["SPY"]), len(returns["TLT"]))
    spy_ret = returns["SPY"][:n]
    tlt_ret = returns["TLT"][:n]
    print(f"  Aligned: {n} trading days")
    print()

    # Observer: continuous returns → quintile sequences
    print("Observer: classifying returns into quintile regimes...")
    spy_q = compute_quintile_ranks(spy_ret)
    tlt_q = compute_quintile_ranks(tlt_ret)
    print(f"  SPY quintile distribution: {dict(zip(*np.unique(spy_q, return_counts=True)))}")
    print(f"  TLT quintile distribution: {dict(zip(*np.unique(tlt_q, return_counts=True)))}")
    print()

    # QA layer: quintile pairs → orbit sequence
    print("QA layer: computing orbit sequence...")
    orbits = compute_orbit_sequence(spy_q, tlt_q)
    counts = orbit_counts(orbits)
    fracs = orbit_fractions(orbits)
    print(f"  Orbit distribution:")
    for orb in ("singularity", "satellite", "cosmos"):
        print(f"    {orb:15s}  n={counts[orb]:4d}  ({fracs[orb]*100:.1f}%)")
    print()

    # Projection: next-day return by orbit
    print("Projection: next-day SPY return by orbit family...")
    ndr = next_day_return_by_orbit(orbits, spy_ret)
    for orb in ("singularity", "satellite", "cosmos"):
        s = ndr[orb]
        if s["pos_rate"] is None:
            print(f"  {orb:15s}  insufficient data")
        else:
            print(f"  {orb:15s}  n={s['n']:4d}  pos_rate={s['pos_rate']:.3f}  "
                  f"mean_ret={s['mean']:+.5f}")
    print()

    # Projection: shuffle null model
    print(f"Running shuffle null model ({N_PERMUTATIONS} permutations)...")
    null_dists = shuffle_null(spy_q, tlt_q, spy_ret)
    print()
    print("Null model comparison (projection layer):")
    results = {}
    for orb in ("singularity", "satellite", "cosmos"):
        obs = ndr[orb].get("pos_rate")
        null = null_dists[orb]
        if obs is None or not null:
            print(f"  {orb:15s}  insufficient data")
            continue
        p_val = empirical_p_value(obs, null, alternative="greater")
        null_mean = float(np.mean(null))
        direction = "ABOVE NULL" if obs > null_mean else "BELOW NULL"
        print(f"  {orb:15s}  obs={obs:.3f}  null_mean={null_mean:.3f}  "
              f"p={p_val:.4f}  ({direction})")
        results[orb] = {"observed_pos_rate": obs, "null_mean": null_mean, "p_value": p_val}
    print()

    # Verdict
    print("=" * 70)
    sing_result = results.get("singularity", {})
    if sing_result:
        p = sing_result.get("p_value", 1.0)
        obs = sing_result.get("observed_pos_rate", 0.5)
        null_m = sing_result.get("null_mean", 0.5)
        if p < 0.05 and obs > null_m:
            print("VERDICT: SINGULARITY orbit → above-null next-day return rate (p<0.05)")
        elif p < 0.1:
            print("VERDICT: MARGINAL signal in singularity orbit (p<0.10)")
        else:
            print("VERDICT: NULL — no significant orbit-return relationship found")
    print()

    # Save results
    output = {
        "qa_compliance": QA_COMPLIANCE,
        "quintile_to_state": QUINTILE_TO_STATE,
        "n_days": n,
        "orbit_distribution": {k: {"n": counts[k], "frac": fracs[k]}
                                for k in counts},
        "next_day_return_by_orbit": ndr,
        "null_model_results": results,
    }
    out_path = Path("qa_finance_orbit_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
