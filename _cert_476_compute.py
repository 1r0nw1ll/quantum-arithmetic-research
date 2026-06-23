#!/usr/bin/env python3
"""Compute ENSO prediction numbers for cert [476].

Data: NOAA ONI monthly anomaly values 1950-present (~916 months).
Bins: rank -> floor(rank * 27 / N) in {0..26}
Tiers: T0=0-8, T1=9-17, T2=18-26

Prediction: T_t -> T_{t+k} for k=1,2,3 months.
Tests: transition matrix chi-squared, conditional persistence, k-step accuracy.
"""
import json, math, random, urllib.request

MOD = 27
SEED = 42
N_PERM = 5000


def _fetch_oni():
    """Fetch NOAA ONI 3-month running-mean anomaly values.
    Format: SEAS YR TOTAL ANOM — one row per 3-month season.
    Returns list of ANOM floats in chronological order."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=20)
    lines = resp.read().decode().splitlines()
    vals = []
    for line in lines:
        parts = line.split()
        if len(parts) != 4: continue
        try:
            float(parts[1])  # year — skips header
            vals.append(float(parts[3]))  # ANOM column
        except ValueError:
            continue
    return vals


def _to_bins(vals):
    n = len(vals)
    si = sorted(range(n), key=lambda i: vals[i])
    rk = [0] * n
    for rank, idx in enumerate(si): rk[idx] = rank
    return [int(math.floor(r * MOD / n)) for r in rk]


def _tier(b): return b // 9  # 0=T0, 1=T1, 2=T2


def _mean(xs): return sum(xs)/len(xs) if xs else 0.0


def _perm_conditional(tiers, from_tier, to_tier, lag=1):
    """Permutation test: P(T_{t+lag}=to_tier | T_t=from_tier) > 1/3."""
    n = len(tiers)
    mask = [t == from_tier for t in tiers[:-lag]]
    hits = [tiers[i+lag] == to_tier for i in range(n-lag) if mask[i]]
    if len(hits) < 5: return 1.0
    obs = _mean(hits)
    # Null: shuffle tiers, recompute
    random.seed(SEED); ct = 0
    pool = list(tiers)
    for _ in range(N_PERM):
        random.shuffle(pool)
        null_hits = [pool[i+lag] == to_tier for i in range(n-lag) if pool[i] == from_tier]
        if _mean(null_hits) >= obs: ct += 1
    return round(ct / N_PERM, 4)


def _accuracy(tiers, transition_matrix, lag=1):
    """Prediction accuracy using argmax(row) vs majority-class baseline."""
    # Most-likely-next-tier from transition matrix
    predictor = [max(range(3), key=lambda j: transition_matrix[i][j]) for i in range(3)]
    n = len(tiers)
    preds = [predictor[tiers[i]] for i in range(n-lag)]
    actual = [tiers[i+lag] for i in range(n-lag)]
    correct = sum(p == a for p, a in zip(preds, actual))
    # Baseline: always predict most common tier
    from collections import Counter
    most_common = Counter(actual).most_common(1)[0][0]
    baseline = sum(a == most_common for a in actual)
    return {
        "accuracy": round(correct / len(actual), 4),
        "baseline": round(baseline / len(actual), 4),
        "n": len(actual),
    }


def _transition_matrix(tiers, lag=1):
    """3x3 transition count matrix and conditional probabilities."""
    counts = [[0]*3 for _ in range(3)]
    n = len(tiers)
    for i in range(n-lag):
        counts[tiers[i]][tiers[i+lag]] += 1
    # Conditional probabilities P(next|cur)
    probs = []
    for row in counts:
        total = sum(row)
        probs.append([round(c/total, 4) if total > 0 else 0.0 for c in row])
    return counts, probs


if __name__ == "__main__":
    print("Fetching NOAA ONI...", flush=True)
    vals = _fetch_oni()
    print(f"  N={len(vals)} monthly values")

    bins = _to_bins(vals)
    tiers = [_tier(b) for b in bins]
    tier_counts = [tiers.count(i) for i in range(3)]
    print(f"  Tier counts: T0={tier_counts[0]} T1={tier_counts[1]} T2={tier_counts[2]}")

    for lag in [1, 2, 3]:
        print(f"\n--- lag={lag} ---")
        counts, probs = _transition_matrix(tiers, lag)
        print("  Transition matrix (probs):")
        for i, row in enumerate(probs):
            print(f"    T{i} -> T0={row[0]} T1={row[1]} T2={row[2]}")

        acc = _accuracy(tiers, counts, lag)
        print(f"  Accuracy: {acc['accuracy']} vs baseline {acc['baseline']} (n={acc['n']})")

    print("\nPermutation tests (lag=1):")
    p22 = _perm_conditional(tiers, 2, 2, lag=1)
    p00 = _perm_conditional(tiers, 0, 0, lag=1)
    p11 = _perm_conditional(tiers, 1, 1, lag=1)
    print(f"  P(T2->T2) perm_p={p22}")
    print(f"  P(T0->T0) perm_p={p00}")
    print(f"  P(T1->T1) perm_p={p11}")

    print("\nPermutation tests (lag=3):")
    p22_3 = _perm_conditional(tiers, 2, 2, lag=3)
    p00_3 = _perm_conditional(tiers, 0, 0, lag=3)
    print(f"  P(T2->T2) lag=3 perm_p={p22_3}")
    print(f"  P(T0->T0) lag=3 perm_p={p00_3}")

    # Compute key numbers for fallback
    _, probs1 = _transition_matrix(tiers, 1)
    _, probs3 = _transition_matrix(tiers, 3)

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "n": len(vals),
        "tier_counts": tier_counts,
        "lag1": {
            "T2_T2_prob": probs1[2][2],
            "T0_T0_prob": probs1[0][0],
            "T1_T1_prob": probs1[1][1],
            "T2_T2_perm_p": p22,
            "T0_T0_perm_p": p00,
            "T1_T1_perm_p": p11,
        },
        "lag3": {
            "T2_T2_prob": probs3[2][2],
            "T0_T0_prob": probs3[0][0],
            "T2_T2_perm_p": p22_3,
            "T0_T0_perm_p": p00_3,
        },
    }, indent=2))
