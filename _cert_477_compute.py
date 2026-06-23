#!/usr/bin/env python3
"""Compute: k-class MI ceiling formula [477]
Derives formula, verifies k=1 reduction, computes ENSO 3-class, shows ceiling lift."""
import math, random, urllib.request

SEED = 42
N_PERM = 5000
MOD = 27

# ── Binary domains from cert [468] ──────────────────────────────────────────
BINARY_DOMAINS = [
    ("Geomagnetic_storm",    17,  1464, 17, 0.204),  # (name, n_event, n_total, n_dom, obs_ratio)
    ("ECG_VFL",              19,   185, 19, 0.382),
    ("EEG_spectral_entropy", 24,   224, 24, 0.384),
    ("EEG_seizure_energy",   29,   228, 29, 0.418),
    ("Seismic_aftershock",   28,   168, 28, 0.487),
    ("SEP_solar",            60,   204, 58, 0.697),
]

# ── Shannon entropy helpers ──────────────────────────────────────────────────
def _xlogy(x):
    return x * math.log2(x) if x > 1e-300 else 0.0

def _h_entropy(probs):
    return -sum(_xlogy(p) for p in probs if p > 0)

def _h_r(r):
    """h(r) = -r*log2(r) - (1-r)*log2((1-r)/2); entropy of dominant/other split."""
    if r >= 1.0 - 1e-12:
        return 0.0
    p_oth = (1.0 - r) / 2.0
    return -_xlogy(r) - 2 * _xlogy(p_oth)

# ── Binary formula: cert [468] original ─────────────────────────────────────
def _mi_formula_binary(q: float, r: float) -> float:
    """Exact formula from cert [468]."""
    p_dom_ev  = q * r
    p_dom_nev = 1.0/3 - p_dom_ev
    p_oth_ev  = q * (1-r) / 2
    p_oth_nev = 1.0/3 - p_oth_ev
    p_ev  = q
    p_nev = 1.0 - q
    p_t   = 1.0/3

    def _term(p_ty, pt, py):
        if p_ty <= 0: return 0.0
        return p_ty * math.log2(p_ty / (pt * py))

    return (
        _term(p_dom_ev,  p_t, p_ev)  + _term(p_dom_nev, p_t, p_nev) +
        2*_term(p_oth_ev, p_t, p_ev) + 2*_term(p_oth_nev, p_t, p_nev)
    )

def _mi_ratio_binary(q, r):
    mi = _mi_formula_binary(q, r)
    hy = _h_entropy([q, 1-q])
    return mi / hy if hy > 1e-12 else 0.0

# ── General k-class formula (rank-uniform 3-tier) ───────────────────────────
def _mi_k_class(classes):
    """
    MI_ratio for k named classes with rank-uniform P(tier)=1/3 constraint.

    classes: list of (q_i, r_i, d_i) where
      q_i = P(class i)  [sum <= 1; remainder = background]
      r_i = P(tier_d_i | class i)  [dominant tier fraction]
      d_i = dominant tier index in {0,1,2}

    Returns (mi, mi_ratio)
    """
    k = len(classes)
    q_bg = 1.0 - sum(c[0] for c in classes)
    assert q_bg >= -1e-9

    # Build joint P(tier t, class y) exploiting rank-uniform P(t)=1/3
    # For each signal class i with dominant tier d_i:
    #   P(d_i, i) = q_i * r_i
    #   P(t≠d_i, i) = q_i * (1-r_i)/2
    # Background: P(t, bg) = 1/3 - sum_{i: d_i=t} q_i*r_i - sum_{i: d_i≠t} q_i*(1-r_i)/2

    # Joint probabilities per (tier, class)
    joint = {}
    for i, (q, r, d) in enumerate(classes):
        joint[(d, i)] = q * r
        for t in range(3):
            if t != d:
                joint[(t, i)] = q * (1-r)/2
        # background handled below
    for t in range(3):
        # marginal P(t) = 1/3; subtract signal contributions
        bg_val = 1.0/3 - sum(joint.get((t, i), 0.0) for i in range(k))
        joint[(t, k)] = max(bg_val, 0.0)  # background class index = k

    # Class marginals
    p_class = {i: classes[i][0] for i in range(k)}
    p_class[k] = max(q_bg, 0.0)

    # Tier marginals (should be 1/3 by construction)
    p_tier = {t: sum(joint.get((t, y), 0.0) for y in range(k+1)) for t in range(3)}

    # MI = sum P(t,y) log P(t,y)/(P(t)*P(y))
    mi = 0.0
    for t in range(3):
        for y in range(k+1):
            p_ty = joint.get((t, y), 0.0)
            if p_ty > 1e-300:
                denom = p_tier[t] * p_class[y]
                if denom > 1e-300:
                    mi += p_ty * math.log2(p_ty / denom)

    # H(Y)
    all_classes = list(range(k+1))
    hy = _h_entropy([p_class[y] for y in all_classes])

    mi_ratio = mi / hy if hy > 1e-12 else 0.0
    return mi, mi_ratio

# ── Compact formula (pure MI, rank-uniform) ──────────────────────────────────
def _mi_formula_compact(classes):
    """
    For rank-uniform P(t)=1/3 and background, MI simplifies:
    MI = H(T) - H(T|Y)
       = log2(3) - [sum_i q_i * h_r(r_i) + q_bg * log2(3)]
       = sum_i q_i * (log2(3) - h_r(r_i))
    """
    return sum(q * (math.log2(3) - _h_r(r)) for q, r, _ in classes)

# ── Verify k=1 reduction against cert [468] ──────────────────────────────────
def verify_k1_reduction():
    print("=== k=1 reduction verification (should be ≡ cert [468]) ===")
    max_err = 0.0
    for name, n_ev, n_tot, n_dom, obs_ratio in BINARY_DOMAINS:
        q = n_ev / n_tot
        r = n_dom / n_ev if n_ev > 0 else 1.0
        ratio_468 = _mi_ratio_binary(q, r)

        # k=1 general formula
        _, ratio_k1 = _mi_k_class([(q, r, 0)])
        err = abs(ratio_468 - ratio_k1)
        max_err = max(max_err, err)
        print(f"  {name}: binary={ratio_468:.6f}  k1={ratio_k1:.6f}  diff={err:.2e}")
    print(f"  Max diff: {max_err:.2e}  (PASS if < 1e-6: {max_err < 1e-6})")
    return max_err < 1e-6

# ── Compact formula vs exact (numerical check) ───────────────────────────────
def verify_compact_formula():
    print("\n=== Compact formula check ===")
    max_err = 0.0
    for name, n_ev, n_tot, n_dom, obs_ratio in BINARY_DOMAINS:
        q = n_ev / n_tot
        r = n_dom / n_ev if n_ev > 0 else 1.0
        mi_exact, _ = _mi_k_class([(q, r, 0)])
        mi_compact  = _mi_formula_compact([(q, r, 0)])
        err = abs(mi_exact - mi_compact)
        max_err = max(max_err, err)
        print(f"  {name}: exact={mi_exact:.6f}  compact={mi_compact:.6f}  diff={err:.2e}")
    print(f"  Max diff: {max_err:.2e}  (PASS if < 1e-6: {max_err < 1e-6})")
    return max_err < 1e-6

# ── k=3 symmetric ceiling analysis ──────────────────────────────────────────
def k3_ceiling():
    print("\n=== k=3 symmetric balanced ceiling ===")
    print("  At q_i=1/3 each (no background), r→1: MI_ratio should → 1.0")
    q = 1.0/3
    # Each class occupies a distinct dominant tier
    for r in [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]:
        classes = [(q, r, 0), (q, r, 1), (q, r, 2)]
        mi, ratio = _mi_k_class(classes)
        compact = _mi_formula_compact(classes)
        hy = math.log2(3)  # H(Y) = log2(3) for balanced 3-class
        ratio_compact = compact / hy
        print(f"  r={r:.3f}: exact={ratio:.4f}  compact={ratio_compact:.4f}")
    # Confirm ceiling at r=1
    r = 1.0 - 1e-10
    classes = [(q, r, 0), (q, r, 1), (q, r, 2)]
    _, ratio = _mi_k_class(classes)
    print(f"  r→1:    exact={ratio:.6f}  (should be ~1.000)")
    return abs(ratio - 1.0) < 0.01

# ── k=1 binary ceiling for comparison ─────────────────────────────────────────
def k1_ceiling():
    print("\n=== k=1 binary ceiling ===")
    print("  At typical ENSO-like params (q=0.29, r=0.97 from SEP domain):")
    q, r = 0.29, 0.97
    ratio = _mi_ratio_binary(q, r)
    print(f"  q={q} r={r}: MI_ratio={ratio:.4f}")
    # Max over q for r=0.97
    best_q = max([i/100 for i in range(1,50)], key=lambda qq: _mi_ratio_binary(qq, r))
    best = _mi_ratio_binary(best_q, r)
    print(f"  Max over q for r=0.97: q={best_q:.2f} → MI_ratio={best:.4f}")
    # k=3 at r=0.97 balanced
    classes = [(1/3, r, 0), (1/3, r, 1), (1/3, r, 2)]
    _, r3 = _mi_k_class(classes)
    print(f"  k=3 balanced at r=0.97: MI_ratio={r3:.4f}  (lift = {r3-best:.4f})")
    return r3 > best

# ── Monotonicity check ───────────────────────────────────────────────────────
def check_monotone():
    print("\n=== Monotonicity in r ===")
    q = 1.0/3
    classes_base = [(q, None, 0), (q, None, 1), (q, None, 2)]
    r_vals = [r/100 for r in range(34, 100, 5)]
    prev = None
    ok = True
    for r in r_vals:
        classes = [(q, r, d) for _, _, d in [(q, r, 0), (q, r, 1), (q, r, 2)]]
        _, ratio = _mi_k_class(classes)
        if prev is not None and ratio < prev - 1e-9:
            print(f"  NON-MONOTONE at r={r:.2f}: {prev:.4f} → {ratio:.4f}")
            ok = False
        prev = ratio
    print(f"  k=3 symmetric monotone in r: {'PASS' if ok else 'FAIL'}")
    return ok

# ── ENSO 3-class application ─────────────────────────────────────────────────
def enso_3class():
    print("\n=== ENSO 3-class (ONI threshold classification) ===")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=20)
        lines = resp.read().decode().splitlines()
    except Exception as e:
        print(f"  Network error: {e}")
        return None

    vals = []
    for line in lines:
        parts = line.split()
        if len(parts) != 4: continue
        try: float(parts[1]); vals.append(float(parts[3]))
        except ValueError: continue
    if len(vals) < 100:
        print("  Too few values")
        return None
    n = len(vals)
    print(f"  ONI N={n}")

    # Rank bins → tiers (same as cert [476])
    si = sorted(range(n), key=lambda i: vals[i])
    rk = [0]*n
    for rank, idx in enumerate(si): rk[idx] = rank
    bins = [int(math.floor(r * MOD / n)) for r in rk]
    tiers = [b // 9 for b in bins]

    # ENSO phase classification (simple threshold per season)
    def _phase(oni):
        if oni >= 0.5: return 2   # El Niño
        if oni <= -0.5: return 0  # La Niña
        return 1                  # Neutral

    phases = [_phase(v) for v in vals]
    q_ElNino = phases.count(2) / n
    q_LaNina = phases.count(0) / n
    q_Neutral = phases.count(1) / n
    print(f"  El Niño fraction: {q_ElNino:.4f}  ({phases.count(2)} seasons)")
    print(f"  La Niña fraction: {q_LaNina:.4f}  ({phases.count(0)} seasons)")
    print(f"  Neutral fraction: {q_Neutral:.4f}  ({phases.count(1)} seasons)")

    # Compute r values: P(dominant_tier | phase)
    def _r_for_phase(phase_label, dominant_tier):
        phase_seasons = [(tiers[i], phases[i]) for i in range(n)]
        count_phase = sum(1 for _, p in phase_seasons if p == phase_label)
        count_both  = sum(1 for t, p in phase_seasons if p == phase_label and t == dominant_tier)
        r = count_both / count_phase if count_phase > 0 else 0.0
        return r, count_phase, count_both

    r_elnino, n_en, n_en_t2 = _r_for_phase(2, 2)  # El Niño → T2
    r_lanina, n_la, n_la_t0 = _r_for_phase(0, 0)  # La Niña → T0
    r_neutral, n_ne, n_ne_t1 = _r_for_phase(1, 1) # Neutral → T1

    print(f"  r_ElNino (→T2): {r_elnino:.4f}  ({n_en_t2}/{n_en})")
    print(f"  r_LaNina (→T0): {r_lanina:.4f}  ({n_la_t0}/{n_la})")
    print(f"  r_Neutral(→T1): {r_neutral:.4f}  ({n_ne_t1}/{n_ne})")

    # k=3 formula (no background; 3 classes sum to 1)
    classes = [
        (q_ElNino, r_elnino, 2),
        (q_LaNina, r_lanina, 0),
        (q_Neutral, r_neutral, 1),
    ]
    mi, mi_ratio_formula = _mi_k_class(classes)
    compact = _mi_formula_compact(classes)
    hy_enso = _h_entropy([q_ElNino, q_LaNina, q_Neutral])
    mi_ratio_compact = compact / hy_enso if hy_enso > 1e-12 else 0.0

    print(f"\n  k=3 formula MI: {mi:.4f} bits")
    print(f"  k=3 formula MI_ratio (exact):  {mi_ratio_formula:.4f}")
    print(f"  k=3 formula MI_ratio (compact): {mi_ratio_compact:.4f}")

    # Empirical MI (for comparison)
    def _empirical_mi(tiers, phases, n):
        from collections import Counter
        joint = Counter(zip(phases, tiers))
        ph_cnt = Counter(phases)
        ti_cnt = Counter(tiers)
        mi_emp = 0.0
        for (p, t), c in joint.items():
            p_pt = c / n
            p_p  = ph_cnt[p] / n
            p_t  = ti_cnt[t] / n
            if p_pt > 0 and p_p > 0 and p_t > 0:
                mi_emp += p_pt * math.log2(p_pt / (p_p * p_t))
        hy = sum(-cnt/n * math.log2(cnt/n) for cnt in ph_cnt.values() if cnt > 0)
        return mi_emp, hy, mi_emp / hy if hy > 0 else 0.0

    mi_emp, hy_emp, mi_ratio_emp = _empirical_mi(tiers, phases, n)
    print(f"\n  Empirical MI:          {mi_emp:.4f} bits")
    print(f"  Empirical MI_ratio:    {mi_ratio_emp:.4f}")
    print(f"  Cert [465] reference:  0.699 (1.07 bits / 1.537 bits)")
    print(f"  Formula vs empirical: diff={abs(mi_ratio_formula - mi_ratio_emp):.4f}")

    return {
        "q_ElNino": q_ElNino, "q_LaNina": q_LaNina, "q_Neutral": q_Neutral,
        "r_ElNino": r_elnino, "r_LaNina": r_lanina, "r_Neutral": r_neutral,
        "mi_ratio_formula": mi_ratio_formula, "mi_ratio_compact": mi_ratio_compact,
        "mi_ratio_empirical": mi_ratio_emp,
        "mi_formula_bits": mi, "mi_empirical_bits": mi_emp,
    }

if __name__ == "__main__":
    ok1 = verify_k1_reduction()
    ok2 = verify_compact_formula()
    ok3 = k3_ceiling()
    ok4 = k1_ceiling()
    ok5 = check_monotone()
    enso = enso_3class()
    print("\n=== Summary ===")
    print(f"  k=1 reduction:  {'PASS' if ok1 else 'FAIL'}")
    print(f"  Compact formula:{'PASS' if ok2 else 'FAIL'}")
    print(f"  k=3 ceiling=1:  {'PASS' if ok3 else 'FAIL'}")
    print(f"  k=3 lifts k=1:  {'PASS' if ok4 else 'FAIL'}")
    print(f"  Monotone in r:  {'PASS' if ok5 else 'FAIL'}")
    if enso:
        diff = abs(enso["mi_ratio_formula"] - enso["mi_ratio_empirical"])
        print(f"  ENSO formula vs empirical: diff={diff:.4f}  {'PASS' if diff < 0.15 else 'FAIL'}")
        print(f"\nFallback values for validator:")
        print(f"  q_ElNino={enso['q_ElNino']:.4f}, q_LaNina={enso['q_LaNina']:.4f}, q_Neutral={enso['q_Neutral']:.4f}")
        print(f"  r_ElNino={enso['r_ElNino']:.4f}, r_LaNina={enso['r_LaNina']:.4f}, r_Neutral={enso['r_Neutral']:.4f}")
        print(f"  mi_ratio_formula={enso['mi_ratio_formula']:.4f}")
        print(f"  mi_ratio_empirical={enso['mi_ratio_empirical']:.4f}")
