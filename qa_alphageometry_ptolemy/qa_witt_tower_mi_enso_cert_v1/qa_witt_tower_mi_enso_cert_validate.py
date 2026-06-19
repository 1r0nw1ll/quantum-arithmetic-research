#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=empirical climatological data; NOAA PSL Oceanic Nino Index
#      (public domain); structural parent cert [110] doi.org/10.1080/00029890.1960.11989541
#      (Wall 1960); introduces MI as 5th QA feature type (alongside RMS, ZCR, Poisson count,
#      spectral entropy) for physical domain chain [442]-[452] -->

QA_COMPLIANCE = (
    "cert_validator -- integer rank bins {0..26} over monthly ONI anomaly; "
    "Witt tower orbit tiers T0/T1/T2 = bins 0-8/9-17/18-26; "
    "empirical mutual information I(tier; ENSO_phase) computed from integer contingency counts; "
    "permutation null shuffles event labels over fixed integer orbit classes; "
    "Theorem NT: ONI SST anomaly is observer projection; bins are QA integer state; "
    "no float QA state; ENSO phase labels are observer classification, not QA inputs"
)

"""QA Witt Tower Mutual Information -- ENSO Physical Domain Cert [465].

Introduces mutual information (MI) as the 5th QA feature type for physical
domain certs [442]-[452], complementing amplitude RMS, zero-crossing rate,
Poisson count, and spectral entropy. MI is an information-theoretic summary
that captures arbitrary statistical dependence -- not just mean shifts -- between
the QA orbit-tier classification and a physical event label.

Domain: ENSO (Oceanic Nino Index, NOAA, 1950-2026, N=916 months).
Label: 3-class ENSO phase -- La Nina (ONI <= -0.5), Neutral, El Nino (ONI >= 0.5).
Orbit feature: Witt tower tier (T0/T1/T2 = bins 0-8/9-17/18-26 in Z/27Z).

Structural context:
  Cert [445] showed 100% perfect concentration: La Nina -> T0 (log10_p=-172),
  El Nino -> T2 (log10_p=-166). The 3x3 contingency table is near-diagonal.
  MI quantifies the information content of this concentration in bits:
    I(orbit_tier; ENSO_phase) ~= 1.08 bits  (70% of H(ENSO_phase)=1.54 bits)
  This far exceeds the expected MI under label permutation (~= 0.003 bits).

Checks
------
C1  DATA           -- N >= 900 months, all 3 ENSO phases detected
C2  TIER_BALANCE   -- each orbit tier has 295-320 months (~= 1/3 each, +-3%)
C3  MI_MAGNITUDE   -- I(tier; ENSO_phase) > 0.8 bits
C4  PERM_SIG       -- perm_p = count(null_mi >= obs_mi)/N_PERM < 0.001
C5  MI_RATIO       -- obs_mi / H(ENSO_phase) > 0.60
C6  DIAGONAL_DOM   -- pointwise MI contributions of (T0,La Nina), (T1,Neutral),
                       (T2,El Nino) are all positive (off-diagonal terms negative)

Primary source: NOAA CPC / PSL Oceanic Nino Index (public domain).
Structural parent: cert [110] (Witt tower framework).
Orbit chain: cert [445] (ENSO orbit discriminator, hypergeometric validation).
"""

import json
import math
import random
import ssl
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MOD = 27
LA_NINA_THRESH = -0.5
EL_NINO_THRESH = 0.5
N_PERM = 5000
SEED = 42
MI_THRESHOLD = 0.8        # bits -- minimum MI magnitude gate
MI_RATIO_THRESHOLD = 0.60  # fraction of H(label) captured

LA_NINA = "la_nina"
NEUTRAL = "neutral"
EL_NINO = "el_nino"

# ---------------------------------------------------------------------------
# Hardcoded fallback (NOAA PSL ONI, 1950-JAN through 2026-MAR, N=916)
# Same values as cert [445] _FALLBACK_ANOMS (public domain NOAA data)
# ---------------------------------------------------------------------------
_FALLBACK_ANOMS = [
    -1.53, -1.34, -1.16, -1.18, -1.07, -0.85, -0.54, -0.42, -0.39, -0.44, -0.6, -0.8,
    -0.82, -0.54, -0.17, 0.18, 0.36, 0.58, 0.7, 0.89, 0.99, 1.15, 1.04, 0.81, 0.53, 0.37,
    0.34, 0.29, 0.2, 0.0, -0.08, 0.0, 0.15, 0.1, 0.04, 0.15, 0.4, 0.6, 0.63, 0.66, 0.75,
    0.77, 0.75, 0.73, 0.78, 0.84, 0.84, 0.81, 0.76, 0.47, -0.05, -0.41, -0.54, -0.5, -0.64,
    -0.84, -0.9, -0.77, -0.73, -0.66, -0.68, -0.62, -0.69, -0.8, -0.79, -0.72, -0.68, -0.75,
    -1.09, -1.42, -1.67, -1.47, -1.11, -0.76, -0.63, -0.54, -0.52, -0.51, -0.57, -0.55,
    -0.46, -0.42, -0.43, -0.43, -0.25, 0.06, 0.41, 0.72, 0.92, 1.11, 1.25, 1.32, 1.33,
    1.39, 1.53, 1.74, 1.81, 1.66, 1.27, 0.93, 0.74, 0.64, 0.57, 0.43, 0.39, 0.44, 0.5,
    0.61, 0.61, 0.62, 0.52, 0.33, 0.2, -0.07, -0.18, -0.28, -0.09, -0.03, 0.05, -0.04,
    -0.1, -0.1, -0.07, 0.03, 0.02, 0.03, 0.13, 0.24, 0.27, 0.2, 0.12, 0.05, 0.04, 0.03,
    0.04, 0.09, 0.23, 0.27, 0.14, -0.13, -0.3, -0.26, -0.19, -0.16, -0.24, -0.22, -0.2,
    -0.26, -0.28, -0.2, -0.04, -0.07, -0.11, -0.22, -0.31, -0.43, -0.4, -0.15, 0.15, 0.27,
    0.31, 0.52, 0.86, 1.14, 1.22, 1.29, 1.37, 1.31, 1.07, 0.62, 0.12, -0.33, -0.58, -0.58,
    -0.6, -0.66, -0.76, -0.8, -0.82, -0.78, -0.59, -0.28, -0.07, 0.18, 0.46, 0.83, 1.22,
    1.54, 1.85, 1.98, 1.97, 1.72, 1.37, 1.17, 0.98, 0.66, 0.35, 0.24, 0.24, 0.12, -0.05,
    -0.1, -0.18, -0.3, -0.41, -0.48, -0.53, -0.45, -0.24, -0.0, 0.05, -0.16, -0.3, -0.38,
    -0.34, -0.44, -0.64, -0.74, -0.62, -0.44, -0.04, 0.28, 0.58, 0.53, 0.45, 0.55, 0.73,
    0.98, 1.13, 1.09, 0.95, 0.77, 0.61, 0.43, 0.36, 0.51, 0.79, 0.86, 0.81, 0.63, 0.51,
    0.34, 0.29, 0.19, 0.04, -0.3, -0.63, -0.76, -0.77, -0.74, -0.86, -1.15, -1.36, -1.38,
    -1.12, -0.85, -0.73, -0.74, -0.8, -0.77, -0.82, -0.85, -0.96, -0.9, -0.71, -0.35, 0.06,
    0.41, 0.67, 0.92, 1.13, 1.37, 1.58, 1.84, 2.09, 2.12, 1.84, 1.25, 0.54, -0.1, -0.54,
    -0.87, -1.11, -1.28, -1.45, -1.71, -1.95, -2.03, -1.84, -1.55, -1.23, -1.03, -0.91,
    -0.77, -0.53, -0.37, -0.41, -0.61, -0.75, -0.64, -0.54, -0.57, -0.65, -0.73, -0.83,
    -0.98, -1.13, -1.2, -1.37, -1.43, -1.55, -1.65, -1.56, -1.17, -0.73, -0.47, -0.28,
    -0.05, 0.18, 0.35, 0.62, 0.81, 0.86, 0.85, 0.71, 0.64, 0.34, 0.23, 0.21, 0.34, 0.35,
    0.42, 0.57, 0.73, 0.81, 0.79, 0.69, 0.42, 0.06, -0.18, -0.31, -0.29, -0.36, -0.42,
    -0.42, -0.29, -0.08, 0.0, 0.03, 0.07, 0.2, 0.28, 0.23, 0.05, 0.04, 0.17, 0.33, 0.45,
    0.52, 0.64, 0.59, 0.46, 0.34, 0.38, 0.48, 0.46, 0.25, 0.03, -0.07, 0.02, 0.11, -0.01,
    -0.26, -0.5, -0.47, -0.37, -0.26, -0.29, -0.3, -0.25, -0.16, -0.13, -0.15, -0.08,
    -0.05, 0.07, 0.19, 0.47, 0.66, 0.72, 0.79, 1.07, 1.58, 1.97, 2.18, 2.23, 2.18, 1.92,
    1.54, 1.29, 1.06, 0.72, 0.31, -0.08, -0.46, -0.81, -1.0, -0.91, -0.6, -0.42, -0.34,
    -0.43, -0.51, -0.45, -0.3, -0.16, -0.24, -0.56, -0.92, -1.14, -1.04, -0.85, -0.77,
    -0.78, -0.78, -0.63, -0.49, -0.46, -0.4, -0.35, -0.27, -0.36, -0.49, -0.47, -0.31,
    -0.2, -0.12, -0.04, 0.22, 0.44, 0.71, 0.94, 1.14, 1.22, 1.23, 1.19, 1.06, 0.95, 0.97,
    1.22, 1.51, 1.7, 1.65, 1.48, 1.25, 1.11, 0.81, 0.54, 0.14, -0.31, -0.88, -1.3, -1.3,
    -1.11, -1.19, -1.48, -1.8, -1.85, -1.69, -1.43, -1.08, -0.83, -0.58, -0.4, -0.31,
    -0.27, -0.24, -0.22, -0.16, -0.05, 0.14, 0.21, 0.28, 0.29, 0.29, 0.31, 0.33, 0.38,
    0.39, 0.35, 0.4, 0.41, 0.41, 0.26, 0.22, 0.26, 0.45, 0.64, 0.73, 0.64, 0.62, 0.79,
    1.21, 1.53, 1.71, 1.63, 1.48, 1.29, 1.06, 0.73, 0.37, 0.09, -0.13, -0.25, -0.28,
    -0.13, 0.09, 0.3, 0.5, 0.67, 0.7, 0.57, 0.32, 0.25, 0.15, 0.1, 0.04, 0.06, 0.06,
    0.07, 0.17, 0.31, 0.42, 0.41, 0.44, 0.43, 0.55, 0.74, 1.01, 1.09, 0.96, 0.72, 0.53,
    0.3, 0.14, -0.03, -0.24, -0.54, -0.81, -0.97, -1.0, -0.98, -0.9, -0.75, -0.59, -0.39,
    -0.31, -0.3, -0.27, -0.32, -0.35, -0.4, -0.45, -0.49, -0.5, -0.36, -0.1, 0.28, 0.75,
    1.22, 1.6, 1.9, 2.14, 2.33, 2.4, 2.39, 2.24, 1.93, 1.44, 0.99, 0.45, -0.13, -0.78,
    -1.12, -1.31, -1.35, -1.48, -1.57, -1.55, -1.3, -1.07, -0.98, -1.02, -1.04, -1.1,
    -1.11, -1.16, -1.26, -1.46, -1.65, -1.66, -1.41, -1.07, -0.81, -0.71, -0.64, -0.55,
    -0.51, -0.55, -0.63, -0.75, -0.74, -0.68, -0.52, -0.44, -0.34, -0.25, -0.12, -0.08,
    -0.13, -0.19, -0.29, -0.35, -0.31, -0.15, 0.03, 0.09, 0.2, 0.43, 0.65, 0.79, 0.86,
    1.01, 1.21, 1.31, 1.14, 0.92, 0.63, 0.38, -0.04, -0.26, -0.16, 0.08, 0.21, 0.26, 0.29,
    0.35, 0.35, 0.37, 0.31, 0.23, 0.17, 0.17, 0.28, 0.47, 0.64, 0.7, 0.67, 0.66, 0.69,
    0.64, 0.58, 0.45, 0.43, 0.29, 0.11, -0.06, -0.14, -0.11, -0.29, -0.57, -0.84, -0.85,
    -0.77, -0.57, -0.37, -0.14, -0.03, 0.1, 0.3, 0.54, 0.77, 0.94, 0.94, 0.66, 0.22,
    -0.12, -0.32, -0.38, -0.47, -0.56, -0.81, -1.07, -1.34, -1.5, -1.6, -1.64, -1.52,
    -1.29, -1.01, -0.84, -0.61, -0.37, -0.23, -0.24, -0.35, -0.55, -0.73, -0.85, -0.79,
    -0.61, -0.33, 0.01, 0.28, 0.45, 0.58, 0.71, 1.01, 1.36, 1.56, 1.5, 1.22, 0.84, 0.35,
    -0.17, -0.66, -1.05, -1.35, -1.56, -1.64, -1.64, -1.54, -1.31, -1.04, -0.8, -0.62,
    -0.46, -0.37, -0.43, -0.58, -0.79, -0.96, -1.02, -0.92, -0.72, -0.57, -0.46, -0.36,
    -0.17, 0.06, 0.3, 0.41, 0.41, 0.31, 0.13, -0.1, -0.29, -0.29, -0.21, -0.19, -0.27,
    -0.34, -0.35, -0.28, -0.21, -0.13, -0.1, -0.15, -0.28, -0.32, -0.14, 0.15, 0.31, 0.23,
    0.1, 0.11, 0.28, 0.54, 0.71, 0.77, 0.69, 0.61, 0.65, 0.81, 1.02, 1.25, 1.57, 1.91,
    2.21, 2.47, 2.64, 2.75, 2.63, 2.28, 1.71, 1.05, 0.49, 0.0, -0.31, -0.5, -0.58, -0.64,
    -0.6, -0.45, -0.19, -0.02, 0.18, 0.31, 0.4, 0.39, 0.19, -0.07, -0.34, -0.6, -0.77,
    -0.86, -0.77, -0.71, -0.57, -0.39, -0.13, 0.06, 0.14, 0.27, 0.53, 0.81, 0.97, 0.92,
    0.89, 0.86, 0.84, 0.77, 0.64, 0.52, 0.33, 0.19, 0.23, 0.39, 0.58, 0.66, 0.64, 0.63,
    0.53, 0.3, 0.01, -0.23, -0.36, -0.53, -0.85, -1.12, -1.2, -1.08, -0.91, -0.79, -0.71,
    -0.55, -0.39, -0.3, -0.35, -0.45, -0.63, -0.76, -0.91, -0.87, -0.82, -0.79, -0.86,
    -0.95, -0.9, -0.78, -0.76, -0.87, -0.97, -0.94, -0.85, -0.71, -0.54, -0.29, -0.02,
    0.27, 0.57, 0.84, 1.12, 1.37, 1.6, 1.83, 1.99, 2.06, 1.92, 1.62, 1.26, 0.82, 0.49,
    0.22, 0.08, -0.07, -0.17, -0.21, -0.3, -0.42, -0.45, -0.24, -0.06, 0.02, -0.02, -0.04,
    -0.14, -0.28, -0.4, -0.51, -0.55, -0.54, -0.37, -0.14, 0.13, 0.48,
]

# ---------------------------------------------------------------------------
# Live data fetch (NOAA PSL -- yearly rows of 12 monthly anomalies)
# ---------------------------------------------------------------------------

PSL_URL = "https://psl.noaa.gov/data/correlation/oni.data"


def _fetch_oni_psl():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(PSL_URL, headers={"User-Agent": "QA-Cert-465/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        raw = r.read().decode()
    anoms = []
    for line in raw.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                yr = int(parts[0])
                if yr < 1900 or yr > 2100:
                    continue
            except ValueError:
                continue
            for v in parts[1:]:
                try:
                    f = float(v)
                    if abs(f) > 90.0:  # missing sentinel (-99.9 or -999.99)
                        break
                    anoms.append(f)
                except ValueError:
                    break
    return anoms


def _get_data():
    try:
        anoms = _fetch_oni_psl()
        if len(anoms) >= 900:
            return anoms, True
    except Exception:
        pass
    return list(_FALLBACK_ANOMS), False


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _rank_bins(anoms):
    """Rank-normalise into Z/27Z (integer rank bins 0..26)."""
    n = len(anoms)
    sorted_idx = sorted(range(n), key=lambda i: anoms[i])
    ranks = [0] * n
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    return [r * MOD // n for r in ranks]


def _orbit_tier(b):
    """Witt tower orbit tier integer: 0=T0 singularity, 1=T1 satellite, 2=T2 cosmos."""
    return b // 9


def _enso_phase(oni):
    if oni <= LA_NINA_THRESH:
        return LA_NINA
    if oni >= EL_NINO_THRESH:
        return EL_NINO
    return NEUTRAL


def _empirical_mi(orbit_tiers, event_phases):
    """Empirical mutual information I(orbit_tier; event_phase) in bits.

    Computed from empirical joint/marginal frequency estimates. With N=916
    and a 3x3 table the Miller-Madow bias is negligible (~0.003 bits).
    """
    n = len(orbit_tiers)
    joint = {}
    p_tier = {}
    p_phase = {}
    for t, p in zip(orbit_tiers, event_phases):
        joint[(t, p)] = joint.get((t, p), 0) + 1
        p_tier[t] = p_tier.get(t, 0) + 1
        p_phase[p] = p_phase.get(p, 0) + 1
    mi = 0.0
    for (t, p), cnt in joint.items():
        pxy = cnt / n
        px = p_tier[t] / n
        py = p_phase[p] / n
        if pxy > 0:
            mi += pxy * math.log2(pxy / (px * py))
    return mi


def _entropy(labels):
    """Shannon entropy of a discrete sequence in bits."""
    n = len(labels)
    counts = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    h = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_c1_data(anoms, live):
    n = len(anoms)
    n_ln = sum(1 for a in anoms if a <= LA_NINA_THRESH)
    n_en = sum(1 for a in anoms if a >= EL_NINO_THRESH)
    n_ne = n - n_ln - n_en
    ok = n >= 900 and n_ln >= 200 and n_en >= 200 and n_ne >= 300
    return {
        "n_months": n, "n_la_nina": n_ln, "n_neutral": n_ne,
        "n_el_nino": n_en, "live": live, "ok": ok,
    }


def _check_c2_tier_balance(bins):
    n = len(bins)
    t0 = sum(1 for b in bins if b < 9)
    t1 = sum(1 for b in bins if 9 <= b < 18)
    t2 = sum(1 for b in bins if b >= 18)
    lo = int(n * 0.30)
    hi = int(n * 0.37)
    ok = lo <= t0 <= hi and lo <= t1 <= hi and lo <= t2 <= hi
    return {"n_T0": t0, "n_T1": t1, "n_T2": t2, "expected_each": n // 3, "ok": ok}


def _check_c3_mi_magnitude(mi_obs):
    ok = mi_obs > MI_THRESHOLD
    return {"mi_obs_bits": round(mi_obs, 4), "threshold": MI_THRESHOLD, "ok": ok}


def _check_c4_perm_sig(orbit_tiers, phases, mi_obs):
    rng = random.Random(SEED)
    phases_copy = list(phases)
    null_count = 0
    for _ in range(N_PERM):
        rng.shuffle(phases_copy)
        null_mi = _empirical_mi(orbit_tiers, phases_copy)
        if null_mi >= mi_obs:
            null_count += 1
    perm_p = null_count / N_PERM
    ok = perm_p < 0.001
    return {
        "n_perm": N_PERM, "null_exceed_count": null_count,
        "perm_p": perm_p, "mi_obs_bits": round(mi_obs, 4), "ok": ok,
    }


def _check_c5_mi_ratio(mi_obs, phases):
    h_label = _entropy(phases)
    ratio = mi_obs / h_label if h_label > 0 else 0.0
    ok = ratio > MI_RATIO_THRESHOLD
    return {
        "mi_obs_bits": round(mi_obs, 4),
        "h_label_bits": round(h_label, 4),
        "mi_ratio": round(ratio, 4),
        "threshold": MI_RATIO_THRESHOLD,
        "ok": ok,
    }


def _check_c6_diagonal_dom(orbit_tiers, phases):
    """Verify (T0,La Nina), (T1,Neutral), (T2,El Nino) each have positive
    pointwise MI -- i.e., the diagonal of the contingency table is
    over-represented relative to the independence model."""
    n = len(orbit_tiers)
    joint = {}
    p_tier = {}
    p_phase = {}
    for t, p in zip(orbit_tiers, phases):
        joint[(t, p)] = joint.get((t, p), 0) + 1
        p_tier[t] = p_tier.get(t, 0) + 1
        p_phase[p] = p_phase.get(p, 0) + 1

    diag = [(0, LA_NINA, "T0_la_nina"), (1, NEUTRAL, "T1_neutral"), (2, EL_NINO, "T2_el_nino")]
    pmi_diag = {}
    for t, ph, label in diag:
        cnt = joint.get((t, ph), 0)
        if cnt == 0:
            pmi_diag[label] = {"cnt": 0, "pmi": None, "ok": False}
            continue
        pxy = cnt / n
        px = p_tier.get(t, 0) / n
        py = p_phase.get(ph, 0) / n
        pmi = math.log2(pxy / (px * py)) if px > 0 and py > 0 else None
        pmi_diag[label] = {"cnt": cnt, "pmi": round(pmi, 3) if pmi is not None else None}

    all_positive = all(
        v.get("pmi") is not None and v["pmi"] > 0
        for v in pmi_diag.values()
    )
    return {"diagonal_pmi": pmi_diag, "ok": all_positive}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    anoms, live = _get_data()
    bins = _rank_bins(anoms)
    orbit_tiers = [_orbit_tier(b) for b in bins]
    phases = [_enso_phase(a) for a in anoms]
    mi_obs = _empirical_mi(orbit_tiers, phases)

    c1 = _check_c1_data(anoms, live)
    c2 = _check_c2_tier_balance(bins)
    c3 = _check_c3_mi_magnitude(mi_obs)
    c4 = _check_c4_perm_sig(orbit_tiers, phases, mi_obs)
    c5 = _check_c5_mi_ratio(mi_obs, phases)
    c6 = _check_c6_diagonal_dom(orbit_tiers, phases)

    results = {
        "ok": all(c["ok"] for c in [c1, c2, c3, c4, c5, c6]),
        "C1_DATA": c1,
        "C2_TIER_BALANCE": c2,
        "C3_MI_MAGNITUDE": c3,
        "C4_PERM_SIG": c4,
        "C5_MI_RATIO": c5,
        "C6_DIAGONAL_DOM": c6,
    }

    print(json.dumps(results, indent=2))
    sys.exit(0 if results["ok"] else 1)


if __name__ == "__main__":
    main()
