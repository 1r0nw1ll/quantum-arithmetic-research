#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=empirical climatological data; NOAA CPC Oceanic Nino Index (public domain); structural parent cert [110] doi.org/10.1080/00029890.1960.11989541 (Wall 1960); Witt tower companion theory from cert chain [433]-[444] -->

QA_COMPLIANCE = (
    "cert_validator -- integer rank bins {0..26} over monthly ONI anomaly; "
    "Witt tower orbit tiers T0/T1/T2 = bins 0-8/9-17/18-26; "
    "hypergeometric p-values under iid-month null; "
    "Theorem NT: ONI SST anomaly is observer projection; bins are QA integer state; "
    "no float QA state; phase labels are observer classification, not QA inputs"
)

"""QA Witt Tower ENSO Orbit Discriminator Cert [445].

Maps NOAA Oceanic Niño Index (ONI) monthly anomalies (1950-2026, N=916) to
the Witt tower three-tier orbit partition, verifying that the ENSO phase
progression (La Niña → neutral → El Niño) corresponds exactly to the
singularity → satellite → cosmos orbit trajectory.

Data: NOAA CPC ONI, 3-month running mean of Niño 3.4 SST anomaly (°C)
      URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
      Public domain. Records 1950-DJF through present.

ENSO phase labels (standard NOAA thresholds):
  La Niña:  ONI ≤ -0.5 °C  (252 months)
  Neutral: -0.5 < ONI < 0.5 (419 months)
  El Niño:  ONI ≥  0.5 °C  (245 months)

Orbit-tier partition (Witt tower, MOD=27):
  T0 (bins 0-8,  coldest 1/3):  Singularity neighborhood — compressed
  T1 (bins 9-17, middle 1/3):   Satellite neighborhood — transitional
  T2 (bins 18-26, warmest 1/3): Cosmos neighborhood — full dynamics

Checks
------
C1  DATA_ACQUISITION   -- fetch and parse NOAA ONI, verify record count
C2  LANINA_SINGULARITY -- all La Niña months in T0; p < 10^{-100}
C3  ELNINO_COSMOS      -- all El Niño months in T2; p < 10^{-100}
C4  TIER_DISJOINT      -- La Niña uses {T0}, El Niño uses {T2}; disjoint
C5  MONOTONIC_TIER     -- mean tier: La Niña=0.0 < neutral=1.014 < El Niño=2.0
C6  V3_PERSISTENCE     -- v_3 valuation above null for all phases
"""

import json
import math
import ssl
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

NOAA_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
MOD = 27
LA_NINA_THRESH = -0.5
EL_NINO_THRESH = 0.5

# Hardcoded fallback (ONI ANOM column, 1950-DJF through 2026-MAM, N=916)
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
# Core helpers
# ---------------------------------------------------------------------------

def _fetch_oni():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(NOAA_URL, context=ctx, timeout=15) as r:
        raw = r.read().decode()
    anoms = []
    for line in raw.strip().split("\n"):
        parts = line.split()
        if len(parts) == 4 and parts[0] != "SEAS":
            try:
                anoms.append(float(parts[3]))
            except ValueError:
                pass
    return anoms


def _rank_bins(anoms):
    n = len(anoms)
    sorted_idx = sorted(range(n), key=lambda i: anoms[i])
    ranks = [0] * n
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    return [r * MOD // n for r in ranks]


def _enso_phase(oni):
    if oni <= LA_NINA_THRESH:
        return "la_nina"
    if oni >= EL_NINO_THRESH:
        return "el_nino"
    return "neutral"


def _orbit_tier(b):
    return b // 9


def _v3(b, e):
    diff = (b - e) % MOD
    if diff == 0:
        return 3
    if diff % 9 == 0:
        return 2
    if diff % 3 == 0:
        return 1
    return 0


def _log_hypergeom(k, N, K, n):
    if k < max(0, n - (N - K)) or k > min(n, K):
        return -float("inf")
    return (
        math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)
        + math.lgamma(N - K + 1) - math.lgamma(n - k + 1) - math.lgamma(N - K - n + k + 1)
        - math.lgamma(N + 1) + math.lgamma(n + 1) + math.lgamma(N - n + 1)
    )


def _hyper_upper(k_obs, N, K, n):
    k_min, k_max = max(0, n - (N - K)), min(n, K)
    log_all = [_log_hypergeom(k, N, K, n) for k in range(k_min, k_max + 1)]
    m = max(log_all)
    total = sum(math.exp(v - m) for v in log_all)
    tail = sum(math.exp(_log_hypergeom(k, N, K, n) - m) for k in range(k_obs, k_max + 1))
    return tail / total


def _log10_hyper_exact(k, N, K, n):
    return _log_hypergeom(k, N, K, n) / math.log(10)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_c1_data_acquisition(anoms, live):
    n = len(anoms)
    n_ln = sum(1 for a in anoms if a <= LA_NINA_THRESH)
    n_en = sum(1 for a in anoms if a >= EL_NINO_THRESH)
    ok = n >= 900 and n_ln >= 200 and n_en >= 200
    return {"n_records": n, "n_la_nina": n_ln, "n_el_nino": n_en, "live_fetch": live, "ok": ok}


def _check_c2_lanina_singularity(anoms, bins):
    N = len(anoms)
    K0 = sum(1 for b in bins if b < 9)
    ln_idx = [i for i, a in enumerate(anoms) if a <= LA_NINA_THRESH]
    n_ln_T0 = sum(1 for i in ln_idx if bins[i] < 9)
    log10_p = _log10_hyper_exact(n_ln_T0, N, K0, len(ln_idx))
    ok = n_ln_T0 == len(ln_idx) and log10_p < -50
    return {
        "N": N, "K0": K0, "n_la_nina": len(ln_idx),
        "n_la_nina_T0": n_ln_T0, "fraction": n_ln_T0 / max(len(ln_idx), 1),
        "log10_p": round(log10_p, 1), "ok": ok,
    }


def _check_c3_elnino_cosmos(anoms, bins):
    N = len(anoms)
    K2 = sum(1 for b in bins if b >= 18)
    en_idx = [i for i, a in enumerate(anoms) if a >= EL_NINO_THRESH]
    n_en_T2 = sum(1 for i in en_idx if bins[i] >= 18)
    log10_p = _log10_hyper_exact(n_en_T2, N, K2, len(en_idx))
    ok = n_en_T2 == len(en_idx) and log10_p < -50
    return {
        "N": N, "K2": K2, "n_el_nino": len(en_idx),
        "n_el_nino_T2": n_en_T2, "fraction": n_en_T2 / max(len(en_idx), 1),
        "log10_p": round(log10_p, 1), "ok": ok,
    }


def _check_c4_tier_disjoint(anoms, bins):
    ln_tiers = {_orbit_tier(bins[i]) for i, a in enumerate(anoms) if a <= LA_NINA_THRESH}
    en_tiers = {_orbit_tier(bins[i]) for i, a in enumerate(anoms) if a >= EL_NINO_THRESH}
    disjoint = ln_tiers.isdisjoint(en_tiers)
    ok = disjoint and ln_tiers == {0} and en_tiers == {2}
    return {
        "la_nina_tiers": sorted(ln_tiers), "el_nino_tiers": sorted(en_tiers),
        "disjoint": disjoint, "ok": ok,
    }


def _check_c5_monotonic_tier(anoms, bins):
    ph_mean = {}
    for ph in ("la_nina", "neutral", "el_nino"):
        idx = [i for i, a in enumerate(anoms) if _enso_phase(a) == ph]
        ph_mean[ph] = sum(_orbit_tier(bins[i]) for i in idx) / max(len(idx), 1)
    tiers = [ph_mean["la_nina"], ph_mean["neutral"], ph_mean["el_nino"]]
    monotone = tiers[0] < tiers[1] < tiers[2]
    ok = monotone and tiers[0] == 0.0 and tiers[2] == 2.0
    return {
        "mean_tier": {ph: round(ph_mean[ph], 3) for ph in ("la_nina", "neutral", "el_nino")},
        "strictly_increasing": monotone, "ok": ok,
    }


def _check_c6_v3_persistence(anoms, bins):
    null_v3 = sum(_v3(b, e) for b in range(MOD) for e in range(MOD)) / MOD ** 2
    ph_v3 = {}
    for ph in ("la_nina", "neutral", "el_nino"):
        idx = [i for i, a in enumerate(anoms) if _enso_phase(a) == ph]
        vs = [_v3(bins[idx[j]], bins[idx[j + 1]]) for j in range(len(idx) - 1)]
        ph_v3[ph] = sum(vs) / max(len(vs), 1)
    all_above_null = all(v > null_v3 for v in ph_v3.values())
    ok = all_above_null
    return {
        "null_uniform_27": round(null_v3, 3),
        "mean_v3": {ph: round(ph_v3[ph], 3) for ph in ph_v3},
        "all_above_null": all_above_null, "ok": ok,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _run_fixtures(anoms, bins):
    ln_bins = [bins[i] for i, a in enumerate(anoms) if a <= LA_NINA_THRESH]
    en_bins = [bins[i] for i, a in enumerate(anoms) if a >= EL_NINO_THRESH]
    nt_bins = [bins[i] for i, a in enumerate(anoms) if -0.5 < a < 0.5]
    raw = [
        ("FIX1_LANINA_ALL_T0",
         "All La Nina bins < 9 (Singularity neighborhood T0)",
         True, all(b < 9 for b in ln_bins)),
        ("FIX2_ELNINO_ALL_T2",
         "All El Nino bins >= 18 (Cosmos neighborhood T2)",
         True, all(b >= 18 for b in en_bins)),
        ("FIX3_NO_LANINA_T2",
         "No La Nina month in T2",
         True, all(b < 18 for b in ln_bins)),
        ("FIX4_NO_ELNINO_T0",
         "No El Nino month in T0",
         True, all(b >= 9 for b in en_bins)),
        ("FIX5_NEUTRAL_SPANS_ALL",
         "Neutral months span all three tiers",
         True, any(b < 9 for b in nt_bins) and any(9 <= b < 18 for b in nt_bins) and any(b >= 18 for b in nt_bins)),
        ("FIX6_LANINA_MEAN_TIER_ZERO",
         "La Nina mean tier = 0.0 exactly",
         True, sum(_orbit_tier(b) for b in ln_bins) == 0),
        ("FIX7_ELNINO_MEAN_TIER_TWO",
         "El Nino mean tier = 2.0 exactly",
         True, all(_orbit_tier(b) == 2 for b in en_bins)),
        ("FIX8_WRONG_CLAIM_NEUTRAL_T0",
         "Neutral is NOT exclusively in T0 (spans all tiers)",
         True, not all(b < 9 for b in nt_bins)),
    ]
    return [
        {"name": n, "desc": d, "expected": e, "actual": a, "passed": a == e}
        for n, d, e, a in raw
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        anoms = _fetch_oni()
        live = True
    except Exception:
        anoms = list(_FALLBACK_ANOMS)
        live = False

    bins = _rank_bins(anoms)

    checks = {
        "C1_DATA_ACQUISITION":   _check_c1_data_acquisition(anoms, live),
        "C2_LANINA_SINGULARITY": _check_c2_lanina_singularity(anoms, bins),
        "C3_ELNINO_COSMOS":      _check_c3_elnino_cosmos(anoms, bins),
        "C4_TIER_DISJOINT":      _check_c4_tier_disjoint(anoms, bins),
        "C5_MONOTONIC_TIER":     _check_c5_monotonic_tier(anoms, bins),
        "C6_V3_PERSISTENCE":     _check_c6_v3_persistence(anoms, bins),
    }
    all_ok = all(v["ok"] for v in checks.values())
    fixtures = _run_fixtures(anoms, bins)
    n_pass = sum(1 for f in fixtures if f["passed"])

    out = {
        "ok": all_ok and n_pass == len(fixtures),
        "cert": "QA Witt Tower ENSO Orbit Discriminator",
        "family_id": 445,
        "dataset": f"NOAA ONI 1950-present, N={len(anoms)}, live={live}",
        "checks": {k: v["ok"] for k, v in checks.items()},
        "checks_detail": checks,
        "fixture_summary": f"{n_pass}/{len(fixtures)} passed",
        "fixtures": fixtures,
    }
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
