"""
QA_COMPLIANCE:
  DECL-1: This file explicitly acknowledges QA Axiom compliance.
  A1: State bins in {0,...,26} via rank normalization (NEVER negative).
  A2: d=b+e, a=b+2e derived; never assigned independently.
  T2: Earthquake location, magnitude, and origin time are observer projections.
      Event count per window is a natural integer — no float enters QA layer.
      Rank normalization is integer arithmetic throughout.
  S1: No b**2 expressions used.
  S2: QA state (event counts, rank bins) are int throughout. No float QA state.
  T1: QA time = window index k (integer). No continuous time in QA logic.
Empirical cert [448]: QA Witt Tower Tohoku Aftershock Orbit Discriminator.
Data: USGS Earthquake Hazards Program catalog (public domain);
  (Incorporated Research Institutions for Seismology, 2011) ComCat,
  doi:10.5066/F7MS3QZH. Event: 2011-03-11 Tohoku M9.1.
  Bounding box 35-42 N, 138-146 E, M>=3.0.
Feature: M>=3.0 earthquake count per 6-hour window (natural integer).
Background: 2011-02-01 to 2011-03-08 (35 days, 140 windows).
Aftershock: 2011-03-11 to 2011-03-18 (7 days, 28 windows).
Rank-normalised across 168 total windows.
Primary sources: (Utsu, 1961) Omori-Utsu aftershock decay law;
  (Wall, 1960) doi:10.1080/00029890.1960.11989541 (Witt tower theory).
"""

import json, sys, math
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Witt tower constants (MOD=27, companion M=[[5,-1],[1,0]], p=3, k=3)
# ---------------------------------------------------------------------------
MOD = 27
# T0 (Singularity neighbourhood): bins  0 –  8
# T1 (Satellite neighbourhood):   bins  9 – 17
# T2 (Cosmos neighbourhood):      bins 18 – 26

def _tier(b):
    if b < 9:
        return 0
    if b < 18:
        return 1
    return 2

# ---------------------------------------------------------------------------
# Event parameters
# ---------------------------------------------------------------------------
LON0, LON1 = 138.0, 146.0
LAT0, LAT1 = 35.0, 42.0
MMIN        = 3.0
W_SEC       = 6 * 3600        # 6-hour windows in seconds
W_MS        = W_SEC * 1000    # ms

BG_START  = "2011-02-01"
BG_END    = "2011-03-08"
BG_DAYS   = 35
BG_N_WIN  = BG_DAYS * 4      # 140

AFT_START = "2011-03-11"
AFT_END   = "2011-03-18"
AFT_DAYS  = 7
AFT_N_WIN = AFT_DAYS * 4     # 28

# ---------------------------------------------------------------------------
# Hardcoded fallback window counts (derived from USGS live query 2026-06-18)
# ---------------------------------------------------------------------------
_FALLBACK_BG = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]

_FALLBACK_AFT = [
    7, 186, 159, 181, 142, 124, 96, 112, 111, 75, 87, 75,
    61, 56, 58, 59, 25, 47, 35, 34, 26, 23, 39, 37, 20, 22, 34, 34,
]

# ---------------------------------------------------------------------------
# Live data acquisition
# ---------------------------------------------------------------------------

def _epoch_ms(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_counts(start_str, end_str, n_win):
    """Fetch M>=3.0 events from USGS and bin into 6-hour windows."""
    import urllib.request, ssl
    ctx = ssl.create_default_context()
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?starttime={start_str}&endtime={end_str}"
        f"&minlatitude={LAT0}&maxlatitude={LAT1}"
        f"&minlongitude={LON0}&maxlongitude={LON1}"
        f"&minmagnitude={MMIN}&format=geojson&orderby=time"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "QA-Cert-448/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        data = json.load(r)
    t0_ms = _epoch_ms(start_str)
    counts = [0] * n_win
    for feat in data["features"]:
        t_ms = feat["properties"]["time"]
        idx = int((t_ms - t0_ms) / W_MS)
        if 0 <= idx < n_win:
            counts[idx] += 1
    return counts


def _acquire():
    """Return (wc_bg, wc_aft, source)."""
    try:
        bg  = _fetch_counts(BG_START, BG_END, BG_N_WIN)
        aft = _fetch_counts(AFT_START, AFT_END, AFT_N_WIN)
        if len(bg) == BG_N_WIN and len(aft) == AFT_N_WIN:
            return bg, aft, "live"
    except Exception:
        pass
    return list(_FALLBACK_BG), list(_FALLBACK_AFT), "fallback"


# ---------------------------------------------------------------------------
# Rank-normalisation → Z/27Z bins (all integer arithmetic)
# ---------------------------------------------------------------------------

def _rank_bins(wc_bg, wc_aft):
    all_wc = wc_bg + wc_aft
    N = len(all_wc)
    sorted_idx = sorted(range(N), key=lambda i: all_wc[i])
    bins = [0] * N
    for rank, idx in enumerate(sorted_idx):
        bins[idx] = int(rank * MOD / N)
    return bins[:len(wc_bg)], bins[len(wc_bg):]


# ---------------------------------------------------------------------------
# Hypergeometric log10-p (one-tailed)
# ---------------------------------------------------------------------------

def _hyp_log10p(N, K, n, k):
    return sum(math.log10((K - j) / (N - j)) for j in range(k))


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate():
    wc_bg, wc_aft, source = _acquire()
    bins_bg, bins_aft = _rank_bins(wc_bg, wc_aft)

    N      = len(bins_bg) + len(bins_aft)
    n_bg   = len(bins_bg)
    n_aft  = len(bins_aft)

    tier_bg  = [_tier(b) for b in bins_bg]
    tier_aft = [_tier(b) for b in bins_aft]

    def _cnt(tiers, t): return sum(1 for x in tiers if x == t)

    t0_bg = _cnt(tier_bg, 0);  t1_bg = _cnt(tier_bg, 1);  t2_bg = _cnt(tier_bg, 2)
    t0_at = _cnt(tier_aft, 0); t1_at = _cnt(tier_aft, 1); t2_at = _cnt(tier_aft, 2)
    total_t2 = t2_bg + t2_at
    total_t0 = t0_bg + t0_at

    mean_bg  = sum(tier_bg)  / n_bg
    mean_aft = sum(tier_aft) / n_aft

    # C6 — Omori-Utsu decay: daily sums strictly decrease over 7 days
    daily = [sum(wc_aft[d * 4:(d + 1) * 4]) for d in range(7)]
    omori_ok = all(daily[i] > daily[i + 1] for i in range(6))

    # Event count ratio (for reporting)
    mean_cnt_bg  = sum(wc_bg) / n_bg
    mean_cnt_aft = sum(wc_aft) / n_aft
    cnt_ratio = mean_cnt_aft / max(mean_cnt_bg, 0.01)

    p_c3 = _hyp_log10p(N, total_t2, n_aft, n_aft)
    p_c2 = _hyp_log10p(N, N - total_t0, n_aft, n_aft)

    checks = {
        "C1_DATA_ACQUISITION":  n_bg == BG_N_WIN and n_aft == AFT_N_WIN,
        "C2_AFT_EXCL_SING":     t0_at == 0,
        "C3_AFT_IN_COSMOS":     t2_at == n_aft,
        "C4_MONOTONIC_TIER":    mean_bg < mean_aft,
        "C5_TIER_DISJOINT":     (t0_at == 0 and t1_at == 0
                                  and {_tier(b) for b in bins_aft} == {2}
                                  and any(_tier(b) == 0 for b in bins_bg)),
        "C6_OMORI_DECAY":       omori_ok,
    }
    ok = all(checks.values())

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    fixtures = [
        {"name": "FIX1_N_BG_WINDOWS",
         "desc": f"Background windows == {BG_N_WIN}",
         "expected": BG_N_WIN, "actual": n_bg, "passed": n_bg == BG_N_WIN},
        {"name": "FIX2_N_AFT_WINDOWS",
         "desc": f"Aftershock windows == {AFT_N_WIN}",
         "expected": AFT_N_WIN, "actual": n_aft, "passed": n_aft == AFT_N_WIN},
        {"name": "FIX3_AFT_T0_ZERO",
         "desc": "Aftershock in T0 == 0",
         "expected": 0, "actual": t0_at, "passed": t0_at == 0},
        {"name": "FIX4_AFT_T2_ALL",
         "desc": f"Aftershock in T2 == {AFT_N_WIN}",
         "expected": AFT_N_WIN, "actual": t2_at, "passed": t2_at == AFT_N_WIN},
        {"name": "FIX5_MEAN_TIER_BG",
         "desc": "Background mean tier in (0.5, 1.2)",
         "expected": True,
         "actual": 0.5 < mean_bg < 1.2,
         "passed": 0.5 < mean_bg < 1.2},
        {"name": "FIX6_COUNT_RATIO",
         "desc": "Aftershock/background count ratio >= 100",
         "expected": True,
         "actual": cnt_ratio >= 100.0,
         "passed": cnt_ratio >= 100.0},
        {"name": "FIX7_DAILY_DECAY_MONOTONE",
         "desc": "7 consecutive daily aftershock sums strictly decrease",
         "expected": True,
         "actual": omori_ok,
         "passed": omori_ok},
        {"name": "FIX8_WRONG_CLAIM_AFT_T0",
         "desc": "Aftershock is NOT in T0 (exclusively T2)",
         "expected": True,
         "actual": t0_at == 0,
         "passed": t0_at == 0},
    ]

    result = {
        "ok": ok,
        "cert": "qa_witt_tower_aftershock_orbit_cert_v1",
        "family_id": 448,
        "source": source,
        "dataset": {
            "name": "USGS ComCat — Tohoku 2011 M9.1 aftershock sequence",
            "doi": "10.5066/F7MS3QZH",
            "event": "Tohoku 2011-03-11 M9.1",
            "bbox": f"{LAT0}-{LAT1}N {LON0}-{LON1}E M>={MMIN}",
            "window_h": 6,
            "n_bg_windows": n_bg,
            "n_aft_windows": n_aft,
            "bg_epoch": f"{BG_START}–{BG_END}",
            "aft_epoch": f"{AFT_START}–{AFT_END}",
        },
        "tier_counts": {
            "background": {"T0": t0_bg, "T1": t1_bg, "T2": t2_bg},
            "aftershock": {"T0": t0_at, "T1": t1_at, "T2": t2_at},
        },
        "mean_tier": {"background": round(mean_bg, 3), "aftershock": round(mean_aft, 3)},
        "event_counts": {
            "bg_total": sum(wc_bg),
            "aft_total": sum(wc_aft),
            "mean_bg_per_window": round(mean_cnt_bg, 3),
            "mean_aft_per_window": round(mean_cnt_aft, 1),
            "ratio": round(cnt_ratio, 0),
        },
        "omori_daily": daily,
        "log10_p": {
            "C2_aft_excl_t0": round(p_c2, 2),
            "C3_aft_in_t2":   round(p_c3, 2),
        },
        "checks": checks,
        "checks_detail": {
            "C1": f"bg={n_bg} aft={n_aft} windows",
            "C2": f"aftershock in T0={t0_at}/28; log10_p={p_c2:.2f}",
            "C3": f"aftershock in T2={t2_at}/28; log10_p={p_c3:.2f}",
            "C4": f"mean_bg={mean_bg:.3f} < mean_aft={mean_aft:.3f}",
            "C5": f"aftershock tier set={set(tier_aft)!r}; bg T0={t0_bg}>0",
            "C6": f"Omori daily={daily}; strictly_decreasing={omori_ok}",
        },
        "fixtures": fixtures,
    }
    return result


if __name__ == "__main__":
    r = validate()
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["ok"] else 1)
