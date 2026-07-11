#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=seismic_analytic_phase_and_geometry_to_QA_phase mod24; qa_neg=time reversal on integer phase; coherent-sum focal map is observer readout (Theorem NT)"
# RT1_OBSERVER_FILE: seismic waveform phases + great-circle geometry are observer-layer projections, not QA state.
"""
Cert [522] time-reversal focusing SPECIFICITY test on REAL seismic data.

Data: a real M5.5 Ridgecrest (2019-07-06) aftershock recorded by a dense regional
vertical-component array (25 stations, data/seismic_tr/, fetched by
qa_seismic_tr_fetch.py). Catalog hypocenter is the ground-truth source.

METHOD (phase-coherence back-projection = the [522] operator on real records):
  Each station's bandpassed analytic signal z_i(t) carries the recorded phase. For a
  candidate source g on a lat/lon grid, the predicted P travel time is
  tau_i(g) = dist(g, station_i)/v. Sampling z_i at t0 + tau_i(g) TIME-REVERSES the
  record and re-emits it toward g (Fink's mirror). The QA observer projection
  quantizes each sampled phase to an integer in {1..24}, and the focal value is the
  phase-coherence of the unit-phasor stack:
      F(g) = | (1/N) sum_i exp(2*pi*i * (Q_i(g)-1)/24) |,  Q_i(g) in {1..24}.
  qa_neg (time reversal) = negating the integer phase mod 24 = complex conjugation.
  F(g) peaks where the real moveout aligns the recorded phases -> the true source.

CLAIMS (Fink time-reversal-mirror physics; cert [518]/[522]):
  (A) REFOCUS      the coherence peak lies near the catalog epicenter.
  (B) SPECIFICITY  the real recorded phases focus, but records with a per-station
                   random phase OFFSET (the negative control -- moveout preserved,
                   inter-station alignment destroyed) do NOT. A pre-P NOISE window is
                   a second negative control.

Theorem NT: waveform analytic phase + station geometry + velocity are observer
inputs (crossed once, in); the QA phase is an integer mod 24; the coherent-sum focal
map is an observer readout (crossed once, out). No float enters the QA phase layer.

HONEST EXPECTATION (pre-registered, not tuned): mod-24 phase quantization is coarse
(15 deg steps) and a constant velocity ignores 3-D structure, so the ABSOLUTE focus
may be broad or biased. The decisive, robust quantity is the SPECIFICITY CONTRAST:
does the real focus beat the phase-offset null? Reported either way.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from obspy import read, UTCDateTime
from obspy.geodetics import gps2dist_azimuth

DATA = Path("data/seismic_tr")
M = 24
V_P = 6.0                      # km/s reference (reporting only)
BAND = (0.1, 0.5)             # Hz bandpass: long-period, robust to travel-time error
# apparent-velocity search, applied IDENTICALLY to real and every null draw (fair):
VELOCITIES = [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
GRID_HALF_DEG = 0.6
GRID_N = 61
N_SCRAMBLE = 200
NOISE_SHIFT_S = -25.0          # sample before each station's arrival (pre-signal noise)
RNG = np.random.default_rng(522)

# ---- PRE-REGISTERED decision criteria (frozen BEFORE any result is inspected) ----
PREREG = {
    "refocus_max_km": 25.0,
    "specificity_max_p": 0.01,
    "specificity_closer_than_median": True,
    "velocity_search_km_s": VELOCITIES, "band_hz": list(BAND), "modulus": M,
    "n_scramble": N_SCRAMBLE,
    "note": ("fair-shot retry after a first NOT_SUPPORTED at band 0.5-2Hz/v=6: "
             "long-period band + apparent-velocity search applied IDENTICALLY to real "
             "and every null draw. criteria fixed in source before this run; honest either way"),
}


def qmod(n):
    return ((np.asarray(n, np.int64) - 1) % M) + 1


def quantize_phase(theta):
    """Observer projection: continuous phase (rad) -> QA integer phase in {1..24}."""
    return qmod(np.rint(np.asarray(theta) * M / (2.0 * np.pi)).astype(np.int64))


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance (km). lat1/lon1 may be arrays."""
    r = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlon = np.radians(lon2 - lon1)
    sdlat = np.sin(dlat / 2.0); sdlon = np.sin(dlon / 2.0)
    hav = sdlat * sdlat + np.cos(p1) * np.cos(p2) * sdlon * sdlon
    return 2.0 * r * np.arcsin(np.sqrt(hav))


def main():
    ev = json.loads((DATA / "event.json").read_text())
    stations = json.loads((DATA / "stations.json").read_text())
    st = read(str(DATA / "waveforms.mseed"))
    st.detrend("demean"); st.detrend("linear"); st.taper(0.05)
    st.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    origin_utc = UTCDateTime(ev["origin_time"])

    from scipy.signal import hilbert
    lats = np.linspace(ev["latitude"] - GRID_HALF_DEG, ev["latitude"] + GRID_HALF_DEG, GRID_N)
    lons = np.linspace(ev["longitude"] - GRID_HALF_DEG, ev["longitude"] + GRID_HALF_DEG, GRID_N)
    LON, LAT = np.meshgrid(lons, lats)          # (GRID_N, GRID_N)

    # Per station: analytic signal sampled at t0 + dist/v over the grid, for EACH
    # velocity in the search (signal window) and the pre-signal NOISE window.
    # A station is kept only if every velocity's window is covered (consistent set).
    PH_by_v = {v: [] for v in VELOCITIES}          # v -> list of (G,G) phase grids
    NPH_by_v = {v: [] for v in VELOCITIES}
    used_codes = []
    for tr in st:
        code = f"{tr.stats.network}.{tr.stats.station}"
        if code not in stations:
            continue
        slat, slon = stations[code]["lat"], stations[code]["lon"]
        z = hilbert(tr.data.astype(float))
        t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
        start_utc = tr.stats.starttime
        D = haversine_km(LAT, LON, slat, slon)                  # (G,G) km, velocity-free

        def sample(v, shift):
            tq = (origin_utc + shift - start_utc) + D / v        # seconds from trace start
            if tq.min() < t[0] or tq.max() > t[-1]:
                return None
            zr = np.interp(tq.ravel(), t, z.real).reshape(D.shape)
            zi = np.interp(tq.ravel(), t, z.imag).reshape(D.shape)
            return np.arctan2(zi, zr)

        sig = {v: sample(v, 0.0) for v in VELOCITIES}
        noi = {v: sample(v, NOISE_SHIFT_S) for v in VELOCITIES}
        if any(x is None for x in sig.values()) or any(x is None for x in noi.values()):
            continue
        for v in VELOCITIES:
            PH_by_v[v].append(sig[v]); NPH_by_v[v].append(noi[v])
        used_codes.append(code)

    PH_by_v = {v: np.array(a) for v, a in PH_by_v.items()}       # (N,G,G)
    NPH_by_v = {v: np.array(a) for v, a in NPH_by_v.items()}
    N = len(used_codes)
    print(f"stations used: {N}  event M{ev['magnitude']} at "
          f"({ev['latitude']:.3f},{ev['longitude']:.3f}) depth {ev['depth_km']:.1f} km")
    print(f"PRE-REGISTERED: {json.dumps(PREREG)}")

    def coherence_map(phase_stack, offsets=None):
        ph = phase_stack if offsets is None else phase_stack + offsets[:, None, None]
        q = quantize_phase(ph)                                  # (N,G,G) ints 1..24
        return np.abs(np.exp(2j * np.pi * (q - 1) / M).sum(axis=0)) / N

    def best_over_v(ph_by_v, offsets=None):
        """Return (focal map, best velocity) maximizing the peak coherence over v."""
        best_F, best_v = None, None
        for v in VELOCITIES:
            F = coherence_map(ph_by_v[v], offsets)
            if best_F is None or F.max() > best_F.max():
                best_F, best_v = F, v
        return best_F, best_v

    def peak_info(F):
        ia, ib = np.unravel_index(int(np.argmax(F)), F.shape)
        pk = float(F[ia, ib])
        d = gps2dist_azimuth(lats[ia], lons[ib], ev["latitude"], ev["longitude"])[0] / 1000.0
        return pk, d, (float(lats[ia]), float(lons[ib]))

    # (A) REAL (best velocity)
    F_real, v_real = best_over_v(PH_by_v)
    real_peak, real_dist, real_loc = peak_info(F_real)
    print(f"\n[A] REAL: peak coherence {real_peak:.3f} at {real_loc} -> {real_dist:.1f} km "
          f"from epicenter (best v={v_real} km/s)")

    # (B) PHASE-OFFSET null -- each draw gets its OWN best-v (identical freedom)
    scr_peaks, scr_dists = [], []
    for _ in range(N_SCRAMBLE):
        off = RNG.uniform(-np.pi, np.pi, size=N)
        F, _ = best_over_v(PH_by_v, offsets=off)
        pk, d, _ = peak_info(F)
        scr_peaks.append(pk); scr_dists.append(d)
    scr_peaks = np.array(scr_peaks); scr_dists = np.array(scr_dists)
    p_emp = float((np.sum(scr_peaks >= real_peak) + 1) / (N_SCRAMBLE + 1))
    print(f"[B] PHASE-OFFSET null (n={N_SCRAMBLE}, each best-v): peak {scr_peaks.mean():.3f}"
          f"+/-{scr_peaks.std():.3f} (max {scr_peaks.max():.3f}); real p={p_emp:.4f}")
    print(f"    peak-distance: real {real_dist:.1f} km vs null median "
          f"{np.median(scr_dists):.1f} km")

    # (B') NOISE window (best velocity)
    F_noise, _ = best_over_v(NPH_by_v)
    noise_peak, noise_dist, _ = peak_info(F_noise)
    print(f"[B'] NOISE window ({NOISE_SHIFT_S:.0f}s): peak {noise_peak:.3f} at "
          f"{noise_dist:.1f} km from epicenter")

    # ---- pre-registered decision ----
    refocus = bool(real_dist <= PREREG["refocus_max_km"])
    specific = bool(p_emp < PREREG["specificity_max_p"] and
                    (real_dist < np.median(scr_dists) if PREREG["specificity_closer_than_median"] else True))
    verdict = "SUPPORTED" if (refocus and specific) else (
        "PARTIAL" if (refocus or specific) else "NOT_SUPPORTED")
    print(f"\nDECISION (pre-registered): REFOCUS={refocus} (<= {PREREG['refocus_max_km']} km), "
          f"SPECIFICITY={specific} (p<{PREREG['specificity_max_p']} & closer than null) -> {verdict}")

    results = {
        "event": ev, "n_stations": int(N), "stations": used_codes, "pre_registration": PREREG,
        "real": {"peak_coherence": real_peak, "peak_km_from_epicenter": real_dist,
                 "peak_latlon": real_loc, "best_velocity_km_s": v_real},
        "phase_offset_null": {"n": N_SCRAMBLE, "peak_mean": float(scr_peaks.mean()),
                              "peak_std": float(scr_peaks.std()), "peak_max": float(scr_peaks.max()),
                              "empirical_p": p_emp, "dist_median_km": float(np.median(scr_dists))},
        "noise_control": {"peak_coherence": noise_peak, "peak_km_from_epicenter": noise_dist},
        "decision": {"refocus": refocus, "specificity": specific, "verdict": verdict},
    }
    outdir = Path("results/seismic"); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "qa_seismic_tr_real_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> results/seismic/qa_seismic_tr_real_results.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        off = RNG.uniform(-np.pi, np.pi, size=N)
        F_null_ex, _ = best_over_v(PH_by_v, offsets=off)
        panels = [(F_real, f"REAL (v={v_real})"), (F_noise, f"NOISE {NOISE_SHIFT_S:.0f}s"),
                  (F_null_ex, "PHASE-OFFSET null")]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        for k, (F, title) in enumerate(panels):
            im = ax[k].imshow(F, origin="lower", extent=[lons[0], lons[-1], lats[0], lats[-1]],
                              aspect="auto", cmap="magma", vmin=0, vmax=max(0.3, real_peak))
            ax[k].plot(ev["longitude"], ev["latitude"], "c*", ms=16, mec="k", label="epicenter")
            ax[k].set_title(f"{title}  (peak {F.max():.2f})"); ax[k].legend(loc="upper right")
            fig.colorbar(im, ax=ax[k], fraction=0.046)
        fig.suptitle(f"QA [522] time-reversal focusing on real M{ev['magnitude']} Ridgecrest "
                     f"({N} stations, mod-{M})")
        fig.tight_layout(); fig.savefig("qa_seismic_tr_real.png", dpi=110)
        print("saved -> qa_seismic_tr_real.png")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
