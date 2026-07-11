#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=synthetic_phase_selftest; validates the back-projection estimator, no QA state feedback"
# RT1_OBSERVER_FILE: synthetic wavelet phases + geometry are observer-layer, not QA state.
"""
Sanity check for qa_seismic_tr_real.py's estimator: does the SAME QA mod-24 phase-
coherence back-projection recover a KNOWN source when the medium is simple and known
(constant velocity, clean arrivals, no noise)?

If YES -> the estimator is sound, so the NOT_SUPPORTED result on real data is a
genuine finding (unknown 3-D structure + mod-24 quantization destroy the coherence),
not a broken method. If NO -> the method itself is the problem and the real-data
negative would be uninterpretable. Honest guard against reporting a bug as physics.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

from qa_seismic_tr_real import (DATA, M, GRID_HALF_DEG, GRID_N, haversine_km,
                                quantize_phase)

V_TRUE = 6.0            # known constant velocity for the synthetic
F0 = 0.3               # Hz carrier of the synthetic wavelet (in the real band)
FS = 20.0              # Hz sample rate


def main():
    ev = json.loads((DATA / "event.json").read_text())
    stations = json.loads((DATA / "stations.json").read_text())
    origin = UTCDateTime(ev["origin_time"])
    src_lat, src_lon = ev["latitude"], ev["longitude"]

    lats = np.linspace(src_lat - GRID_HALF_DEG, src_lat + GRID_HALF_DEG, GRID_N)
    lons = np.linspace(src_lon - GRID_HALF_DEG, src_lon + GRID_HALF_DEG, GRID_N)
    LON, LAT = np.meshgrid(lons, lats)

    # Build a synthetic analytic phase per station: a Gaussian-tapered tone whose
    # arrival is exactly origin + dist(source,station)/V_TRUE. Sampling this at the
    # predicted arrival for a candidate g gives coherent phases ONLY at g = source.
    t = np.arange(int(180 * FS)) / FS                      # 180 s trace from origin-60
    start = origin - 60.0
    PH = []
    for code, s in stations.items():
        d_true = gps2dist_azimuth(src_lat, src_lon, s["lat"], s["lon"])[0] / 1000.0
        t_arr = (origin - start) + d_true / V_TRUE          # arrival, sec from trace start
        env = np.exp(-0.5 * ((t - t_arr) / 3.0) ** 2)       # 3 s Gaussian envelope
        sig = env * np.cos(2 * np.pi * F0 * (t - t_arr))
        z = np.asarray(np.abs(1) * (sig + 1j * env * np.sin(2 * np.pi * F0 * (t - t_arr))))
        D = haversine_km(LAT, LON, s["lat"], s["lon"])
        tau = D / V_TRUE
        tq = (origin - start) + tau
        zr = np.interp(tq.ravel(), t, z.real).reshape(D.shape)
        zi = np.interp(tq.ravel(), t, z.imag).reshape(D.shape)
        PH.append(np.arctan2(zi, zr))
    PH = np.array(PH)
    N = PH.shape[0]

    q = quantize_phase(PH)
    F = np.abs(np.exp(2j * np.pi * (q - 1) / M).sum(axis=0)) / N
    ia, ib = np.unravel_index(int(np.argmax(F)), F.shape)
    d_km = gps2dist_azimuth(lats[ia], lons[ib], src_lat, src_lon)[0] / 1000.0
    grid_step_km = gps2dist_azimuth(lats[0], lons[0], lats[1], lons[0])[0] / 1000.0

    print(f"SYNTHETIC self-test (known v={V_TRUE} km/s, {N} stations, clean arrivals):")
    print(f"  peak coherence {F.max():.3f} at ({lats[ia]:.3f},{lons[ib]:.3f}) -> "
          f"{d_km:.1f} km from true source (grid step {grid_step_km:.1f} km)")
    ok = d_km <= 2 * grid_step_km and F.max() > 0.9
    print(f"  ESTIMATOR {'RECOVERS the source (method sound)' if ok else 'FAILS (method suspect)'}: "
          f"peak within 2 grid steps and coherence>0.9 -> {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
