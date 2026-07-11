#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=real_seismic_egf_download; no QA state — pure data acquisition for the [522] EGF specificity test"
"""
Fetch three real events (all recorded by a COMMON station set) for the cert [522]
empirical-Green's-function (EGF) specificity test:

  T  = target moderate event (M~4.5-5.2, Ridgecrest 2019)
  E  = CO-LOCATED small companion (M~2.3-3.4, within a few km of T) -> MATCHED medium
       (its records ARE the medium's Green's function from T's source patch)
  Ep = DISTANT control event (~70-160 km from T, different path to the same stations)
       -> MISMATCHED medium

The [522] idea on real data: R_T * conj(R_EGF) cancels the shared medium (|G|^2 is
real) and stacks coherently for the co-located EGF; a distant EGF has a different
path (G_T != G_Ep) so the product keeps a station-dependent phase and decoheres.

Saves to data/seismic_egf/: events.json, stations.json, and T.mseed / E.mseed /
Ep.mseed (vertical component, common stations, window from origin).
Pure observer-layer acquisition; no QA arithmetic.
"""
from __future__ import annotations
import json
from pathlib import Path

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

OUT = Path("data/seismic_egf")
OUT.mkdir(parents=True, exist_ok=True)
PRE, POST = 10.0, 90.0          # window seconds around each event origin


def hypo_km(a, b):
    horiz = gps2dist_azimuth(a["latitude"], a["longitude"], b["latitude"], b["longitude"])[0] / 1000.0
    dz = (a.get("depth_km") or 0.0) - (b.get("depth_km") or 0.0)
    return (horiz * horiz + dz * dz) ** 0.5


def evdict(ev):
    o = ev.preferred_origin() or ev.origins[0]
    m = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
    return {"event_id": str(ev.resource_id), "origin_time": str(o.time),
            "latitude": float(o.latitude), "longitude": float(o.longitude),
            "depth_km": float(o.depth) / 1000.0 if o.depth else 0.0, "magnitude": float(m),
            "_origin": o}


def main():
    ev_client = Client("USGS")
    wf = Client("EARTHSCOPE")

    # --- T: moderate Ridgecrest event ---
    catT = ev_client.get_events(starttime=UTCDateTime("2019-07-05"), endtime=UTCDateTime("2019-07-12"),
                                minmagnitude=4.4, maxmagnitude=5.3,
                                minlatitude=35.5, maxlatitude=36.1,
                                minlongitude=-117.9, maxlongitude=-117.3)
    T = evdict(max(catT, key=lambda e: (e.preferred_magnitude() or e.magnitudes[0]).mag))
    print(f"T  : M{T['magnitude']} {T['origin_time']} ({T['latitude']:.3f},{T['longitude']:.3f}) "
          f"z={T['depth_km']:.1f}km")

    # --- E: co-located companion (matched EGF): LARGEST event within ~5 km of T
    # (well-recorded for SNR, still sharing T's source-patch medium) ---
    catE = ev_client.get_events(starttime=UTCDateTime("2019-07-05"), endtime=UTCDateTime("2019-07-31"),
                                minmagnitude=3.0, maxmagnitude=4.4,
                                latitude=T["latitude"], longitude=T["longitude"], maxradius=0.1)
    cands = [evdict(e) for e in catE]
    # co-located (<5 km) and smaller than T; take the LARGEST -> best SNR as an EGF.
    # (The closest events are too small to clear SNR at regional stations; largest
    # co-located is the best-usable empirical Green's function -- see docs/families/522.)
    cands = [c for c in cands if 0.2 < hypo_km(c, T) < 5.0 and c["magnitude"] < T["magnitude"]]
    E = max(cands, key=lambda c: c["magnitude"])
    print(f"E  : M{E['magnitude']} {E['origin_time']} ({E['latitude']:.3f},{E['longitude']:.3f}) "
          f"z={E['depth_km']:.1f}km  -> {hypo_km(E, T):.1f} km from T (co-located EGF)")

    # --- Ep: distant control (mismatched medium), same station footprint ---
    # widen until we find a well-recorded event on a DIFFERENT path (~70-170 km from T)
    catP = None
    for (mmin, r0, r1) in [(3.8, 0.7, 1.4), (3.2, 0.6, 1.6), (2.8, 0.55, 1.7), (2.5, 0.5, 1.8)]:
        try:
            catP = ev_client.get_events(starttime=UTCDateTime("2019-01-01"),
                                        endtime=UTCDateTime("2019-12-31"),
                                        minmagnitude=mmin, maxmagnitude=5.8,
                                        latitude=T["latitude"], longitude=T["longitude"],
                                        minradius=r0, maxradius=r1)
            if len(catP) > 0:
                print(f"    (distant-control search: M>={mmin}, ring {r0}-{r1} deg -> {len(catP)} events)")
                break
        except Exception:
            catP = None
    if not catP:
        raise SystemExit("no distant control event found; widen the search")
    Ep = min((evdict(e) for e in catP), key=lambda c: abs(hypo_km(c, T) - 110.0))
    print(f"Ep : M{Ep['magnitude']} {Ep['origin_time']} ({Ep['latitude']:.3f},{Ep['longitude']:.3f}) "
          f"z={Ep['depth_km']:.1f}km  -> {hypo_km(Ep, T):.1f} km from T (distant control)")

    # --- stations: CI vertical near T ---
    inv = wf.get_stations(network="CI,NN,PB,ZY,GS", channel="BHZ,HHZ,EHZ,HNZ", level="channel",
                          latitude=T["latitude"], longitude=T["longitude"], maxradius=1.2,
                          starttime=UTCDateTime(T["origin_time"]), endtime=UTCDateTime(T["origin_time"]) + 200)
    stations = {}
    for net in inv:
        for sta in net:
            code = f"{net.code}.{sta.code}"
            stations.setdefault(code, {"net": net.code, "sta": sta.code, "lat": float(sta.latitude),
                                       "lon": float(sta.longitude), "elev_m": float(sta.elevation)})
    print(f"candidate stations near T: {len(stations)}")

    def fetch(ev):
        o = UTCDateTime(ev["origin_time"])
        bulk = [(s["net"], s["sta"], "*", ch, o - PRE, o + POST)
                for s in stations.values() for ch in ("BHZ", "HHZ", "EHZ", "HNZ")]
        st = wf.get_waveforms_bulk(bulk)
        pref = {"BHZ": 0, "HHZ": 1, "EHZ": 2, "HNZ": 3}
        kept = {}
        for tr in st:
            c = f"{tr.stats.network}.{tr.stats.station}"
            if c not in kept or pref.get(tr.stats.channel, 9) < pref.get(kept[c].stats.channel, 9):
                kept[c] = tr
        return kept

    kT, kE, kP = fetch(T), fetch(E), fetch(Ep)
    common = sorted(set(kT) & set(kE) & set(kP))
    print(f"common stations across T, E, Ep: {len(common)}")

    stations = {c: stations[c] for c in common}
    Stream([kT[c] for c in common]).write(str(OUT / "T.mseed"), format="MSEED")
    Stream([kE[c] for c in common]).write(str(OUT / "E.mseed"), format="MSEED")
    Stream([kP[c] for c in common]).write(str(OUT / "Ep.mseed"), format="MSEED")
    for d in (T, E, Ep):
        d.pop("_origin", None)
    (OUT / "events.json").write_text(json.dumps({"T": T, "E": E, "Ep": Ep,
                                                 "hypo_km_E_to_T": hypo_km(E, T),
                                                 "hypo_km_Ep_to_T": hypo_km(Ep, T)}, indent=2))
    (OUT / "stations.json").write_text(json.dumps(stations, indent=2))
    print(f"saved -> {OUT}/ (events.json, stations.json, T/E/Ep.mseed)")


if __name__ == "__main__":
    main()
