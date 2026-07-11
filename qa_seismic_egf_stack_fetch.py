#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=real_seismic_egf_cluster_download; no QA state — data acquisition for the stacked-EGF [522] test"
"""
Fetch data for the STACKED empirical-Green's-function (EGF) specificity test
(cert [522] significance push).

  T            = target moderate event (M~4.9-5.2, Ridgecrest 2019)
  matched set  = a CLUSTER of co-located aftershocks (<5 km of T, well recorded)
                 -> stacked EGF for T's source patch; offset phases average out
  mismatched   = a CLUSTER co-located at a DISTANT anchor (~80-150 km away) -> a
                 valid EGF for the WRONG patch (fair same-structure control)

Saves to data/seismic_egf_stack/: manifest.json, stations.json, and one
<role>_<idx>.mseed per event (T_0, M_0..M_k matched, X_0..X_j mismatched).
Pure observer-layer acquisition.
"""
from __future__ import annotations
import json
from pathlib import Path

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

OUT = Path("data/seismic_egf_stack")
OUT.mkdir(parents=True, exist_ok=True)
PRE, POST = 10.0, 90.0
KMAX = 30                    # cap events per cluster


def hypo_km(a, b):
    horiz = gps2dist_azimuth(a["latitude"], a["longitude"], b["latitude"], b["longitude"])[0] / 1000.0
    dz = (a.get("depth_km") or 0.0) - (b.get("depth_km") or 0.0)
    return (horiz * horiz + dz * dz) ** 0.5


def evdict(ev):
    o = ev.preferred_origin() or ev.origins[0]
    m = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
    return {"event_id": str(ev.resource_id), "origin_time": str(o.time),
            "latitude": float(o.latitude), "longitude": float(o.longitude),
            "depth_km": float(o.depth) / 1000.0 if o.depth else 0.0, "magnitude": float(m)}


def main():
    ev_client = Client("USGS")
    wf = Client("EARTHSCOPE")

    def safe_events(**kw):
        try:
            return list(ev_client.get_events(**kw))
        except Exception:
            return []

    T = evdict(max(safe_events(starttime=UTCDateTime("2019-07-05"),
                                        endtime=UTCDateTime("2019-07-12"),
                                        minmagnitude=4.4, maxmagnitude=5.3,
                                        minlatitude=35.5, maxlatitude=36.1,
                                        minlongitude=-117.9, maxlongitude=-117.3),
                   key=lambda e: (e.preferred_magnitude() or e.magnitudes[0]).mag))
    print(f"T : M{T['magnitude']} {T['origin_time']} ({T['latitude']:.3f},{T['longitude']:.3f})")

    # matched cluster: co-located aftershocks, well recorded
    catM = safe_events(starttime=UTCDateTime("2019-07-05"), endtime=UTCDateTime("2019-08-15"),
                                minmagnitude=3.0, maxmagnitude=4.5,
                                latitude=T["latitude"], longitude=T["longitude"], maxradius=0.06)
    matched = [c for c in (evdict(e) for e in catM)
               if 0.2 < hypo_km(c, T) < 5.0 and c["magnitude"] < T["magnitude"]]
    matched.sort(key=lambda c: -c["magnitude"]); matched = matched[:KMAX]
    if not matched:
        raise SystemExit("no matched co-located cluster found; widen the search")
    print(f"matched cluster: {len(matched)} events, "
          f"{min(hypo_km(c,T) for c in matched):.1f}-{max(hypo_km(c,T) for c in matched):.1f} km from T")

    # distant anchor + its co-located cluster (mismatched: right structure, wrong patch)
    catA = []
    for (mmin, r0, r1) in [(3.8, 0.6, 1.6), (3.2, 0.55, 1.7), (2.8, 0.5, 1.8), (2.5, 0.5, 1.9)]:
        catA = safe_events(starttime=UTCDateTime("2019-01-01"), endtime=UTCDateTime("2019-12-31"),
                           minmagnitude=mmin, maxmagnitude=5.8,
                           latitude=T["latitude"], longitude=T["longitude"],
                           minradius=r0, maxradius=r1)
        if catA:
            print(f"    (distant-anchor search: M>={mmin}, ring {r0}-{r1} deg -> {len(catA)} events)")
            break
    if not catA:
        raise SystemExit("no distant anchor found; widen the search")
    anchor = min((evdict(e) for e in catA), key=lambda c: abs(hypo_km(c, T) - 110.0))
    print(f"distant anchor: M{anchor['magnitude']} {hypo_km(anchor,T):.1f} km from T "
          f"({anchor['latitude']:.3f},{anchor['longitude']:.3f})")
    catX = safe_events(starttime=UTCDateTime("2019-01-01"), endtime=UTCDateTime("2019-12-31"),
                                minmagnitude=3.2, maxmagnitude=5.5,
                                latitude=anchor["latitude"], longitude=anchor["longitude"], maxradius=0.15)
    mismatched = [c for c in (evdict(e) for e in catX) if hypo_km(c, T) > 60.0]
    mismatched.sort(key=lambda c: -c["magnitude"]); mismatched = mismatched[:KMAX]
    print(f"mismatched cluster: {len(mismatched)} events near the distant anchor")

    inv = wf.get_stations(network="CI,NN,PB,ZY,GS", channel="BHZ,HHZ,EHZ,HNZ", level="channel",
                          latitude=T["latitude"], longitude=T["longitude"], maxradius=1.2,
                          starttime=UTCDateTime(T["origin_time"]), endtime=UTCDateTime(T["origin_time"]) + 200)
    stations = {}
    for net in inv:
        for sta in net:
            stations.setdefault(f"{net.code}.{sta.code}",
                                {"net": net.code, "sta": sta.code, "lat": float(sta.latitude),
                                 "lon": float(sta.longitude)})

    pref = {"BHZ": 0, "HHZ": 1, "EHZ": 2, "HNZ": 3}

    def fetch(ev, tag):
        o = UTCDateTime(ev["origin_time"])
        bulk = [(s["net"], s["sta"], "*", ch, o - PRE, o + POST)
                for s in stations.values() for ch in ("BHZ", "HHZ", "EHZ", "HNZ")]
        try:
            st = wf.get_waveforms_bulk(bulk)
        except Exception:
            return []
        kept = {}
        for tr in st:
            c = f"{tr.stats.network}.{tr.stats.station}"
            if c not in kept or pref.get(tr.stats.channel, 9) < pref.get(kept[c].stats.channel, 9):
                kept[c] = tr
        if not kept:
            return []
        Stream(list(kept.values())).write(str(OUT / f"{tag}.mseed"), format="MSEED")
        return sorted(kept)

    manifest = {"T": {**T, "tag": "T_0", "stations": fetch(T, "T_0")}, "matched": [], "mismatched": []}
    for i, e in enumerate(matched):
        codes = fetch(e, f"M_{i}")
        if codes:
            manifest["matched"].append({**e, "tag": f"M_{i}", "hypo_km_to_T": hypo_km(e, T), "stations": codes})
    for i, e in enumerate(mismatched):
        codes = fetch(e, f"X_{i}")
        if codes:
            manifest["mismatched"].append({**e, "tag": f"X_{i}", "hypo_km_to_T": hypo_km(e, T), "stations": codes})

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "stations.json").write_text(json.dumps(stations, indent=2))
    print(f"saved -> {OUT}/ : T + {len(manifest['matched'])} matched + "
          f"{len(manifest['mismatched'])} mismatched events")


if __name__ == "__main__":
    main()
