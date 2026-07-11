#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=real_seismic_egf_target2_download; no QA state — data acquisition for the [522] EGF replication + foreshock control"
"""
Second-target data for the cert [522] EGF specificity test: replication on an
INDEPENDENT source patch + a FORESHOCK-vs-AFTERSHOCK time-symmetry control.

  T2           = a moderate event in the SW Ridgecrest zone (near the M6.4 foreshock
                 ~35.70,-117.51) -- a DIFFERENT source patch from target 1 (35.90,
                 -117.70), so different paths/medium to the same network.
  matched set  = co-located events (<5 km of T2) spanning BOTH before (foreshocks)
                 and after (aftershocks) T2's origin. The Green's function is
                 time-invariant, so a co-located companion is a valid EGF regardless
                 of when it occurred -- the experiment splits fore/after by time.
  mismatched   = a distant cluster (~80-160 km) -> valid EGF for the wrong patch.

Saves to data/seismic_egf_t2/ (same layout as qa_seismic_egf_stack_fetch.py:
manifest.json with per-event origin_time, stations.json, <tag>.mseed).
"""
from __future__ import annotations
import json
from pathlib import Path

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

OUT = Path("data/seismic_egf_t2")
OUT.mkdir(parents=True, exist_ok=True)
PRE, POST = 10.0, 90.0
KMAX = 30
SW = (35.70, -117.51)             # M6.4 foreshock zone -- the second source patch


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

    # T2: largest M4.4-5.3 near the SW patch (a different source region than target 1)
    catT = safe_events(starttime=UTCDateTime("2019-07-04"), endtime=UTCDateTime("2019-07-20"),
                       minmagnitude=4.4, maxmagnitude=5.3,
                       latitude=SW[0], longitude=SW[1], maxradius=0.12)
    if not catT:
        raise SystemExit("no T2 candidate in the SW zone")
    T = evdict(max(catT, key=lambda e: (e.preferred_magnitude() or e.magnitudes[0]).mag))
    print(f"T2 : M{T['magnitude']} {T['origin_time']} ({T['latitude']:.3f},{T['longitude']:.3f}) "
          f"z={T['depth_km']:.1f}km  ({hypo_km(T, {'latitude':35.904,'longitude':-117.700,'depth_km':8.3}):.0f} km from target 1)")

    # co-located cluster spanning fore + after (wide time window)
    catM = safe_events(starttime=UTCDateTime("2019-07-04"), endtime=UTCDateTime("2019-09-01"),
                       minmagnitude=3.0, maxmagnitude=4.6,
                       latitude=T["latitude"], longitude=T["longitude"], maxradius=0.06)
    matched = [c for c in (evdict(e) for e in catM)
               if 0.2 < hypo_km(c, T) < 5.0 and c["magnitude"] < T["magnitude"]]
    matched.sort(key=lambda c: -c["magnitude"]); matched = matched[:KMAX]
    if not matched:
        raise SystemExit("no matched co-located cluster for T2")
    n_fore = sum(1 for c in matched if UTCDateTime(c["origin_time"]) < UTCDateTime(T["origin_time"]))
    print(f"matched cluster: {len(matched)} events ({n_fore} foreshocks, {len(matched)-n_fore} aftershocks), "
          f"{min(hypo_km(c,T) for c in matched):.1f}-{max(hypo_km(c,T) for c in matched):.1f} km from T2")

    # distant anchor + cluster (mismatched)
    catA = []
    for (mmin, r0, r1) in [(3.8, 0.6, 1.6), (3.2, 0.55, 1.7), (2.8, 0.5, 1.8), (2.5, 0.5, 1.9)]:
        catA = safe_events(starttime=UTCDateTime("2019-01-01"), endtime=UTCDateTime("2019-12-31"),
                           minmagnitude=mmin, maxmagnitude=5.8,
                           latitude=T["latitude"], longitude=T["longitude"], minradius=r0, maxradius=r1)
        if catA:
            break
    if not catA:
        raise SystemExit("no distant anchor for T2")
    anchor = min((evdict(e) for e in catA), key=lambda c: abs(hypo_km(c, T) - 110.0))
    catX = safe_events(starttime=UTCDateTime("2019-01-01"), endtime=UTCDateTime("2019-12-31"),
                       minmagnitude=3.0, maxmagnitude=5.5,
                       latitude=anchor["latitude"], longitude=anchor["longitude"], maxradius=0.2)
    mismatched = [c for c in (evdict(e) for e in catX) if hypo_km(c, T) > 60.0]
    mismatched.sort(key=lambda c: -c["magnitude"]); mismatched = mismatched[:KMAX]
    print(f"distant control: {len(mismatched)} events near anchor M{anchor['magnitude']} "
          f"{hypo_km(anchor,T):.0f} km from T2")

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
    print(f"saved -> {OUT}/ : T2 + {len(manifest['matched'])} matched + "
          f"{len(manifest['mismatched'])} mismatched events")


if __name__ == "__main__":
    main()
