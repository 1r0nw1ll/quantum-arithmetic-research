#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=real_seismic_egf_target3_download; no QA state — data acquisition for a CLEAN-window [522] foreshock time-symmetry test"
"""
Third-target data for the cert [522] EGF foreshock time-symmetry test, chosen so BOTH
the foreshock AND aftershock windows are CLEAN (T2 failed because it sat 3 min before
the M7.1, contaminating its aftershock window).

T3 selection: a moderate event (M 4.6-5.3) that is a LOCAL MAGNITUDE MAXIMUM -- no
event with magnitude >= mag(T3)+0.2 occurs within +/-6 h and 20 km. That guarantees
its aftershock window is not overprinted by a larger event, and (symmetrically) its
foreshock window either. Preference for later in the sequence, where the M7.1 cascade
has decayed.

Saves to data/seismic_egf_t3/ (same layout as qa_seismic_egf_t2_fetch.py).
"""
from __future__ import annotations
import json
from pathlib import Path

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

OUT = Path("data/seismic_egf_t3")
OUT.mkdir(parents=True, exist_ok=True)
PRE, POST = 10.0, 90.0
KMAX = 30
CLEAN_DT_H = 6.0            # no larger event within +/- this many hours
CLEAN_DKM = 20.0           # ... and within this distance
CLEAN_DMAG = 0.2           # "larger" = mag >= mag(T3) + this
T1 = {"latitude": 35.904, "longitude": -117.700, "depth_km": 8.3}


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

    # regional catalog (for both candidate targets and the larger-neighbour test)
    region = dict(minlatitude=35.4, maxlatitude=36.2, minlongitude=-118.0, maxlongitude=-117.2)
    reg = [evdict(e) for e in safe_events(starttime=UTCDateTime("2019-07-06T04:00:00"),
                                          endtime=UTCDateTime("2019-10-01"),
                                          minmagnitude=3.5, **region)]
    print(f"regional M>=3.5 catalog (post-M7.1): {len(reg)} events")

    def is_clean(c):
        ct = UTCDateTime(c["origin_time"])
        for e in reg:
            if e["event_id"] == c["event_id"]:
                continue
            if (e["magnitude"] >= c["magnitude"] + CLEAN_DMAG
                    and abs(UTCDateTime(e["origin_time"]) - ct) < CLEAN_DT_H * 3600
                    and hypo_km(e, c) < CLEAN_DKM):
                return False, e
        return True, None

    # candidates: M4.6-5.3 local maxima (clean window) WITH enough co-located
    # aftershock production -- pick the most BALANCED fore/after (needs both arms).
    def fore_after(c):
        ct = UTCDateTime(c["origin_time"])
        fore = after = 0
        for e in reg:
            if e["event_id"] == c["event_id"] or hypo_km(e, c) >= 20 or e["magnitude"] >= c["magnitude"]:
                continue
            if UTCDateTime(e["origin_time"]) < ct:
                fore += 1
            else:
                after += 1
        return fore, after

    best = None
    for c in [c for c in reg if 4.6 <= c["magnitude"] <= 5.3]:
        clean, _ = is_clean(c)
        if not clean:
            continue
        fore, after = fore_after(c)                    # M>=3.5 proxy; cluster fetch uses M>=3.0
        if after < 4 or fore < 4:                       # need both arms
            continue
        score = min(fore, after)
        if best is None or score > best[1]:
            best = (c, score, fore, after)
    if best is None:
        raise SystemExit("no clean local-maximum target with balanced fore/after found")
    T, _, pf, pa = best
    print(f"  (selected clean local-max with M>=3.5 proxy balance: {pf} fore / {pa} after)")
    ct = UTCDateTime(T["origin_time"])
    # report the nearest larger event's time gap (confirm cleanliness)
    gaps = [abs(UTCDateTime(e["origin_time"]) - ct) / 3600.0 for e in reg
            if e["magnitude"] > T["magnitude"] and hypo_km(e, T) < CLEAN_DKM]
    nearest_larger_h = min(gaps) if gaps else float("inf")
    print(f"T3 : M{T['magnitude']} {T['origin_time']} ({T['latitude']:.3f},{T['longitude']:.3f}) "
          f"z={T['depth_km']:.1f}km  ({hypo_km(T, T1):.0f} km from target 1); "
          f"nearest larger event within 20km: {nearest_larger_h:.1f} h away (clean window)")

    catM = safe_events(starttime=UTCDateTime("2019-07-06T04:00:00"), endtime=UTCDateTime("2019-10-01"),
                       minmagnitude=2.8, maxmagnitude=4.6,
                       latitude=T["latitude"], longitude=T["longitude"], maxradius=0.10)
    matched = [c for c in (evdict(e) for e in catM)
               if 0.2 < hypo_km(c, T) < 8.0 and c["magnitude"] < T["magnitude"]]
    matched.sort(key=lambda c: -c["magnitude"]); matched = matched[:KMAX]
    if not matched:
        raise SystemExit("no matched co-located cluster for T3")
    n_fore = sum(1 for c in matched if UTCDateTime(c["origin_time"]) < ct)
    print(f"matched cluster: {len(matched)} events ({n_fore} foreshocks, {len(matched)-n_fore} aftershocks), "
          f"{min(hypo_km(c,T) for c in matched):.1f}-{max(hypo_km(c,T) for c in matched):.1f} km from T3")

    catA = []
    for (mmin, r0, r1) in [(3.8, 0.6, 1.6), (3.2, 0.55, 1.7), (2.8, 0.5, 1.8), (2.5, 0.5, 1.9)]:
        catA = safe_events(starttime=UTCDateTime("2019-01-01"), endtime=UTCDateTime("2019-12-31"),
                           minmagnitude=mmin, maxmagnitude=5.8,
                           latitude=T["latitude"], longitude=T["longitude"], minradius=r0, maxradius=r1)
        if catA:
            break
    if not catA:
        raise SystemExit("no distant anchor for T3")
    anchor = min((evdict(e) for e in catA), key=lambda c: abs(hypo_km(c, T) - 110.0))
    catX = safe_events(starttime=UTCDateTime("2019-01-01"), endtime=UTCDateTime("2019-12-31"),
                       minmagnitude=2.6, maxmagnitude=5.5,
                       latitude=anchor["latitude"], longitude=anchor["longitude"], maxradius=0.35)
    mismatched = [c for c in (evdict(e) for e in catX) if hypo_km(c, T) > 55.0]
    mismatched.sort(key=lambda c: -c["magnitude"]); mismatched = mismatched[:KMAX]
    print(f"distant control: {len(mismatched)} events near anchor M{anchor['magnitude']} "
          f"{hypo_km(anchor,T):.0f} km from T3")

    inv = wf.get_stations(network="CI,NN,PB,ZY,GS", channel="BHZ,HHZ,EHZ,HNZ", level="channel",
                          latitude=T["latitude"], longitude=T["longitude"], maxradius=1.2,
                          starttime=ct, endtime=ct + 200)
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

    manifest = {"T": {**T, "tag": "T_0", "nearest_larger_event_h": nearest_larger_h,
                      "stations": fetch(T, "T_0")}, "matched": [], "mismatched": []}
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
    print(f"saved -> {OUT}/ : T3 + {len(manifest['matched'])} matched + "
          f"{len(manifest['mismatched'])} mismatched events")


if __name__ == "__main__":
    main()
