#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=real_seismic_download; no QA state here — pure data acquisition for the [522] time-reversal experiment"
"""
Fetch a real, well-recorded regional earthquake + a dense vertical-component array
for the QA time-reversal focusing experiment (cert [522] on real data).

Target: the 2019 Ridgecrest, CA sequence — extremely dense CI-network coverage,
well-located catalog hypocenters. We pick a MODERATE (M~5) event so the source is
close to a point (an M7 mainshock has an extended rupture, not a point focus).

Saves to data/seismic_tr/:
  event.json     — chosen event (id, origin time, lat, lon, depth, mag)
  stations.json  — per-station code + lat/lon/elev
  waveforms.mseed— vertical-component records, window around the origin

Pure observer-layer acquisition: no QA arithmetic here.
"""
from __future__ import annotations
import json
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees

OUT = Path("data/seismic_tr")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ev_client = Client("USGS")        # event catalog (ComCat)
    client = Client("EARTHSCOPE")     # stations + waveforms (formerly IRIS)

    # Ridgecrest sequence window; grab moderate events and pick the best-magnitude M5.x
    t0, t1 = UTCDateTime("2019-07-04T00:00:00"), UTCDateTime("2019-07-12T00:00:00")
    cat = ev_client.get_events(starttime=t0, endtime=t1, minmagnitude=4.9, maxmagnitude=5.6,
                            minlatitude=35.4, maxlatitude=36.1,
                            minlongitude=-117.9, maxlongitude=-117.2)
    print(f"candidate events: {len(cat)}")
    # deterministic pick: the largest magnitude in the M5 band
    ev = max(cat, key=lambda e: (e.preferred_magnitude() or e.magnitudes[0]).mag)
    origin = ev.preferred_origin() or ev.origins[0]
    mag = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
    evd = {
        "event_id": str(ev.resource_id),
        "origin_time": str(origin.time),
        "latitude": float(origin.latitude),
        "longitude": float(origin.longitude),
        "depth_km": float(origin.depth) / 1000.0 if origin.depth else None,
        "magnitude": float(mag),
    }
    print(f"chosen event: M{mag} {origin.time}  ({origin.latitude:.3f},{origin.longitude:.3f}) "
          f"depth {evd['depth_km']:.1f} km")

    # dense vertical-component stations within ~1.2 deg (broadband, short-period,
    # strong-motion — whatever recorded this event), across the regional networks
    inv = client.get_stations(network="CI,NN,PB,ZY,GS", channel="BHZ,HHZ,EHZ,HNZ",
                              level="channel",
                              latitude=origin.latitude, longitude=origin.longitude,
                              maxradius=1.2, starttime=origin.time, endtime=origin.time + 300)
    stations = {}
    for net in inv:
        for sta in net:
            # one entry per station; prefer BHZ, fall back to HHZ
            code = f"{net.code}.{sta.code}"
            if code in stations:
                continue
            deg = locations2degrees(origin.latitude, origin.longitude, sta.latitude, sta.longitude)
            stations[code] = {"net": net.code, "sta": sta.code,
                              "lat": float(sta.latitude), "lon": float(sta.longitude),
                              "elev_m": float(sta.elevation), "dist_deg": float(deg)}
    print(f"stations within 1.2 deg: {len(stations)}")

    # bulk waveform request: 20 s before origin to 120 s after (captures P moveout)
    bulk = []
    for code, s in stations.items():
        for ch in ("BHZ", "HHZ", "EHZ", "HNZ"):
            bulk.append((s["net"], s["sta"], "*", ch, origin.time - 60, origin.time + 120))
    st = client.get_waveforms_bulk(bulk)
    # keep one trace per station, preferring broadband (BHZ > HHZ > EHZ > HNZ)
    pref = {"BHZ": 0, "HHZ": 1, "EHZ": 2, "HNZ": 3}
    kept = {}
    for tr in st:
        code = f"{tr.stats.network}.{tr.stats.station}"
        if code not in kept or pref.get(tr.stats.channel, 9) < pref.get(kept[code].stats.channel, 9):
            kept[code] = tr
    from obspy import Stream
    st_keep = Stream(traces=list(kept.values()))
    print(f"waveform traces kept (1/station): {len(st_keep)}")

    # keep only stations we actually have waveforms for
    stations = {c: stations[c] for c in kept if c in stations}

    (OUT / "event.json").write_text(json.dumps(evd, indent=2))
    (OUT / "stations.json").write_text(json.dumps(stations, indent=2))
    st_keep.write(str(OUT / "waveforms.mseed"), format="MSEED")
    print(f"saved -> {OUT}/  (event.json, stations.json, waveforms.mseed)")


if __name__ == "__main__":
    main()
