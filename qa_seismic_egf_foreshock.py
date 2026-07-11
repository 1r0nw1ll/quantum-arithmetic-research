#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=stacked_cross_spectral_phase_to_QA mod24; qa_neg=conjugation=time reversal; coherence is observer readout (Theorem NT)"
# RT1_OBSERVER_FILE: seismic cross-spectral phases are observer-layer projections, not QA state.
"""
Cert [522] EGF specificity -- second target (replication) + FORESHOCK-vs-AFTERSHOCK
time-symmetry control.

Same stacked-EGF method as qa_seismic_egf_stack.py (per-event global phase removed,
cross-spectral phasors stacked over a co-located cluster, QA mod-24 phase coherence),
now on an INDEPENDENT source patch (data/seismic_egf_t2/). The co-located cluster is
split by origin time into FORESHOCKS (before T2) and AFTERSHOCKS (after T2), giving
three matched stacks -- fore, after, combined -- each vs the distant control + null.

Two pre-registered claims:
  (R) REPLICATION: on a second target, the combined co-located stack coheres
      (scramble-null p < 0.01) and >> the distant control -- the target-1 SUPPORTED
      result is not event-specific.
  (T) TIME-SYMMETRY (foreshock validity): the Green's function is time-invariant, so
      a FORESHOCK-only stack and an AFTERSHOCK-only stack BOTH cohere and BOTH exceed
      the distant control -- a co-located companion is a valid EGF regardless of
      whether it occurred before or after the target.

qa_neg (conjugation) = time reversal; mod-24 is the QA observer projection.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from obspy import read, UTCDateTime

DATA = Path("data/seismic_egf_t2")
M = 24
FS = 20.0
WIN_S = 50.0
BAND = (0.5, 2.0)
SNR_MIN = 2.0
KMIN = 3
N_NULL = 5000
RNG = np.random.default_rng(522)

PREREG = {
    "replication": "combined co-located stack: coherence beats scramble null (p<0.01) AND >> distant control",
    "time_symmetry": "foreshock-only AND aftershock-only stacks each beat null (p<0.01) and exceed the distant control",
    "matched_significant_p": 0.01,
    "band_hz": list(BAND), "win_s": WIN_S, "fs_hz": FS, "modulus": M, "kmin_events": KMIN,
    "snr_min": SNR_MIN, "n_null": N_NULL, "note": "second target; fore/after split by origin time; criteria fixed before run",
}


def qmod(n):
    return ((np.asarray(n, np.int64) - 1) % M) + 1


def prep(tr, origin):
    tr = tr.copy(); tr.detrend("demean"); tr.detrend("linear")
    tr.trim(origin, origin + WIN_S, pad=True, fill_value=0.0)
    tr.resample(FS); tr.detrend("demean"); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    n = int(WIN_S * FS); d = tr.data.astype(float)[:n]
    return np.pad(d, (0, n - d.size)) if d.size < n else d


def snr(tr, origin):
    tr = tr.copy(); tr.detrend("demean"); tr.detrend("linear")
    tr.trim(origin - 8.0, origin + 42.0, pad=True, fill_value=0.0)
    tr.resample(FS); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    d = tr.data.astype(float); ns = int(8.0 * FS)
    rn = np.sqrt(np.mean(d[:ns] * d[:ns])) + 1e-12
    return float(np.sqrt(np.mean(d[ns:] * d[ns:])) / rn)


def main():
    man = json.loads((DATA / "manifest.json").read_text())
    n = int(WIN_S * FS)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    oT = UTCDateTime(man["T"]["origin_time"])

    def load(tag):
        st = read(str(DATA / f"{tag}.mseed"))
        return {f"{tr.stats.network}.{tr.stats.station}": tr for tr in st}

    Tst = load(man["T"]["tag"])
    T_spec, T_ok = {}, {}
    for c, tr in Tst.items():
        T_ok[c] = snr(tr, oT) >= SNR_MIN
        T_spec[c] = np.fft.rfft(prep(tr, oT))[band]

    def event_phasors(ev):
        st = load(ev["tag"]); o = UTCDateTime(ev["origin_time"])
        phi = {}
        for c, tr in st.items():
            if c not in T_spec or not T_ok.get(c) or snr(tr, o) < SNR_MIN:
                continue
            R = np.fft.rfft(prep(tr, o))[band]
            phi[c] = np.angle(np.sum(T_spec[c] * np.conj(R)))
        if len(phi) < 3:
            return {}
        gk = np.angle(np.sum([np.exp(1j * p) for p in phi.values()]))
        return {c: np.exp(1j * (p - gk)) for c, p in phi.items()}

    def stack(events):
        acc, cnt = {}, {}
        for ev in events:
            for c, r in event_phasors(ev).items():
                acc[c] = acc.get(c, 0j) + r
                cnt[c] = cnt.get(c, 0) + 1
        return acc, cnt

    def coherence(phi, N, quantize=True):
        if quantize:
            q = qmod(np.rint(phi * M / (2.0 * np.pi)).astype(np.int64))
            phi = 2.0 * np.pi * (q - 1) / M
        return float(np.abs(np.sum(np.exp(1j * phi))) / N)

    def evaluate(events, kmin):
        acc, cnt = stack(events)
        codes = sorted(c for c in acc if cnt.get(c, 0) >= kmin)
        Nn = len(codes)
        if Nn < 5:
            return {"n_stations": Nn, "coherence": None, "scramble_p": None, "insufficient": True}
        phi = np.array([np.angle(acc[c]) for c in codes])
        C = coherence(phi, Nn)
        null = np.array([coherence(RNG.uniform(-np.pi, np.pi, size=Nn), Nn) for _ in range(N_NULL)])
        p = float((np.sum(null >= C) + 1) / (N_NULL + 1))
        return {"n_stations": Nn, "coherence": C,
                "coherence_full": coherence(phi, Nn, False), "scramble_p": p,
                "null_mean": float(null.mean())}

    fore = [e for e in man["matched"] if UTCDateTime(e["origin_time"]) < oT]
    after = [e for e in man["matched"] if UTCDateTime(e["origin_time"]) > oT]
    # count-balanced arms: equal event counts, largest-magnitude of each (matched data
    # quality) -> separates a pure count effect from a real time-asymmetry.
    kbal = min(len(fore), len(after))
    fore_bal = sorted(fore, key=lambda e: -e["magnitude"])[:kbal]
    after_bal = sorted(after, key=lambda e: -e["magnitude"])[:kbal]

    def mag_range(evs):
        ms = [e["magnitude"] for e in evs]
        return f"M{min(ms):.1f}-{max(ms):.1f}" if ms else "-"

    print(f"T2 = M{man['T']['magnitude']} at ({man['T']['latitude']:.3f},{man['T']['longitude']:.3f}); "
          f"matched {len(man['matched'])} = {len(fore)} foreshocks + {len(after)} aftershocks; "
          f"distant control {len(man['mismatched'])} events")
    print(f"count-balanced arms: {kbal} each -- foreshock {mag_range(fore_bal)}, "
          f"aftershock {mag_range(after_bal)}")
    print(f"PRE-REGISTERED: {json.dumps(PREREG)}\n")

    res = {
        "combined": evaluate(man["matched"], KMIN),
        "foreshock": evaluate(fore, KMIN),
        "aftershock": evaluate(after, KMIN),
        "foreshock_balanced": evaluate(fore_bal, KMIN),
        "aftershock_balanced": evaluate(after_bal, KMIN),
        "mismatched": evaluate(man["mismatched"], 1),
    }
    for name in ("combined", "foreshock", "aftershock", "foreshock_balanced",
                 "aftershock_balanced", "mismatched"):
        r = res[name]
        if r.get("insufficient"):
            print(f"[{name:>10}] insufficient stations ({r['n_stations']})")
        else:
            print(f"[{name:>10}] C={r['coherence']:.3f} (full {r['coherence_full']:.3f})  "
                  f"null p={r['scramble_p']:.4f}  ({r['n_stations']} stations)")

    mis_C = res["mismatched"].get("coherence") or 1.0

    def ok(r):
        return (not r.get("insufficient")) and r["scramble_p"] < PREREG["matched_significant_p"] and r["coherence"] > mis_C

    replication = ok(res["combined"])
    time_symmetry = ok(res["foreshock"]) and ok(res["aftershock"])
    time_symmetry_balanced = ok(res["foreshock_balanced"]) and ok(res["aftershock_balanced"])
    print(f"\nDECISION (pre-registered):")
    print(f"  (R) REPLICATION on second target (combined beats null & > control): {replication}")
    print(f"  (T) TIME-SYMMETRY, all events (fore & after each beat null & > control): {time_symmetry}")
    print(f"  (T-bal) DIAGNOSTIC (not part of the verdict), count-balanced {kbal} each "
          f"(isolates count vs real asymmetry): {time_symmetry_balanced}")
    # verdict follows the PRE-REGISTERED time_symmetry (all events); the balanced arm is
    # a post-hoc diagnostic reported alongside, never a verdict input.
    verdict = ("SUPPORTED" if (replication and time_symmetry) else
               "REPLICATED_ONLY" if replication else
               "PARTIAL" if (ok(res["foreshock"]) or ok(res["aftershock"])) else "NOT_SUPPORTED")
    print(f"  -> {verdict}")

    out = Path("results/seismic"); out.mkdir(parents=True, exist_ok=True)
    (out / "qa_seismic_egf_foreshock_results.json").write_text(json.dumps({
        "target": man["T"], "n_foreshocks": len(fore), "n_aftershocks": len(after),
        "n_balanced_each": kbal, "n_mismatched": len(man["mismatched"]), "pre_registration": PREREG,
        "results": res, "decision": {"replication": replication, "time_symmetry": time_symmetry,
                                      "time_symmetry_balanced": time_symmetry_balanced,
                                      "verdict": verdict}}, indent=2))
    print("\nsaved -> results/seismic/qa_seismic_egf_foreshock_results.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["foreshock", "aftershock", "combined", "mismatched"]
        vals = [res[k]["coherence"] or 0.0 for k in names]
        cols = ["#457b9d", "#2a9d8f", "#1d3557", "#e76f51"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar([f"{k}\n({res[k]['n_stations']} sta)" for k in names], vals, color=cols, edgecolor="k")
        if not res["combined"].get("insufficient"):
            ax.axhline(res["combined"]["null_mean"], color="gray", ls="--", lw=1, label="scramble null mean")
        ax.set_ylabel("QA mod-24 phase coherence"); ax.set_ylim(0, 1)
        ax.set_title(f"[522] EGF replication + foreshock control (T2 M{man['T']['magnitude']}) -> {verdict}")
        ax.legend(); fig.tight_layout(); fig.savefig("qa_seismic_egf_foreshock.png", dpi=110)
        print("saved -> qa_seismic_egf_foreshock.png")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
