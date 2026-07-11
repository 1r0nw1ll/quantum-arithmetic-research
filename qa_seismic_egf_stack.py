#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=stacked_cross_spectral_phase_to_QA mod24; qa_neg=conjugation=time reversal; coherence is observer readout (Theorem NT)"
# RT1_OBSERVER_FILE: seismic cross-spectral phases are observer-layer projections, not QA state.
"""
Cert [522] STACKED-EGF specificity test on REAL data -- significance push.

The single-EGF test (qa_seismic_egf_specificity.py) recovered the same-medium
specificity DIRECTION (co-located 0.331 vs distant 0.067) but the matched coherence
did not clear individual significance: the one co-located EGF still sat 3.2 km from T,
leaving a station-dependent residual phase.

Fix: STACK a CLUSTER of co-located aftershocks. Each aftershock E_k has offset phase
~2*pi*f*(dist(T,i)-dist(E_k,i))/v at station i; for aftershocks scattered around T the
offset vectors point different ways, so averaging exp(i*offset) over k converges to a
station-INDEPENDENT residual (~0) -> the stacked matched phase re-coheres, while SNR
improves. Per event we remove the station-independent global phase (source/origin
term) before stacking. The distant cluster (wrong patch) stays decoherent.

  Phi_i^k   = angle( sum_band R_i^T * conj(R_i^{E_k}) )
  r_i^k     = exp( i*(Phi_i^k - global_k) ),  global_k = circular mean over stations
  stack_i   = sum_k r_i^k  (matched: co-located cluster; mismatched: distant cluster)
  C         = | (1/N) sum_i exp(2*pi*i*(Q(angle stack_i)-1)/24) |,  Q in {1..24}.
qa_neg (conjugation) is the time reversal; mod-24 is the QA observer projection.

Pre-registered, honest either way. Theorem NT: phases + band are observer inputs; the
QA phase is an integer mod 24; the coherence is an observer readout.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from obspy import read, UTCDateTime

DATA = Path("data/seismic_egf_stack")
M = 24
FS = 20.0
WIN_S = 50.0
BAND = (0.5, 2.0)
SNR_MIN = 2.0
KMIN = 3                 # a station must have >= this many contributing matched events
N_NULL = 5000
RNG = np.random.default_rng(522)

PREREG = {
    "primary": "C_matched > C_mismatched (co-located cluster focuses, distant cluster does not)",
    "matched_significant_p": 0.01,
    "mismatched_null_min_p": 0.05,
    "band_hz": list(BAND), "win_s": WIN_S, "fs_hz": FS, "modulus": M, "kmin_events": KMIN,
    "n_null": N_NULL, "note": "stacked co-located EGF cluster; criteria fixed before run",
}


def qmod(n):
    return ((np.asarray(n, np.int64) - 1) % M) + 1


def prep(tr, origin):
    tr = tr.copy(); tr.detrend("demean"); tr.detrend("linear")
    tr.trim(origin, origin + WIN_S, pad=True, fill_value=0.0)
    tr.resample(FS); tr.detrend("demean"); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    n = int(WIN_S * FS)
    d = tr.data.astype(float)[:n]
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

    def load(tag):
        st = read(str(DATA / f"{tag}.mseed"))
        return {f"{tr.stats.network}.{tr.stats.station}": tr for tr in st}

    Tst = load(man["T"]["tag"]); oT = UTCDateTime(man["T"]["origin_time"])
    # cache T spectra + SNR per station
    T_spec, T_ok = {}, {}
    for c, tr in Tst.items():
        T_ok[c] = snr(tr, oT) >= SNR_MIN
        T_spec[c] = np.fft.rfft(prep(tr, oT))[band]

    def event_phasors(ev):
        """Referenced unit phasor r_i (global phase removed) per station for one EGF."""
        st = load(ev["tag"]); o = UTCDateTime(ev["origin_time"])
        phi = {}
        for c, tr in st.items():
            if c not in T_spec or not T_ok.get(c) or snr(tr, o) < SNR_MIN:
                continue
            R = np.fft.rfft(prep(tr, o))[band]
            cs = np.sum(T_spec[c] * np.conj(R))
            phi[c] = np.angle(cs)
        if len(phi) < 3:
            return {}
        gk = np.angle(np.sum([np.exp(1j * p) for p in phi.values()]))   # circular-mean global phase
        return {c: np.exp(1j * (p - gk)) for c, p in phi.items()}

    def stack(events):
        acc, cnt = {}, {}
        for ev in events:
            for c, r in event_phasors(ev).items():
                acc[c] = acc.get(c, 0j) + r
                cnt[c] = cnt.get(c, 0) + 1
        return acc, cnt

    acc_m, cnt_m = stack(man["matched"])
    acc_x, cnt_x = stack(man["mismatched"])

    # Evaluate each on ITS OWN station set (significance needs many stations -> a low
    # null; the single distant control must not throttle the matched station count).
    def coherence(phi, N, quantize=True):
        if quantize:
            q = qmod(np.rint(phi * M / (2.0 * np.pi)).astype(np.int64))
            phi = 2.0 * np.pi * (q - 1) / M
        return float(np.abs(np.sum(np.exp(1j * phi))) / N)

    def evaluate(acc, cnt, kmin):
        codes = sorted(c for c in acc if cnt.get(c, 0) >= kmin)
        phi = np.array([np.angle(acc[c]) for c in codes])
        n = len(codes)
        C = coherence(phi, n); Cf = coherence(phi, n, False)
        null = np.array([coherence(RNG.uniform(-np.pi, np.pi, size=n), n) for _ in range(N_NULL)])
        p = float((np.sum(null >= C) + 1) / (N_NULL + 1))
        return codes, C, Cf, p, null

    codes_m, C_match, C_match_full, p_match, null = evaluate(acc_m, cnt_m, KMIN)
    codes_x, C_mis, C_mis_full, p_mis, null_x = evaluate(acc_x, cnt_x, 1)
    N = len(codes_m)
    if N < 8:
        raise SystemExit(f"only {N} matched stations (KMIN={KMIN}); lower KMIN or SNR")

    print(f"STACKED-EGF specificity -- target M{man['T']['magnitude']}, "
          f"{len(man['matched'])} matched + {len(man['mismatched'])} mismatched events; "
          f"matched on {N} stations (>= {KMIN} events), mismatched on {len(codes_x)}, "
          f"band {BAND[0]}-{BAND[1]} Hz")
    print(f"PRE-REGISTERED: {json.dumps(PREREG)}\n")
    print(f"[MATCHED]    co-located cluster  C={C_match:.3f} (full {C_match_full:.3f})  null p={p_match:.4f}")
    print(f"[MISMATCHED] distant cluster     C={C_mis:.3f} (full {C_mis_full:.3f})  null p={p_mis:.4f}")
    print(f"  null: {null.mean():.3f}+/-{null.std():.3f} (95th pct {np.percentile(null,95):.3f}, "
          f"99th {np.percentile(null,99):.3f})")

    specificity = bool(C_match > C_mis)
    matched_sig = bool(p_match < PREREG["matched_significant_p"])
    mismatched_null = bool(p_mis >= PREREG["mismatched_null_min_p"])
    verdict = ("SUPPORTED" if (specificity and matched_sig and mismatched_null)
               else "PARTIAL" if (specificity and matched_sig) else "NOT_SUPPORTED")
    print(f"\nDECISION (pre-registered): specificity={specificity}, matched_significant "
          f"(p<{PREREG['matched_significant_p']})={matched_sig}, mismatched~null={mismatched_null} "
          f"-> {verdict}")

    results = {
        "target": man["T"], "n_matched_events": len(man["matched"]),
        "n_mismatched_events": len(man["mismatched"]),
        "n_matched_stations": N, "n_mismatched_stations": len(codes_x),
        "matched_stations": codes_m, "pre_registration": PREREG,
        "matched": {"coherence": C_match, "coherence_full": C_match_full, "scramble_p": p_match},
        "mismatched": {"coherence": C_mis, "coherence_full": C_mis_full, "scramble_p": p_mis},
        "null": {"mean": float(null.mean()), "std": float(null.std()),
                 "pct95": float(np.percentile(null, 95)), "pct99": float(np.percentile(null, 99))},
        "decision": {"specificity": specificity, "matched_significant": matched_sig,
                     "mismatched_is_null": mismatched_null, "verdict": verdict},
    }
    out = Path("results/seismic"); out.mkdir(parents=True, exist_ok=True)
    (out / "qa_seismic_egf_stack_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> results/seismic/qa_seismic_egf_stack_results.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(["matched\n(co-located cluster)", "mismatched\n(distant cluster)"], [C_match, C_mis],
               color=["#2a9d8f", "#e76f51"], edgecolor="k")
        ax.axhspan(0, np.percentile(null, 99), color="gray", alpha=0.25, label="scramble null (<=99th pct)")
        ax.axhline(np.percentile(null, 99), color="gray", ls="--", lw=1)
        ax.set_ylabel("QA mod-24 phase coherence"); ax.set_ylim(0, 1)
        ax.set_title(f"[522] stacked-EGF same-medium specificity\nM{man['T']['magnitude']} target, "
                     f"{len(man['matched'])}-event stack, {N} stations -> {verdict}")
        ax.legend()
        fig.tight_layout(); fig.savefig("qa_seismic_egf_stack.png", dpi=110)
        print("saved -> qa_seismic_egf_stack.png")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
