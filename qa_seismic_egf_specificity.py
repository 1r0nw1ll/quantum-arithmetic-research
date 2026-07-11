#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=cross_spectral_phase_to_QA mod24; qa_neg=conjugation=time reversal; coherence is observer readout (Theorem NT)"
# RT1_OBSERVER_FILE: seismic cross-spectral phases are observer-layer projections, not QA state.
"""
Cert [522] EGF specificity test on REAL data -- identify the medium empirically.

The real-data failure of qa_seismic_tr_real.py came from re-emitting through a
GUESSED medium (constant velocity). Fink's mirror never needs to know the medium --
it re-emits through the SAME real medium. Here we do that computationally with an
EMPIRICAL GREEN'S FUNCTION: a CO-LOCATED companion event E records the true medium
G_i from the target T's source patch (R_i^E = G_i * Src_E).

Observable (the [522] operator on real records): the QA mod-24 quantized phase of the
per-station cross-spectrum
    CS_i = sum_{f in band} R_i^T(f) * conj(R_i^EGF(f)).
For the MATCHED (co-located) EGF the shared medium cancels (R^T conj(R^E) ~ |G_i|^2 *
Src_T Src_E*, |G_i|^2 real) -> CS_i phase is ~station-independent -> the unit-phasor
stack is COHERENT. For a MISMATCHED (distant) EGF the path differs (G_i^T != G_i^Ep)
-> a station-dependent residual phase -> the stack DECOHERES. qa_neg (conjugation) is
the time reversal.

  coherence C = | (1/N) sum_i exp(2*pi*i*(Q_i - 1)/24) |,  Q_i in {1..24}.

CLAIM (cert [518]/[522] same-medium fingerprint, on real earthquakes):
  MATCHED (co-located EGF) coherence >> MISMATCHED (distant EGF) coherence,
  and MATCHED beats a phase-scramble null while MISMATCHED does not.

Theorem NT: cross-spectral phase + band choice are observer inputs (crossed once,
in); the QA phase is an integer mod 24; the coherence is an observer readout (out).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from obspy import read, UTCDateTime

DATA = Path("data/seismic_egf")
M = 24
FS = 20.0                 # Hz common resample rate
WIN_S = 50.0              # seconds after origin
BAND = (0.5, 2.0)         # Hz cross-spectral band
N_NULL = 5000
SNR_MIN = 3.0             # keep a station only if all three events clear this SNR
RNG = np.random.default_rng(522)

PREREG = {
    "primary": "C_matched > C_mismatched (co-located EGF focuses, distant EGF does not)",
    "matched_significant_p": 0.01,          # C_matched beats scramble null at p<0.01
    "mismatched_null_min_p": 0.05,          # C_mismatched consistent with null (p>=0.05)
    "band_hz": list(BAND), "win_s": WIN_S, "fs_hz": FS, "modulus": M, "n_null": N_NULL,
    "note": "criteria fixed in source before running; EGF = co-located companion event",
}


def qmod(n):
    return ((np.asarray(n, np.int64) - 1) % M) + 1


def prep(tr, origin):
    tr = tr.copy()
    tr.detrend("demean"); tr.detrend("linear")
    tr.trim(origin, origin + WIN_S, pad=True, fill_value=0.0)
    tr.resample(FS)
    tr.detrend("demean"); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    n = int(WIN_S * FS)
    d = tr.data.astype(float)[:n]
    if d.size < n:
        d = np.pad(d, (0, n - d.size))
    return d


def spectrum(d):
    return np.fft.rfft(d)


def snr(tr, origin):
    """Band-limited SNR = rms(post-origin signal) / rms(pre-origin noise)."""
    tr = tr.copy(); tr.detrend("demean"); tr.detrend("linear")
    tr.trim(origin - 8.0, origin + 42.0, pad=True, fill_value=0.0)
    tr.resample(FS); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    d = tr.data.astype(float)
    ns = int(8.0 * FS)
    noise = d[:ns]; sig = d[ns:]
    rn = np.sqrt(np.mean(noise * noise)) + 1e-12
    return float(np.sqrt(np.mean(sig * sig)) / rn)


def main():
    ev = json.loads((DATA / "events.json").read_text())
    stations = json.loads((DATA / "stations.json").read_text())
    oT = UTCDateTime(ev["T"]["origin_time"])
    oE = UTCDateTime(ev["E"]["origin_time"])
    oP = UTCDateTime(ev["Ep"]["origin_time"])

    def by_code(fname):
        st = read(str(DATA / fname))
        return {f"{tr.stats.network}.{tr.stats.station}": tr for tr in st}

    T, E, P = by_code("T.mseed"), by_code("E.mseed"), by_code("Ep.mseed")
    n = int(WIN_S * FS)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])

    phi_match, phi_mis, codes = [], [], []
    n_snr_drop = 0
    for code in sorted(set(T) & set(E) & set(P) & set(stations)):
        if min(snr(T[code], oT), snr(E[code], oE), snr(P[code], oP)) < SNR_MIN:
            n_snr_drop += 1
            continue
        RT = spectrum(prep(T[code], oT))[band]
        RE = spectrum(prep(E[code], oE))[band]
        RP = spectrum(prep(P[code], oP))[band]
        cs_m = np.sum(RT * np.conj(RE))          # matched: co-located EGF
        cs_p = np.sum(RT * np.conj(RP))          # mismatched: distant EGF
        phi_match.append(np.angle(cs_m)); phi_mis.append(np.angle(cs_p)); codes.append(code)
    phi_match = np.array(phi_match); phi_mis = np.array(phi_mis)
    N = len(codes)
    if N < 5:
        raise SystemExit(f"only {N} stations clear SNR>={SNR_MIN}; too few for the test")

    def coherence(phi, quantize=True):
        if quantize:
            q = qmod(np.rint(phi * M / (2.0 * np.pi)).astype(np.int64))
            phi = 2.0 * np.pi * (q - 1) / M
        return float(np.abs(np.sum(np.exp(1j * phi))) / N)

    C_match = coherence(phi_match)
    C_mis = coherence(phi_mis)
    # full-precision (no mod-24) diagnostic: is the signal there before quantization?
    C_match_full = coherence(phi_match, quantize=False)
    C_mis_full = coherence(phi_mis, quantize=False)
    print(f"(SNR>={SNR_MIN}: kept {N} stations, dropped {n_snr_drop})")
    print(f"(full-precision coherence: matched {C_match_full:.3f}, mismatched {C_mis_full:.3f} "
          f"-> mod-24 vs full tells whether quantization is the limiter)")

    # phase-scramble null: N random phases -> coherence (single number, no grid search)
    null = np.array([coherence(RNG.uniform(-np.pi, np.pi, size=N)) for _ in range(N_NULL)])
    p_match = float((np.sum(null >= C_match) + 1) / (N_NULL + 1))
    p_mis = float((np.sum(null >= C_mis) + 1) / (N_NULL + 1))

    print(f"EGF specificity test -- real M{ev['T']['magnitude']} target, "
          f"{N} common stations, band {BAND[0]}-{BAND[1]} Hz")
    print(f"  EGF matched (co-located, {ev['hypo_km_E_to_T']:.1f} km): "
          f"E M{ev['E']['magnitude']}")
    print(f"  EGF mismatched (distant, {ev['hypo_km_Ep_to_T']:.1f} km): "
          f"Ep M{ev['Ep']['magnitude']}")
    print(f"PRE-REGISTERED: {json.dumps(PREREG)}\n")
    print(f"[MATCHED]    co-located-EGF coherence C={C_match:.3f}  (scramble-null p={p_match:.4f})")
    print(f"[MISMATCHED] distant-EGF    coherence C={C_mis:.3f}  (scramble-null p={p_mis:.4f})")
    print(f"  null coherence: {null.mean():.3f}+/-{null.std():.3f} (95th pct {np.percentile(null,95):.3f})")

    specificity = bool(C_match > C_mis)
    matched_sig = bool(p_match < PREREG["matched_significant_p"])
    mismatched_null = bool(p_mis >= PREREG["mismatched_null_min_p"])
    verdict = ("SUPPORTED" if (specificity and matched_sig and mismatched_null)
               else "PARTIAL" if (specificity and matched_sig)
               else "NOT_SUPPORTED")
    print(f"\nDECISION (pre-registered): specificity C_match>C_mis={specificity}, "
          f"matched beats null (p<{PREREG['matched_significant_p']})={matched_sig}, "
          f"mismatched ~ null (p>={PREREG['mismatched_null_min_p']})={mismatched_null} -> {verdict}")

    results = {
        "events": {k: ev[k] for k in ("T", "E", "Ep")},
        "hypo_km_E_to_T": ev["hypo_km_E_to_T"], "hypo_km_Ep_to_T": ev["hypo_km_Ep_to_T"],
        "n_stations": N, "stations": codes, "pre_registration": PREREG,
        "matched": {"coherence": C_match, "coherence_full_precision": C_match_full, "scramble_p": p_match},
        "mismatched": {"coherence": C_mis, "coherence_full_precision": C_mis_full, "scramble_p": p_mis},
        "n_snr_dropped": n_snr_drop,
        "null": {"mean": float(null.mean()), "std": float(null.std()),
                 "pct95": float(np.percentile(null, 95))},
        "decision": {"specificity": specificity, "matched_significant": matched_sig,
                     "mismatched_is_null": mismatched_null, "verdict": verdict},
    }
    out = Path("results/seismic"); out.mkdir(parents=True, exist_ok=True)
    (out / "qa_seismic_egf_specificity_results.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> results/seismic/qa_seismic_egf_specificity_results.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(["matched\n(co-located EGF)", "mismatched\n(distant EGF)"], [C_match, C_mis],
               color=["#2a9d8f", "#e76f51"], edgecolor="k")
        ax.axhspan(0, np.percentile(null, 95), color="gray", alpha=0.25,
                   label="scramble null (<=95th pct)")
        ax.axhline(np.percentile(null, 95), color="gray", ls="--", lw=1)
        ax.set_ylabel("QA mod-24 phase coherence"); ax.set_ylim(0, 1)
        ax.set_title(f"[522] EGF same-medium specificity on real data\n"
                     f"M{ev['T']['magnitude']} target, {N} stations -> {verdict}")
        ax.legend()
        fig.tight_layout(); fig.savefig("qa_seismic_egf_specificity.png", dpi=110)
        print("saved -> qa_seismic_egf_specificity.png")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
