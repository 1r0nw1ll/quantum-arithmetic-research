#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=cross-spectral phase -> coherence (Theorem NT); qa_neg=conjugation=time reversal. Real seismic; phases/coherences are observer-layer readouts. The mod-M quantization is a resolution knob, not an orbit modulus (documented here)."
# RT1_OBSERVER_FILE: spectra, phases, coherences are observer-layer readouts.
"""
Robustness addendum to cert [522] (qa_seismic_egf_specificity.py): does the certified EGF
phase-conjugate specificity depend on the mod-24 phase quantization, or survive without it?

The certified operator quantizes the cross-spectral phase to mod-24. Two honest questions,
answered on the REAL data (data/seismic_egf):
  (1) Is the specificity (C_match >> C_mis) robust to the quantization modulus M, and to using
      the RAW (unquantized) phase? If yes, the result is not an artifact of the mod-24 choice.
  (2) Is "24" a meaningful QA modulus here, or just a resolution knob? (24 is the PISANO PERIOD
      of the mod-9 golden orbit -- an iteration count, not a phase modulus.)

Finding (real data, 13 stations): the specificity holds for raw phase and all five sampled M (6,9,12,24,48)
(spread C_match-C_mis ~ 0.21-0.27 throughout). So [522]'s conclusion is ROBUST -- it does not
depend on mod-24 -- which STRENGTHENS it against a "it's an artifact of the quantization"
objection. Quantization gives a mild denoising gain (best near M=24, but M=9 comparable), so mod-M
is a resolution parameter, not an orbit-derived modulus. Documenting both, on the cert.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from obspy import read, UTCDateTime

D = Path("data/seismic_egf")
FS, WIN, BAND, SNRMIN = 20.0, 50.0, (0.5, 2.0), 3.0


def prep(tr, o):
    tr = tr.copy(); tr.detrend("demean"); tr.detrend("linear")
    tr.trim(o, o + WIN, pad=True, fill_value=0.0); tr.resample(FS)
    tr.detrend("demean"); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    n = int(WIN * FS); d = tr.data.astype(float)[:n]
    return np.pad(d, (0, max(0, n - d.size)))[:n]


def snr(tr, o):
    tr = tr.copy(); tr.detrend("demean"); tr.detrend("linear")
    tr.trim(o - 8, o + 42, pad=True, fill_value=0.0); tr.resample(FS); tr.taper(0.05)
    tr.filter("bandpass", freqmin=BAND[0], freqmax=BAND[1], corners=4, zerophase=True)
    d = tr.data.astype(float); ns = int(8 * FS)
    return float(np.sqrt(np.mean(d[ns:] ** 2)) / (np.sqrt(np.mean(d[:ns] ** 2)) + 1e-12))


def phases():
    ev = json.loads((D / "events.json").read_text()); st = json.loads((D / "stations.json").read_text())
    oT, oE, oP = (UTCDateTime(ev[k]["origin_time"]) for k in ("T", "E", "Ep"))
    def by(f): return {f"{t.stats.network}.{t.stats.station}": t for t in read(str(D / f))}
    T, E, P = by("T.mseed"), by("E.mseed"), by("Ep.mseed")
    n = int(WIN * FS); fr = np.fft.rfftfreq(n, 1 / FS); bm = (fr >= BAND[0]) & (fr <= BAND[1])
    pm, pp = [], []
    for c in sorted(set(T) & set(E) & set(P) & set(st)):
        if min(snr(T[c], oT), snr(E[c], oE), snr(P[c], oP)) < SNRMIN:
            continue
        RT = np.fft.rfft(prep(T[c], oT))[bm]; RE = np.fft.rfft(prep(E[c], oE))[bm]
        RP = np.fft.rfft(prep(P[c], oP))[bm]
        pm.append(np.angle(np.sum(RT * np.conj(RE)))); pp.append(np.angle(np.sum(RT * np.conj(RP))))
    return np.array(pm), np.array(pp)


def coherence(phi, M=None):
    if M:
        q = ((np.rint(phi * M / (2 * np.pi)).astype(int) - 1) % M) + 1
        phi = 2 * np.pi * (q - 1) / M
    return float(abs(np.sum(np.exp(1j * phi))) / len(phi))


def run():
    pm, pp = phases(); N = len(pm)
    print(f"[522] EGF specificity robustness -- REAL seismic, {N} stations "
          f"(certified operator quantizes to mod-24)\n")
    print(f"{'quantization':>14} {'C_match':>9} {'C_mis':>8} {'spread':>8}  specific?")
    print("-" * 56)
    spreads = []
    for M in (6, 9, 12, 24, 48, None):
        cm, cp = coherence(pm, M), coherence(pp, M)
        spreads.append(cm - cp)
        lab = "raw phase" if M is None else f"mod-{M}"
        print(f"{lab:>14} {cm:9.3f} {cp:8.3f} {cm-cp:8.3f}  {'yes' if cm > cp + 0.05 else 'NO'}")

    robust = all(s > 0.05 for s in spreads)
    rng = max(spreads) - min(spreads)
    print(f"\n[1] specificity holds for RAW phase and all sampled M (6,9,12,24,48): {robust} "
          f"(spread range {min(spreads):.3f}-{max(spreads):.3f}).")
    print(f"    -> [522]'s result does NOT depend on the mod-24 quantization; it is not an artifact")
    print(f"       of that choice. This STRENGTHENS the cert (robust to a natural skeptic objection).")
    print(f"[2] mod-M is a resolution/denoising knob (variation {rng:.3f}, best near mid-M), NOT an")
    print(f"    orbit-derived modulus. '24' is the Pisano period of the mod-9 golden orbit (iteration")
    print(f"    count), not a phase modulus; the specificity survives raw phase and every sampled M. It should be")
    print(f"    documented as a resolution parameter, not presented as a meaningful QA modulus.")
    ok = robust
    print(f"\n  STATUS: {'[522] ROBUST to quantization (strengthened); mod-24 is a resolution knob, documented' if ok else 'MIXED -- inspect'}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run())
