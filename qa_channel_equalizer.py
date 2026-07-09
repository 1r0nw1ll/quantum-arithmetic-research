#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=symbol_to_phase, state_alphabet=mod24_A1_compliant"
"""
QA Phase-Conjugate Channel Equalizer (Tier-2 build; original signal_experiments domain).

Cert [518] proved exact distortion correction: a phase-conjugated wave returned
through the SAME distorting medium undoes the distortion. This applies that to a
communications channel — the project's original domain — in the FORWARD-equalization
form real receivers use:

  transmit symbols s  ->  distorting channel H (per-symbol modular phase screen)
  -> receive r = H(s)  ->  characterise H from known PILOTS  ->  equalize by the
  QA conjugate  ->  recover s.

This is NOT discrete-class recall, so the continuum-class failure mode of the
applicability boundary does not apply (see docs/theory/QA_PHASE_CONJUGATE_APPLICABILITY.md):
equalization is exact signal reconstruction, [518]'s home turf.

Same-medium specificity (the [518] fingerprint): an equalizer characterised on
channel H recovers symbols sent through H, and FAILS on a different channel H'.

A1/S2/Theorem-NT: symbol phase state integer in {1,...,m}; channel/noise are
observer-layer; the QA conjugate/equalize ops are integer.
"""
from __future__ import annotations
import numpy as np

M = 24


def qa_mod(x):
    return ((np.asarray(x, np.int64) - 1) % M) + 1


def qa_add(a, b):
    return qa_mod(np.asarray(a, np.int64) + np.asarray(b, np.int64))


def qa_neg(a):
    return qa_mod(-np.asarray(a, np.int64))


# ---------------------------------------------------------------------------
# Channel + equalizer
# ---------------------------------------------------------------------------
def make_channel(n, rng):
    """A per-symbol modular phase screen H (a frequency-selective distortion):
    each symbol position gets a fixed phase offset."""
    return rng.integers(1, M + 1, n)


def transmit(symbols, H):
    """r = H(s): apply the channel's per-symbol phase offsets."""
    return qa_add(symbols, H)


def estimate_channel(pilots, received_pilots):
    """Characterise H from known pilots: H_i = received_i - pilot_i (the offset
    that maps the known pilot to what was received)."""
    return qa_add(received_pilots, qa_neg(pilots))


def equalize(received, H_est):
    """QA conjugate equalization: subtract the estimated per-symbol channel phase.
    This is the discrete analog of returning the phase-conjugate through the SAME
    medium (cert [518]): qa_add(r, qa_neg(H_est)) = s when H_est == H."""
    return qa_add(received, qa_neg(H_est))


def add_noise(x, sigma, rng):
    """Observer-layer phase noise (integer jitter), then re-quantize to {1..m}."""
    if sigma <= 0:
        return x
    jitter = np.rint(rng.standard_normal(len(x)) * sigma).astype(np.int64)
    return qa_add(x, jitter)


def ser(a, b):
    """Symbol error rate."""
    return float(np.mean(a != b))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(42)
    N = 2000            # payload symbols
    N_PILOT = 400       # known pilot symbols to characterise the channel

    print(f"QA PHASE-CONJUGATE CHANNEL EQUALIZER  (m={M}, {N} symbols, {N_PILOT} pilots)\n")

    # [1] Core mechanism: recover the payload through a per-symbol distorting
    #     channel with KNOWN H (oracle) — the exact [518] conjugate equalization.
    print("[1] Symbol-error-rate: raw (no EQ) vs QA phase-conjugate equalized (known H)")
    print(f"{'noise sigma':>11s} {'SER raw':>9s} {'SER equalized':>14s}")
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
        raw_ser = eq_ser = 0.0
        trials = 40
        for _ in range(trials):
            H = make_channel(N, rng)                       # per-symbol phase screen
            payload = rng.integers(1, M + 1, N)
            rx = add_noise(transmit(payload, H), sigma, rng)
            eq = equalize(rx, H)                            # conjugate through same H
            raw_ser += ser(rx, payload)
            eq_ser += ser(eq, payload)
        print(f"{sigma:11.1f} {raw_ser/trials:9.3f} {eq_ser/trials:14.3f}")

    # [2] Same-medium specificity (the [518] fingerprint): an equalizer for channel
    #     H must FAIL on a different channel H'.
    print("\n[2] Channel-match specificity (SER after equalizing with H vs H'):")
    print(f"{'noise sigma':>11s} {'match H':>9s} {'mismatch Hp':>12s}")
    for sigma in (0.0, 1.0, 2.0):
        m_ser = mm_ser = 0.0
        trials = 40
        for _ in range(trials):
            H = make_channel(N, rng)
            Hp = make_channel(N, rng)               # different channel
            payload = rng.integers(1, M + 1, N)
            rx = add_noise(transmit(payload, H), sigma, rng)
            m_ser += ser(equalize(rx, H), payload)      # correct channel
            mm_ser += ser(equalize(rx, Hp), payload)    # wrong channel
        print(f"{sigma:11.1f} {m_ser/trials:9.3f} {mm_ser/trials:12.3f}")

    # [3] Pilot-based blind estimate quality: how well pilots recover a static
    #     channel, and resulting payload SER (realistic receiver).
    print("\n[3] Pilot-estimated equalization (static channel, estimate from pilots):")
    print(f"{'noise sigma':>11s} {'pilot-EQ SER':>13s} {'oracle-EQ SER':>14s}")
    for sigma in (0.0, 0.5, 1.0, 2.0):
        pilot_ser = oracle_ser = 0.0
        trials = 40
        for _ in range(trials):
            # static channel: ONE offset applied to every symbol (flat channel)
            offset = int(rng.integers(1, M + 1))
            payload = rng.integers(1, M + 1, N)
            pilots = rng.integers(1, M + 1, N_PILOT)
            rx_p = add_noise(qa_add(pilots, offset), sigma, rng)
            rx_d = add_noise(qa_add(payload, offset), sigma, rng)
            # estimate offset as the modal per-pilot estimate (majority vote)
            est = qa_add(rx_p, qa_neg(pilots))
            off_est = int(np.bincount(est).argmax())
            pilot_ser += ser(qa_add(rx_d, qa_neg(off_est)), payload)
            oracle_ser += ser(qa_add(rx_d, qa_neg(offset)), payload)
        print(f"{sigma:11.1f} {pilot_ser/trials:13.3f} {oracle_ser/trials:14.3f}")
    print("\nSame-medium specificity (mismatch SER ~ chance 1-1/m) is the [518] "
          "distortion-correction fingerprint in a comms channel.")


if __name__ == "__main__":
    run()
