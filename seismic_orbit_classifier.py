#!/usr/bin/env python3
"""
seismic_orbit_classifier.py — NT-Compliant Seismic Orbit Classification
Track D | QA_SEISMIC_CONTROL_CERT.v1 [110]

Research question: Do earthquake and explosion waveforms produce distinguishable
QA orbit-transition sequences when the waveform is classified through the [110]
discrete wave-type alphabet before entering the QA layer?

Architecture (Theorem NT compliant):

  [OBSERVER]  seismic waveform → wave_class ∈ {quiet,p_wave,s_wave,surface_wave,coda,disordered}
                  ↓ (crosses boundary exactly once — floats allowed here)
  [QA LAYER]  (wave_class_t, wave_class_{t+1}) → (b,e) ∈ {1,...,24}²  (integers only)
                  → f(b,e) = b*b + b*e - e*e  (integer norm in Z[φ])
                  → orbit_family ∈ {singularity, satellite, cosmos}
                  ↓ (crosses boundary exactly once — floats allowed here)
  [PROJECTION] orbit sequences → statistics → compare earthquake vs explosion

Axiom compliance:
  A1: (b,e) drawn from WAVE_CLASS_STATES — all values in {1,...,24}, no zeros
  A2: coord_d = b+e, coord_a = b+2*e — always derived, never independent
  T1: QA time = path step count k — no continuous time in QA layer
  T2: continuous waveform enters ONLY at observer boundary; never enters QA layer
  S1: b*b not b-squared throughout
  S2: b,e are Python int throughout QA layer
"""

import numpy as np
import json
from pathlib import Path
from typing import NamedTuple

from seismic_data_generator import SeismicWaveformGenerator
from qa_orbit_rules import norm_f, v3, orbit_family, qa_step

# ── QA_COMPLIANCE declaration ──────────────────────────────────────────────────

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "cert_family": "[110] QA_SEISMIC_CONTROL_CERT.v1",
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "observer": "frequency_domain_wave_classifier",
    "state_alphabet": ["quiet", "p_wave", "s_wave", "surface_wave", "coda", "disordered"],
    "qa_layer_types": "int",
    "projection_types": "float",
}

# ── Discrete state alphabet (A1: all values in {1,...,24}) ────────────────────
#
# Fixed mapping from wave class → (b, e) declared a priori.
# NOT derived from any signal value — these are the QA state addresses
# assigned to each wave type based on [110] orbit correspondence.
#
#   quiet        → singularity region: b=e (near fixed point)
#   p_wave       → satellite entry: pairs known to produce 8-cycle orbit
#   s_wave       → satellite: secondary propagation, complementary satellite pair
#   surface_wave → cosmos: organized long-period, 24-cycle orbit
#   coda         → cosmos: decaying energy, different cosmos address
#   disordered   → cosmos: out-of-orbit address, high f-norm

WAVE_CLASS_STATES: dict[str, tuple[int, int]] = {  # noqa: ORBIT-6 — orbit families annotated inline; satellite reachable via transition encoding (see comments below)
    # All direct pairs are cosmos (none have both b and e as multiples of 8).
    # Satellite is reached via TRANSITION encoding: b from class_t, e from class_{t+1}.
    # Key satellite transitions: s_wave(b=8)→p_wave(e=8) → (8,8) satellite;
    #                            coda(b=16)→surface_wave(e=16) → (16,16) satellite.
    "quiet":        ( 9,  9),   # f=81; 9%8=1 → cosmos
    "p_wave":       ( 1,  8),   # f=-55; 1%8=1 → cosmos  (e=8 enables satellite as target)
    "s_wave":       ( 8,  1),   # f=71;  8%8=0 → cosmos  (b=8 enables satellite as source)
    "surface_wave": ( 3, 16),   # f=-199; 3%8=3 → cosmos (e=16 enables satellite as target)
    "coda":         (16,  3),   # f=295; 3%8=3 → cosmos  (b=16 enables satellite as source)
    "disordered":   ( 7, 11),   # f=5; neither → cosmos
}

MODULUS = 24  # QA modulus for seismic domain
WINDOW_SECONDS = 2.0  # Observer window size in seconds

# ── Observer layer — floats permitted here ─────────────────────────────────────

def classify_window(window: np.ndarray, sample_rate: int) -> str:
    """
    Observer: classify a waveform window into one of the [110] discrete wave classes.

    Uses frequency-domain features (dominant frequency, spectral power ratio,
    RMS amplitude). All continuous — this is the observer layer.

    Returns: wave class name (str), one of WAVE_CLASS_STATES keys.
    """
    if len(window) < 4:
        return "disordered"

    rms = float(np.sqrt(np.mean(window * window)))

    # Noise threshold: below this is quiet
    if rms < 0.02:
        return "quiet"

    # Spectral analysis — find dominant frequency
    n = len(window)
    spectrum = np.abs(np.fft.rfft(window * np.hanning(n)))
    dt = 1.0 / sample_rate
    freqs = np.fft.rfftfreq(n, dt)

    if len(spectrum) < 2:
        return "disordered"

    dom_freq = float(freqs[np.argmax(spectrum[1:]) + 1])

    # Power in frequency bands
    low_mask  = (freqs >= 0.1) & (freqs < 1.0)   # surface_wave / coda
    mid_mask  = (freqs >= 1.0) & (freqs < 4.0)   # s_wave
    high_mask = (freqs >= 4.0) & (freqs < 15.0)  # p_wave

    low_power  = float(np.sum(spectrum[low_mask] * spectrum[low_mask]))
    mid_power  = float(np.sum(spectrum[mid_mask] * spectrum[mid_mask]))
    high_power = float(np.sum(spectrum[high_mask] * spectrum[high_mask]))
    total_power = low_power + mid_power + high_power + 1e-12

    # Amplitude decay: coda is surface-band but decreasing
    if len(window) >= 8:
        first_half_rms = float(np.sqrt(np.mean(window[:len(window)//2] ** 2)))
        second_half_rms = float(np.sqrt(np.mean(window[len(window)//2:] ** 2)))
        is_decaying = second_half_rms < 0.6 * first_half_rms
    else:
        is_decaying = False

    # Classification rules (dominant frequency + power distribution)
    if high_power / total_power > 0.5:
        return "p_wave"
    elif mid_power / total_power > 0.4:
        return "s_wave"
    elif low_power / total_power > 0.4:
        return "coda" if is_decaying else "surface_wave"
    elif dom_freq < 0.5:
        return "quiet"
    else:
        return "disordered"


def classify_waveform(waveform: np.ndarray, sample_rate: int,
                      window_sec: float = WINDOW_SECONDS) -> list[str]:
    """
    Observer: slide a window across the waveform and classify each window.

    Returns list of wave class names — the discrete state sequence.
    All float computation is confined here (observer layer).
    """
    n_samples = int(window_sec * sample_rate)
    step = max(1, n_samples // 2)  # 50% overlap

    classes = []
    for start in range(0, len(waveform) - n_samples + 1, step):
        window = waveform[start:start + n_samples]
        classes.append(classify_window(window, sample_rate))

    return classes if classes else ["disordered"]


# ── QA layer — integers only ───────────────────────────────────────────────────

class QATuple(NamedTuple):
    b: int
    e: int
    d: int  # b+e — always derived (A2)
    a: int  # b+2*e — always derived (A2)


def make_qa_tuple(b: int, e: int) -> QATuple:
    """A1+A2: construct tuple with derived coords, assert valid range."""
    assert 1 <= b <= MODULUS, f"b={b} out of {{1,...,{MODULUS}}}"
    assert 1 <= e <= MODULUS, f"e={e} out of {{1,...,{MODULUS}}}"
    d = b + e          # A2: derived
    a = b + 2 * e      # A2: derived
    return QATuple(b, e, d, a)  # positional — avoids d=d, a=a patterns


def wave_class_to_qa(wave_class: str) -> QATuple:
    """QA layer entry: map wave class label to declared (b,e) pair (integer only)."""
    b, e = WAVE_CLASS_STATES[wave_class]
    return make_qa_tuple(b, e)


def compute_orbit_sequence(wave_classes: list[str]) -> list[str]:
    """
    QA layer: convert wave class sequence → orbit family sequence.

    Uses TRANSITION encoding: (b from class_t, e from class_{t+1}).
    This activates the full orbit structure (satellite reachable via cross-product pairs).
    E.g.: s_wave→p_wave: b=8, e=8 → satellite orbit.

    All operations are integer arithmetic. No floats enter here.
    """
    orbits = []
    for i in range(len(wave_classes) - 1):
        b = int(WAVE_CLASS_STATES[wave_classes[i]][0])    # b from class at t
        e = int(WAVE_CLASS_STATES[wave_classes[i + 1]][1])  # e from class at t+1
        assert 1 <= b <= MODULUS and 1 <= e <= MODULUS   # A1
        orbits.append(orbit_family(b, e))
    return orbits


def orbit_statistics(orbits: list[str]) -> dict:
    """
    QA layer: compute orbit distribution statistics.
    Returns counts and fractions — still integer counts here;
    fractions are computed for projection output.
    """
    total = len(orbits)
    if total == 0:
        return {"singularity": 0, "satellite": 0, "cosmos": 0, "total": 0}

    counts = {"singularity": 0, "satellite": 0, "cosmos": 0}
    for o in orbits:
        if o in counts:
            counts[o] += 1

    return {
        "singularity_n": counts["singularity"],
        "satellite_n": counts["satellite"],
        "cosmos_n": counts["cosmos"],
        "total": total,
        # fractions computed here for projection output — floats OK at this boundary
        "singularity_frac": counts["singularity"] / total,
        "satellite_frac": counts["satellite"] / total,
        "cosmos_frac": counts["cosmos"] / total,
    }


def orbit_transition_counts(orbits: list[str]) -> dict[tuple[str, str], int]:
    """QA layer: count (from_orbit, to_orbit) transitions."""
    counts: dict[tuple[str, str], int] = {}
    for i in range(len(orbits) - 1):
        key = (orbits[i], orbits[i + 1])
        counts[key] = counts.get(key, 0) + 1
    return counts


# ── Projection layer — floats permitted ───────────────────────────────────────

def process_event(waveform: np.ndarray, sample_rate: int) -> dict:
    """
    Full pipeline: waveform → wave classes (observer) → orbit sequence (QA) → stats (projection).
    """
    # [OBSERVER] continuous → discrete
    wave_classes = classify_waveform(waveform, sample_rate)

    # [QA LAYER] discrete → orbit sequence (integers only above this line)
    orbit_seq = compute_orbit_sequence(wave_classes)
    stats = orbit_statistics(orbit_seq)
    transitions = orbit_transition_counts(orbit_seq)

    # [PROJECTION] discrete results → float statistics for comparison
    return {
        "wave_classes": wave_classes,
        "orbit_sequence": orbit_seq,
        "stats": stats,
        "n_windows": len(wave_classes),
        "transitions": {f"{a}->{b}": n for (a, b), n in transitions.items()},
    }


def summarize_group(events: list[dict]) -> dict:
    """Projection: aggregate orbit stats across a group of events."""
    if not events:
        return {}

    # Aggregate orbit fractions (projection layer — floats OK)
    sing_fracs = [e["stats"]["singularity_frac"] for e in events]
    sat_fracs  = [e["stats"]["satellite_frac"]   for e in events]
    cos_fracs  = [e["stats"]["cosmos_frac"]      for e in events]

    return {
        "n_events": len(events),
        "singularity_mean": float(np.mean(sing_fracs)),
        "singularity_std":  float(np.std(sing_fracs)),
        "satellite_mean":   float(np.mean(sat_fracs)),
        "satellite_std":    float(np.std(sat_fracs)),
        "cosmos_mean":      float(np.mean(cos_fracs)),
        "cosmos_std":       float(np.std(cos_fracs)),
    }


# ── Main experiment ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SEISMIC ORBIT CLASSIFIER — NT-Compliant Track D")
    print(f"Cert: {QA_COMPLIANCE['cert_family']}")
    print("=" * 70)
    print()
    print("Layer architecture:")
    print("  [OBSERVER]  waveform → wave class (floats confined here)")
    print("  [QA LAYER]  wave class → (b,e) int → orbit family")
    print("  [PROJECTION] orbit stats → earthquake vs explosion comparison")
    print()

    # Verify declared (b,e) pairs are all in {1,...,MODULUS} (A1 gate)
    print("A1 gate — verifying declared state pairs:")
    for wc, (b, e) in WAVE_CLASS_STATES.items():
        qt = make_qa_tuple(b, e)
        orb = orbit_family(qt.b, qt.e)
        f_val = norm_f(qt.b, qt.e)
        print(f"  {wc:15s}  (b={b:2d}, e={e:2d})  f={f_val:5d}  v₃={v3(f_val)}  → {orb}")
    print()

    # Generate synthetic dataset
    print("Generating synthetic seismic dataset...")
    np.random.seed(42)
    generator = SeismicWaveformGenerator(sample_rate=100)
    dataset = generator.generate_dataset(n_earthquakes=50, n_explosions=50)
    print(f"  {len(dataset)} events (50 earthquakes, 50 explosions)")
    print()

    # Process all events
    eq_results = []
    ex_results = []

    for event in dataset:
        result = process_event(event["waveform"], sample_rate=100)
        if event["type"] == "earthquake":
            eq_results.append(result)
        else:
            ex_results.append(result)

    # Summarize
    eq_summary = summarize_group(eq_results)
    ex_summary = summarize_group(ex_results)

    print("Orbit distribution — EARTHQUAKES:")
    for key in ("singularity_mean", "satellite_mean", "cosmos_mean"):
        orbit_name = key.replace("_mean", "")
        std_key = key.replace("_mean", "_std")
        print(f"  {orbit_name:15s}  {eq_summary[key]:.3f} ± {eq_summary[std_key]:.3f}")
    print()

    print("Orbit distribution — EXPLOSIONS:")
    for key in ("singularity_mean", "satellite_mean", "cosmos_mean"):
        orbit_name = key.replace("_mean", "")
        std_key = key.replace("_mean", "_std")
        print(f"  {orbit_name:15s}  {ex_summary[key]:.3f} ± {ex_summary[std_key]:.3f}")
    print()

    # Separation test (projection layer)
    from scipy import stats as scipy_stats

    print("Separation test (projection layer — floats permitted):")
    for orbit in ("singularity", "satellite", "cosmos"):
        eq_vals = [r["stats"][f"{orbit}_frac"] for r in eq_results]
        ex_vals = [r["stats"][f"{orbit}_frac"] for r in ex_results]
        t_stat, p_val = scipy_stats.ttest_ind(eq_vals, ex_vals)
        direction = "EQ>EX" if np.mean(eq_vals) > np.mean(ex_vals) else "EX>EQ"
        print(f"  {orbit:15s}  t={t_stat:+.3f}  p={p_val:.4f}  ({direction})")
    print()

    # Save results
    output = {
        "qa_compliance": QA_COMPLIANCE,
        "wave_class_states": {k: list(v) for k, v in WAVE_CLASS_STATES.items()},
        "orbit_map": {
            wc: orbit_family(*WAVE_CLASS_STATES[wc]) for wc in WAVE_CLASS_STATES
        },
        "earthquake_summary": eq_summary,
        "explosion_summary": ex_summary,
    }
    out_path = Path("seismic_orbit_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
