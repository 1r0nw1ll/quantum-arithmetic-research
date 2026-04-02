#!/usr/bin/env python3
"""
qa_resonance_waveform_test.py
==============================
Tests whether OFR resonance peaks at f = k*SR/m hold for non-sinusoidal
periodic signals: square, sawtooth, triangle.

Hypothesis:
  If OFR elevation is purely period-commensurability (arithmetic aliasing),
  it should appear equally for ANY periodic waveform at f = k*SR/m.

  If OFR elevation is sine-specific (harmonic structure of sine amplifies
  the recurrence), it will be weaker or absent for other waveforms.

Method:
  For each modulus in [5, 9, 12, 24]:
    Pick 4 resonant frequencies (f = k*SR/m) and 4 off-resonant frequencies.
    Generate sine, square, sawtooth, triangle at each frequency.
    Compute OFR_eq for each (freq, modulus, waveform) combo.
    Compare OFR at resonant vs off-resonant for each waveform type.

Output:
  - Console table + verdict per waveform type
  - qa_resonance_waveform.png
  - qa_resonance_waveform.json
"""

import numpy as np
import json
from pathlib import Path

SR       = 8000
DURATION = 2.0
N        = int(SR * DURATION)

MODULI_TEST = [5, 9, 12, 24]


# ── Waveform generators ──────────────────────────────────────────────────────

def gen_sine(freq):
    t = np.linspace(0, DURATION, N, endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def gen_square(freq, duty=0.5):
    t = np.linspace(0, DURATION, N, endpoint=False)
    phase = (t * freq) % 1.0
    return np.where(phase < duty, 1.0, -1.0).astype(float)

def gen_sawtooth(freq):
    t = np.linspace(0, DURATION, N, endpoint=False)
    phase = (t * freq) % 1.0
    return 2.0 * phase - 1.0

def gen_triangle(freq):
    t = np.linspace(0, DURATION, N, endpoint=False)
    phase = (t * freq) % 1.0
    return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0

WAVEFORMS = {
    "sine":     gen_sine,
    "square":   gen_square,
    "sawtooth": gen_sawtooth,
    "triangle": gen_triangle,
}


# ── Core OFR functions (equalized quantization) ──────────────────────────────

def quantize_eq(samples, m):
    n      = len(samples)
    ranks  = np.argsort(np.argsort(samples))
    states = (ranks * m // n).astype(int)
    return np.clip(states, 0, m - 1)

def compute_ofr(states, m):
    b   = states[:-2]
    e   = states[1:-1]
    nxt = states[2:]
    return float(np.sum(nxt == (b + e) % m)) / len(b)


# ── Select test frequencies ───────────────────────────────────────────────────

def resonant_freqs(m, sr, fmin=100, fmax=3900, n=4):
    """Pick n frequencies where f*m/sr is close to an integer."""
    candidates = []
    for k in range(1, 100):
        f = k * sr / m
        if fmin <= f <= fmax:
            candidates.append(int(round(f)))
    # deduplicate and take evenly spaced n
    candidates = sorted(set(candidates))
    if len(candidates) <= n:
        return candidates
    step = len(candidates) // n
    return [candidates[i * step] for i in range(n)]

def off_resonant_freqs(m, sr, res_set, fmin=100, fmax=3900, n=4):
    """Pick n frequencies maximally away from any resonant frequency."""
    # Midpoints between consecutive resonant frequencies
    res_sorted = sorted(res_set)
    candidates = []
    for i in range(len(res_sorted) - 1):
        mid = (res_sorted[i] + res_sorted[i+1]) // 2
        if fmin <= mid <= fmax:
            candidates.append(mid)
    # fill if needed
    if len(candidates) < n:
        for f in range(fmin, fmax, 100):
            ratio = f * m / sr
            near_int = min(abs(ratio - round(ratio)), abs(ratio - round(ratio)+1))
            if near_int > 0.3:
                candidates.append(f)
    # take n
    step = max(1, len(candidates) // n)
    selected = [candidates[i * step] for i in range(min(n, len(candidates)))]
    return selected[:n]


# ── Main sweep ────────────────────────────────────────────────────────────────

print("QA RESONANCE WAVEFORM TEST")
print("=" * 70)
print(f"  SR={SR}Hz  Duration={DURATION}s  N={N}")
print(f"  Waveforms: {list(WAVEFORMS.keys())}")
print(f"  Moduli: {MODULI_TEST}")
print()

results = {}

for m in MODULI_TEST:
    chance = 1.0 / m
    res_f   = resonant_freqs(m, SR)
    off_f   = off_resonant_freqs(m, SR, set(res_f))
    res_set = set(res_f)

    print(f"m={m}  chance={chance:.4f}")
    print(f"  Resonant   freqs: {[f'{f}Hz (k={f*m//SR})' for f in res_f]}")
    print(f"  Off-resont freqs: {[f'{f}Hz' for f in off_f]}")
    print()

    all_freqs = sorted(set(res_f + off_f))
    results[m] = {
        "chance": chance,
        "resonant_freqs": res_f,
        "off_resonant_freqs": off_f,
        "data": {}
    }

    print(f"  {'Freq':>6}  {'Type':>10}  " +
          "  ".join(f"{wn:>9}" for wn in WAVEFORMS) + "  {'is_res':>7}")
    print("  " + "-" * 74)

    for fi in all_freqs:
        is_res = fi in res_set
        row_data = {}
        cells = []
        for wn, wfunc in WAVEFORMS.items():
            sig    = wfunc(fi)
            states = quantize_eq(sig, m)
            ofr    = compute_ofr(states, m)
            exc    = ofr - chance
            row_data[wn] = {"ofr": ofr, "excess": exc}
            marker = "*" if exc > 0.03 else " "
            cells.append(f"{exc:>+8.4f}{marker}")
        results[m]["data"][fi] = {"is_resonant": is_res, "waveforms": row_data}
        tag = " ← RES" if is_res else ""
        print(f"  {fi:>6}Hz  {'':>10}  " + "  ".join(cells) + f"  {tag}")

    print()


# ── Verdict per waveform ──────────────────────────────────────────────────────

print("=" * 70)
print("VERDICT: Is OFR elevation waveform-independent?")
print("=" * 70)
print()
print("  Method: compute mean(excess at resonant) - mean(excess at off-resonant)")
print("  per (modulus, waveform) combo.")
print()

print(f"  {'Modulus':>8}  {'Waveform':>10}  {'Res mean':>9}  {'Off mean':>9}  {'Delta':>8}  {'Effect?':>10}")
print("  " + "-" * 68)

all_deltas = {wn: [] for wn in WAVEFORMS}

for m in MODULI_TEST:
    chance  = 1.0 / m
    m_data  = results[m]["data"]
    res_f   = results[m]["resonant_freqs"]
    off_f   = results[m]["off_resonant_freqs"]

    for wn in WAVEFORMS:
        res_exc = [m_data[fi]["waveforms"][wn]["excess"]
                   for fi in res_f if fi in m_data]
        off_exc = [m_data[fi]["waveforms"][wn]["excess"]
                   for fi in off_f if fi in m_data]
        if not res_exc or not off_exc:
            continue
        res_mean = float(np.mean(res_exc))
        off_mean = float(np.mean(off_exc))
        delta    = res_mean - off_mean
        effect   = "YES (+)" if delta > 0.02 else ("WEAK" if delta > 0 else "NO")
        all_deltas[wn].append(delta)
        print(f"  m={m:>5}  {wn:>10}  {res_mean:>+9.4f}  {off_mean:>+9.4f}  {delta:>+8.4f}  {effect:>10}")
    print()

print()
print("  AGGREGATE DELTA BY WAVEFORM (mean across all moduli):")
print()
for wn in WAVEFORMS:
    ds = all_deltas[wn]
    if ds:
        agg = float(np.mean(ds))
        print(f"    {wn:>10}: mean Δ = {agg:>+7.4f}  " +
              ("— RESONANCE GENERALIZES" if agg > 0.02 else
               "— WEAK" if agg > 0 else "— NO EFFECT"))

print()


# ── Ratio analysis ────────────────────────────────────────────────────────────

print("=" * 70)
print("RATIO: sine / other-waveform resonance strength")
print("=" * 70)
print()
print("  Ratio > 1 → sine uniquely strong (harmonic structure effect)")
print("  Ratio ≈ 1 → period-commensurability only (waveform-independent)")
print()

for m in MODULI_TEST:
    m_data = results[m]["data"]
    res_f  = results[m]["resonant_freqs"]
    print(f"  m={m}:")
    for fi in res_f:
        if fi not in m_data:
            continue
        sine_exc = m_data[fi]["waveforms"]["sine"]["excess"]
        for wn in ["square", "sawtooth", "triangle"]:
            other_exc = m_data[fi]["waveforms"][wn]["excess"]
            if abs(other_exc) > 1e-6:
                ratio = sine_exc / other_exc
                print(f"    {fi}Hz  sine_exc={sine_exc:>+7.4f}  {wn}_exc={other_exc:>+7.4f}  "
                      f"ratio={ratio:>6.2f}x")
            else:
                print(f"    {fi}Hz  sine_exc={sine_exc:>+7.4f}  {wn}_exc={other_exc:>+7.4f}  "
                      f"ratio=∞ (zero)")
    print()


# ── Save results ──────────────────────────────────────────────────────────────

out_data = {
    "experiment": "qa_resonance_waveform_test",
    "sr": SR, "duration": DURATION,
    "moduli": MODULI_TEST,
    "waveforms": list(WAVEFORMS.keys()),
    "results": {
        str(m): {
            "chance": results[m]["chance"],
            "resonant_freqs": results[m]["resonant_freqs"],
            "off_resonant_freqs": results[m]["off_resonant_freqs"],
            "data": {
                str(fi): results[m]["data"][fi]
                for fi in results[m]["data"]
            }
        }
        for m in MODULI_TEST
    }
}
Path("qa_resonance_waveform.json").write_text(json.dumps(out_data, indent=2))
print(f"  Data saved to qa_resonance_waveform.json")


# ── PNG ───────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(MODULI_TEST), 1, figsize=(14, 4 * len(MODULI_TEST)))
    if len(MODULI_TEST) == 1:
        axes = [axes]

    colors = {"sine": "steelblue", "square": "firebrick",
              "sawtooth": "darkorange", "triangle": "seagreen"}

    for ax, m in zip(axes, MODULI_TEST):
        chance = 1.0 / m
        m_data = results[m]["data"]
        freqs_sorted = sorted(m_data.keys())
        res_set = set(results[m]["resonant_freqs"])
        x = np.arange(len(freqs_sorted))
        width = 0.2

        for i, wn in enumerate(WAVEFORMS):
            excesses = [m_data[fi]["waveforms"][wn]["excess"] for fi in freqs_sorted]
            ax.bar(x + i * width, excesses, width=width, label=wn,
                   color=colors[wn], alpha=0.75)

        ax.axhline(0,    color="black", linewidth=0.8)
        ax.axhline(0.03, color="gray",  linewidth=0.8, linestyle="--", label="threshold +0.03")

        # mark resonant frequencies
        for j, fi in enumerate(freqs_sorted):
            if fi in res_set:
                ax.axvspan(j - 0.1, j + 0.9, alpha=0.08, color="gold")

        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([str(fi) for fi in freqs_sorted], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"m={m}  (chance={chance:.3f})  Gold bands = resonant f = k·{SR}/m")
        ax.set_ylabel("OFR excess above chance")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frequency (Hz)")
    plt.suptitle("QA Resonance: OFR excess by waveform type\n"
                 "Hypothesis: if arithmetic aliasing only → all waveforms show equal elevation at resonant freqs",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig("qa_resonance_waveform.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to qa_resonance_waveform.png")
except ImportError:
    print("  (matplotlib not available — skipping plot)")


print()
print("=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print("""
  Case A — All waveforms show equal OFR elevation at resonant freqs:
    → Effect is PURELY period-commensurability (arithmetic aliasing).
    → The resonance is waveform-independent: any periodic signal at
      f = k·SR/m will produce elevated OFR_eq.
    → This is the strongest possible claim for "quantized arithmetic resonance."

  Case B — Sine uniquely elevated, others flat:
    → Effect is SINE-SPECIFIC: the smooth zero-crossings or harmonic
      content of a sine wave interacts with mod-m quantization differently
      than other waveforms.
    → Still real and reproducible, but claim must be narrowed:
      "OFR detects sine-aliasing, not general period-commensurability."

  Case C — Some waveforms elevated, some not:
    → Effect depends on waveform bandwidth/harmonic content.
    → Richer claim: OFR is sensitive to spectral composition × modulus.
    → This would connect to the Fourier decomposition of the waveform
      and which harmonics fall on resonant frequencies.
""")
