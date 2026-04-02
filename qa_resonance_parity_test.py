#!/usr/bin/env python3
"""
qa_resonance_parity_test.py
============================
Targeted test of the alternating elevation/suppression pattern in OFR at
resonant frequencies f = k*SR/m.

Questions:
  1. Parity law: does OFR excess sign depend on k mod 2?
  2. Phase: does a π/2 phase shift flip the pattern?
  3. Window alignment: is it a finite-N artifact?
  4. Near-resonant: how sharp is the resonance (bandwidth)?
  5. Modulus dependence: does the parity rule vary with m?

Signals used: sine only (triangle is equivalent by rank-equivalence theorem).
Moduli: 5, 9, 12, 24.

Output:
  - Console table for each question
  - qa_resonance_parity.png
  - qa_resonance_parity.json
"""

import numpy as np
import json
from pathlib import Path

SR       = 8000
DURATION = 2.0
N        = int(SR * DURATION)


# ── Core ──────────────────────────────────────────────────────────────────────

def gen_sine(freq, phase=0.0, sr=SR, duration=DURATION):
    n = int(sr * duration)
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t + phase)

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


# ── Q1: Parity sweep — k from 1..20 for each modulus ─────────────────────────

print("QA RESONANCE PARITY TEST")
print("=" * 70)
print()

MODULI = [5, 9, 12, 24]
K_MAX  = 20

q1_results = {}

print("Q1: OFR EXCESS VS k  (f = k*SR/m, sine, N=16000)")
print("=" * 70)

for m in MODULI:
    chance = 1.0 / m
    print(f"\n  m={m}  chance={chance:.4f}  f_step={SR//m}Hz")
    print(f"  {'k':>4}  {'f':>6}  {'OFR':>7}  {'excess':>8}  {'sign':>5}  {'k%2':>4}  {'k%m':>4}")
    print("  " + "-" * 52)

    row_data = []
    signs_odd  = []
    signs_even = []

    for k in range(1, K_MAX + 1):
        f_exact = k * SR / m
        if f_exact > 4000:
            break
        f_int = int(round(f_exact))
        sig   = gen_sine(f_exact)          # use exact (float) freq
        st    = quantize_eq(sig, m)
        ofr   = compute_ofr(st, m)
        exc   = ofr - chance
        sign  = "+" if exc >= 0 else "-"
        km2   = k % 2
        kmm   = k % m

        row_data.append({
            "k": k, "f": round(f_exact, 2), "ofr": round(ofr, 6),
            "excess": round(exc, 6), "k_mod2": km2, "k_modm": kmm
        })

        marker = " ***" if abs(exc) > 0.03 else ("  * " if abs(exc) > 0.01 else "    ")
        print(f"  {k:>4}  {f_exact:>6.1f}  {ofr:>7.4f}  {exc:>+8.4f}  {sign:>5}  {km2:>4}  {kmm:>4}{marker}")

        if km2 == 1:
            signs_odd.append(exc)
        else:
            signs_even.append(exc)

    q1_results[m] = row_data

    mean_odd  = float(np.mean(signs_odd))  if signs_odd  else 0
    mean_even = float(np.mean(signs_even)) if signs_even else 0
    print(f"\n  Mean excess: k odd={mean_odd:>+8.5f}  k even={mean_even:>+8.5f}")
    parity_rule = "k odd → +, k even → −" if mean_odd > 0 and mean_even < 0 else \
                  "k even → +, k odd → −" if mean_even > 0 and mean_odd < 0 else \
                  "NO CLEAR PARITY"
    print(f"  Parity rule: {parity_rule}")


# ── Q2: Phase shift ───────────────────────────────────────────────────────────

print()
print("=" * 70)
print("Q2: PHASE OFFSET — does phase flip the OFR sign?")
print("=" * 70)

PHASES = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
phase_labels = ["0", "π/4", "π/2", "3π/4", "π"]

for m in [5, 9]:
    chance = 1.0 / m
    print(f"\n  m={m}:")
    # Test k=1 (expect +) and k=2 (expect −) based on Q1 findings
    for k in [1, 2, 3, 4]:
        f_exact = k * SR / m
        if f_exact > 4000:
            continue
        row = f"  k={k} f={f_exact:.0f}Hz  "
        for ph, phl in zip(PHASES, phase_labels):
            sig = gen_sine(f_exact, phase=ph)
            st  = quantize_eq(sig, m)
            ofr = compute_ofr(st, m)
            exc = ofr - chance
            row += f"φ={phl}: {exc:>+6.4f}  "
        print(row)


# ── Q3: Window alignment — vary duration ──────────────────────────────────────

print()
print("=" * 70)
print("Q3: WINDOW ALIGNMENT — is parity a finite-N artifact?")
print("=" * 70)
print("  Test: vary N so that exact integer number of cycles fits")
print()

m      = 9
chance = 1.0 / m

for k in [1, 2, 3]:
    f_exact    = k * SR / m
    period_s   = 1.0 / f_exact
    print(f"  k={k} f={f_exact:.2f}Hz  period={period_s*1000:.3f}ms")

    for n_cycles in [5, 10, 20, 50, 100, 200]:
        dur     = n_cycles * period_s
        n_samp  = int(round(dur * SR))
        if n_samp < 20:
            continue
        t       = np.arange(n_samp) / SR
        sig     = np.sin(2 * np.pi * f_exact * t)
        st      = quantize_eq(sig, m)
        ofr     = compute_ofr(st, m)
        exc     = ofr - chance
        print(f"    {n_cycles:>4} cycles  N={n_samp:>6}  OFR={ofr:.5f}  excess={exc:>+8.5f}")
    print()


# ── Q4: Resonance bandwidth — how sharp? ──────────────────────────────────────

print("=" * 70)
print("Q4: RESONANCE BANDWIDTH — OFR vs frequency offset from f=k*SR/m")
print("=" * 70)

BANDWIDTH_MODULI = [9]

for m in BANDWIDTH_MODULI:
    chance = 1.0 / m
    for k in [1, 2, 3]:
        f0      = k * SR / m
        if f0 > 4000:
            continue
        print(f"\n  m={m} k={k} f0={f0:.2f}Hz:")
        print(f"  {'offset':>8}  {'f':>8}  {'OFR':>7}  {'excess':>8}")
        offsets = [0, 1, 2, 5, 10, 20, 50, 100]
        for off in offsets:
            f_test = f0 + off
            sig    = gen_sine(f_test)
            st     = quantize_eq(sig, m)
            ofr    = compute_ofr(st, m)
            exc    = ofr - chance
            print(f"  {off:>+8}Hz  {f_test:>8.1f}  {ofr:>7.4f}  {exc:>+8.4f}")


# ── Q5: Parity rule formula ───────────────────────────────────────────────────

print()
print("=" * 70)
print("Q5: PARITY RULE — candidate formula derivation")
print("=" * 70)
print()
print("  For a sine at f = k*SR/m, the phase advances by 2πk/m per sample.")
print("  After m samples, the phase has advanced by 2πk (k full turns).")
print()
print("  Rank sequence period = m / gcd(k, m) samples")
print()

for m in MODULI:
    chance = 1.0 / m
    print(f"  m={m}:")
    for k in range(1, K_MAX + 1):
        f_exact = k * SR / m
        if f_exact > 4000:
            break
        g          = int(np.gcd(k, m))
        rank_period = m // g
        phase_step = 2 * np.pi * k / m
        # Predict: OFR sign from rank sequence symmetry
        # For k odd with unimodal (sine): positive excess?
        row_data = next((r for r in q1_results[m] if r["k"] == k), None)
        exc = row_data["excess"] if row_data else 0
        print(f"    k={k:>2}  gcd(k,m)={g:>2}  rank_period={rank_period:>3}  "
              f"phase_step={phase_step/np.pi:.3f}π  excess={exc:>+7.4f}")
    print()


# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

for m in MODULI:
    chance = 1.0 / m
    data   = q1_results[m]
    odd_k  = [r for r in data if r["k_mod2"] == 1]
    even_k = [r for r in data if r["k_mod2"] == 0]
    if odd_k and even_k:
        m_odd  = float(np.mean([r["excess"] for r in odd_k]))
        m_even = float(np.mean([r["excess"] for r in even_k]))
        odds_pos  = sum(1 for r in odd_k  if r["excess"] > 0)
        evens_neg = sum(1 for r in even_k if r["excess"] < 0)
        print(f"  m={m:>2}:  mean_k_odd={m_odd:>+7.5f}  mean_k_even={m_even:>+7.5f}  "
              f"  k_odd→+ : {odds_pos}/{len(odd_k)}  k_even→− : {evens_neg}/{len(even_k)}")


# ── Save ──────────────────────────────────────────────────────────────────────

out_data = {
    "experiment": "qa_resonance_parity_test",
    "sr": SR, "duration": DURATION, "moduli": MODULI, "k_max": K_MAX,
    "q1_parity_sweep": {str(m): q1_results[m] for m in MODULI},
}
Path("qa_resonance_parity.json").write_text(json.dumps(out_data, indent=2))
print(f"\n  Data saved to qa_resonance_parity.json")


# ── PNG ───────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, m in zip(axes, MODULI):
        chance = 1.0 / m
        data   = q1_results[m]
        ks     = [r["k"]      for r in data]
        excs   = [r["excess"] for r in data]
        k_mod2 = [r["k_mod2"] for r in data]

        colors = ["steelblue" if km2 == 1 else "firebrick" for km2 in k_mod2]
        ax.bar(ks, excs, color=colors, alpha=0.8, width=0.7)
        ax.axhline(0,     color="black", linewidth=0.8)
        ax.axhline(+0.03, color="gray",  linewidth=0.6, linestyle="--")
        ax.axhline(-0.03, color="gray",  linewidth=0.6, linestyle="--")

        ax.set_xlabel("k  (f = k·SR/m)")
        ax.set_ylabel("OFR excess above chance")
        ax.set_title(f"m={m}  (blue=k odd, red=k even)")
        ax.set_xticks(ks)

    plt.suptitle("OFR Resonance Parity: does k-parity predict OFR sign?\n"
                 "Blue = k odd, Red = k even",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("qa_resonance_parity.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to qa_resonance_parity.png")
except ImportError:
    print("  (matplotlib not available)")
