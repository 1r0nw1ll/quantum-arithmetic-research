#!/usr/bin/env python3
"""
qa_audio_orbit_test.py

Tests the hypothesis:
    Authentic signals (from real dynamical systems) → stable QA orbit families
    Synthetic/noise signals (sampled distributions) → incoherent orbit trajectories

Method: map consecutive sample pairs to QA (b, e) states, track orbit family over time.

No external data needed — generates test signals internally.
Add --wav path1.wav path2.wav ... to test real files.
"""

QA_COMPLIANCE = "empirical_observer — audio signal is observer input; QA orbit is discrete state"


import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter

# ─── QA Arithmetic (mod 9, states {0..8}, where 0 = "9" in no-zero convention) ─

MODULUS = 9

def qa_next(b, e, m=MODULUS):
    """QA update: T(b,e) = (e, d) where d in {1,...,m} (A1: no-zero)"""
    return (e, ((b + e - 1) % m) + 1)

def orbit_length(b, e, m=MODULUS):
    """Trace orbit under T until revisit; return length."""
    seen = {}
    cur = (b, e)
    t = 0
    while cur not in seen:
        seen[cur] = t
        cur = qa_next(*cur, m)
        t += 1
    return t

def precompute_orbit_families(m=MODULUS):
    """
    Returns dict: (b,e) -> 'cosmos' | 'satellite' | 'singularity'
    Orbit lengths in mod-9:  1 = singularity, 8 = satellite, 24 = cosmos
    """
    families = {}
    for b in range(m):
        for e in range(m):
            length = orbit_length(b, e, m)
            if length == 1:
                families[(b, e)] = 'singularity'
            elif length <= 8:
                families[(b, e)] = 'satellite'
            else:
                families[(b, e)] = 'cosmos'
    return families

# ─── Signal Generation ─────────────────────────────────────────────────────────

SR = 8000      # sample rate (Hz)
DURATION = 2.0 # seconds

def gen_sine(freq=440.0, sr=SR, duration=DURATION):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def gen_chirp(f0=200.0, f1=2000.0, sr=SR, duration=DURATION):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2)
    return np.sin(phase)

def gen_am_sine(carrier=440.0, mod=5.0, depth=0.8, sr=SR, duration=DURATION):
    """Amplitude-modulated sine — structured but more complex dynamical system."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    envelope = 1.0 + depth * np.sin(2 * np.pi * mod * t)
    return envelope * np.sin(2 * np.pi * carrier * t)

def gen_white_noise(sr=SR, duration=DURATION, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration))

def gen_pink_noise(sr=SR, duration=DURATION, seed=42):
    """Pink (1/f) noise via spectral shaping — has structure but not a simple attractor."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]  # match DC to first harmonic — avoids massive DC amplification
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spectrum, n=n)
    pink -= np.mean(pink)  # remove residual DC offset
    return pink / (np.max(np.abs(pink)) + 1e-12)

def gen_harmonic_complex(sr=SR, duration=DURATION):
    """Sum of harmonics — real instrument-like, strong dynamical structure."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = sum(
        (1.0 / k) * np.sin(2 * np.pi * 220 * k * t + np.random.uniform(0, 0.1))
        for k in range(1, 8)
    )
    return sig / np.max(np.abs(sig) + 1e-12)

# ─── QA Analysis ──────────────────────────────────────────────────────────────

def quantize(samples, m=MODULUS):
    """Map float samples in [-1, 1] to integer states in {1..m} (A1: no-zero)."""
    clipped = np.clip(samples, -1.0, 1.0)
    states = (((clipped + 1.0) / 2.0) * m).astype(int)
    return np.clip(states, 1, m)  # A1: states in {1,...,m}

def equalize_quantize(samples, m=MODULUS):
    """
    Histogram-equalized quantization: force uniform distribution over {1..m}.
    Removes amplitude-distribution confound — bins are equally populated by rank.
    """
    n = len(samples)
    ranks = np.argsort(np.argsort(samples))  # stable rank 0..n-1
    states = (ranks * m // n).astype(int)
    return np.clip(states, 1, m)  # A1: states in {1,...,m}

def analyze_states(states, families, m=MODULUS):
    """
    Given a pre-quantized state sequence, compute QA orbit metrics.
    Call with raw-quantized or equalized-quantized states for comparison.
    """

    b_seq = states[:-1]
    e_seq = states[1:]

    family_seq = [families[(b, e)] for b, e in zip(b_seq, e_seq)]

    n = len(family_seq)
    counts = Counter(family_seq)
    cosmos_frac     = counts['cosmos'] / n
    satellite_frac  = counts['satellite'] / n
    singularity_frac = counts['singularity'] / n

    # Orbit family entropy (low = concentrated = coherent)
    probs = np.array([cosmos_frac, satellite_frac, singularity_frac])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))

    # Family switch rate — how often does the orbit family change?
    switches = sum(family_seq[i] != family_seq[i-1] for i in range(1, n))
    switch_rate = switches / (n - 1)

    # Dominant family persistence — mean run length in dominant family
    dominant = counts.most_common(1)[0][0]
    run_lengths = []
    run = 1
    for i in range(1, len(family_seq)):
        if family_seq[i] == family_seq[i-1]:
            run += 1
        else:
            run_lengths.append(run)
            run = 1
    run_lengths.append(run)
    mean_run = np.mean(run_lengths)
    dominant_run = np.mean([r for r, f in zip(run_lengths,
        [family_seq[0]] + [family_seq[i] for i in range(1, n) if family_seq[i] != family_seq[i-1]])
        if f == dominant] or [0])

    # QA norm f(b,e) = (b² + b*e - e²) mod m — invariant structure
    b_arr = b_seq.astype(float)
    e_arr = e_seq.astype(float)
    f_vals = ((b_arr**2 + b_arr * e_arr - e_arr**2) % m).astype(int)
    f_counts = Counter(f_vals.tolist())
    f_entropy = 0.0
    for cnt in f_counts.values():
        p = cnt / n
        if p > 0:
            f_entropy -= p * np.log2(p)

    # Orbit-following rate: given (b,e) at t, does (b,e) at t+1 == T(b,e)?
    # T(b,e) = (e, (b+e)%m)
    # Chance level: 1/m = 11.1% (random next state)
    # Authentic dynamical signal hypothesis: orbit_follow_rate >> 1/m
    follow_count = 0
    for i in range(len(b_seq) - 1):
        b, e = b_seq[i], e_seq[i]
        b_next, e_next = b_seq[i+1], e_seq[i+1]
        t_b, t_e = e, ((b + e - 1) % m) + 1  # T(b, e) — A1: no-zero
        if b_next == t_b and e_next == t_e:
            follow_count += 1
    orbit_follow_rate = follow_count / (len(b_seq) - 1)

    return {
        'n_samples': n,
        'cosmos_frac':      round(cosmos_frac, 4),
        'satellite_frac':   round(satellite_frac, 4),
        'singularity_frac': round(singularity_frac, 4),
        'orbit_entropy':    round(entropy, 4),
        'switch_rate':      round(switch_rate, 4),
        'mean_run_length':  round(mean_run, 2),
        'f_entropy':        round(f_entropy, 4),
        'orbit_follow_rate': round(orbit_follow_rate, 4),  # KEY: > 1/m = above-chance QA orbit adherence
        'dominant_family':  dominant,
    }

def analyze_signal(samples, families, m=MODULUS):
    """Analyze a signal under both raw and equalized quantization."""
    raw_states = quantize(samples, m)
    eq_states  = equalize_quantize(samples, m)
    raw = analyze_states(raw_states, families, m)
    eq  = analyze_states(eq_states,  families, m)
    return raw, eq

# ─── WAV loading ──────────────────────────────────────────────────────────────

def load_wav(path):
    """Load a WAV file, return mono float samples in [-1, 1]."""
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(float) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(float) / 2**31
        elif data.dtype != np.float32 and data.dtype != np.float64:
            data = data.astype(float) / np.iinfo(data.dtype).max
        return data.astype(float)
    except ImportError:
        raise ImportError("scipy required for WAV loading: pip install scipy")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='QA orbit analysis for audio signals')
    parser.add_argument('--wav', nargs='*', help='WAV files to analyze')
    parser.add_argument('--modulus', type=int, default=9, help='QA modulus (default: 9)')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    m = args.modulus
    print(f"Precomputing orbit families (mod {m})...")
    families = precompute_orbit_families(m)

    # Show orbit family inventory
    inv = Counter(families.values())
    print(f"  State space: {m}×{m}={m*m} states")
    print(f"  Cosmos: {inv['cosmos']}  Satellite: {inv['satellite']}  Singularity: {inv['singularity']}")
    print()

    # Build test signals
    test_signals = [
        ('sine_440Hz',      gen_sine(440),        'dynamical — simple harmonic oscillator'),
        ('sine_880Hz',      gen_sine(880),        'dynamical — harmonic oscillator (2×)'),
        ('chirp_200-2kHz',  gen_chirp(),          'dynamical — time-varying oscillator'),
        ('harmonic_complex',gen_harmonic_complex(),'dynamical — sum of harmonics (instrument-like)'),
        ('am_sine',         gen_am_sine(),        'dynamical — AM modulated (coupled oscillators)'),
        ('pink_noise_1f',   gen_pink_noise(),     'stochastic — 1/f shaped, some structure'),
        ('white_noise',     gen_white_noise(),    'stochastic — IID, no dynamical structure'),
    ]

    if args.wav:
        for path in args.wav:
            p = Path(path)
            try:
                sig = load_wav(path)
                test_signals.append((p.stem, sig, f'from file: {path}'))
                print(f"Loaded {path}: {len(sig)} samples")
            except Exception as ex:
                print(f"Warning: could not load {path}: {ex}")

    # Analyze
    results = []
    for name, sig, description in test_signals:
        mx = np.max(np.abs(sig))
        if mx > 0:
            sig = sig / mx
        raw, eq = analyze_signal(sig, families, m)
        results.append({
            'name': name,
            'description': description,
            'raw': raw,
            'eq':  eq,
        })

    if args.json:
        print(json.dumps(results, indent=2))
        return

    chance = 1.0 / m

    # ── Table ──────────────────────────────────────────────────────────────────
    hdr = f"{'Signal':<22}  {'RAW_follow':>10}  {'EQ_follow':>10}  {'delta_raw':>9}  {'delta_eq':>9}  {'EQ_entropy':>10}  {'EQ_switch':>9}"
    print(hdr)
    print(f"  (chance = {chance:.4f})")
    print("-" * len(hdr))

    for r in results:
        rf  = r['raw']['orbit_follow_rate']
        ef  = r['eq']['orbit_follow_rate']
        dr  = rf - chance
        de  = ef - chance
        ee  = r['eq']['orbit_entropy']
        es  = r['eq']['switch_rate']
        ra  = '↑' if dr > 0 else '↓'
        ea  = '↑' if de > 0 else '↓'
        print(
            f"{r['name']:<22}  "
            f"{rf:>9.4f}{ra}  "
            f"{ef:>9.4f}{ea}  "
            f"{dr:>+9.4f}  "
            f"{de:>+9.4f}  "
            f"{ee:>10.4f}  "
            f"{es:>9.4f}  "
            f"{r['description']}"
        )

    print()
    print(f"Cosmos base rate = {72/(m*m)*100:.1f}%  |  orbit_follow chance = 1/{m} = {chance:.4f}")
    print()

    # ── Summary by group ───────────────────────────────────────────────────────
    groups = [
        ('dynamical', [r for r in results if 'dynamical' in r['description']]),
        ('stochastic', [r for r in results if 'stochastic' in r['description']]),
    ]
    print(f"{'Group':<12}  {'mean RAW':>9}  {'mean EQ':>9}  {'Δ RAW-chance':>13}  {'Δ EQ-chance':>12}")
    print("-" * 65)
    for gname, grp in groups:
        if not grp:
            continue
        mr  = np.mean([r['raw']['orbit_follow_rate'] for r in grp])
        me  = np.mean([r['eq']['orbit_follow_rate']  for r in grp])
        print(f"{gname:<12}  {mr:>9.4f}  {me:>9.4f}  {mr-chance:>+13.4f}  {me-chance:>+12.4f}")

    print()
    print("INTERPRETATION:")
    dyn_grp = [r for r in results if 'dynamical' in r['description']]
    sto_grp = [r for r in results if 'stochastic' in r['description']]
    if dyn_grp and sto_grp:
        eq_dyn = np.mean([r['eq']['orbit_follow_rate'] for r in dyn_grp])
        eq_sto = np.mean([r['eq']['orbit_follow_rate'] for r in sto_grp])
        gap = eq_dyn - eq_sto
        if abs(gap) < 0.005:
            verdict = "EQ gap negligible — effect was mostly amplitude distribution. No structural QA signal."
        elif gap > 0:
            verdict = f"EQ gap = {gap:+.4f} — gap survives equalization. Temporal structure present beyond amplitude distribution."
        else:
            verdict = f"EQ gap = {gap:+.4f} — stochastic > dynamical after equalization. Unexpected."
        print(f"  {verdict}")

if __name__ == '__main__':
    main()
