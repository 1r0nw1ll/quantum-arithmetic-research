#!/usr/bin/env python3
# RT1_OBSERVER_FILE: continuous-signal analysis (Lorenz/Rossler/sine/noise, permutation entropy, Lyapunov); trig/sqrt act on observer-layer signals, never QA state.
"""
Syntropic Projector Probe — candidate cert family [518].

Thesis (falsifiable): the QA orbit-fold acts as a *syntropic projector*.
A chaotic continuous source generates information at rate h_KS = sum of
positive Lyapunov exponents (Pesin). QA dynamics live on a finite state
space {1..9}^2 and are eventually periodic, so h_KS(QA layer) = 0. The
question is whether the fold CONCENTRATES a deterministic chaotic trajectory
onto the 5 T-orbit families MORE than a size-matched random partition of the
same lattice does. That "excess collapse" is the QA-specific, falsifiable claim.

Controls that make this able to FAIL:
  - size-matched random partitions of the occupied lattice cells (null dist)
  - white-noise source (should scatter -> ~zero excess collapse)
  - periodic source (should concentrate hard)

Boundary crossings (Theorem NT): continuous source -> quantize to (b,e) lattice
[crossing 1]. All downstream is integer (T2 respected). Orbit-family symbol
stream is the output [crossing 2]. No float re-enters QA as causal input.

A1/S1 compliant: qa_mod in {1..m}, b*b never b**2.
"""
from __future__ import annotations

QA_COMPLIANCE = "observer=continuous_signal_to_phase_bins, state_alphabet=mod{9,24}_A1; negative-result companion to cert [518], see docs/theory/QA_SYNTROPY_PHASE_CONJUGATE_INVESTIGATION.md"
import numpy as np
from collections import Counter
from math import log2, factorial

RNG = np.random.default_rng(42)
M = 9

# ---------------------------------------------------------------------------
# Canonical QA arithmetic (copied verbatim from qa_svp_validation_harness.py)
# ---------------------------------------------------------------------------
def qa_mod(x: int, m: int = M) -> int:
    return ((int(x) - 1) % m) + 1

def qa_step(b: int, e: int, m: int = M):
    return e, ((b + e - 1) % m) + 1

def orbit(b: int, e: int, m: int = M):
    b, e = qa_mod(b, m), qa_mod(e, m)
    states, seen, cur = [], {}, (b, e)
    while cur not in seen:
        seen[cur] = len(states)
        states.append(cur)
        cur = qa_step(cur[0], cur[1], m)
    return states[seen[cur]:]

_MAXLEN = max(len(orbit(b, e)) for b in range(1, M + 1) for e in range(1, M + 1))

def classify_state(b: int, e: int, m: int = M) -> str:
    L = len(orbit(b, e, m))
    if L == 1:
        return "singularity"
    if L == _MAXLEN:
        return "cosmos"
    return "satellite"

def eisenstein_norm(b: int, e: int, m: int = M) -> int:
    return (b * b + b * e - e * e) % m  # S1: b*b, not b**2

_NORM_PAIR = {frozenset({1, 8}): "fibonacci",
              frozenset({4, 5}): "lucas",
              frozenset({2, 7}): "phibonacci"}

def orbit_family_s9(b: int, e: int) -> str:
    coarse = classify_state(b, e)
    if coarse == "singularity":
        return "ninbonacci"
    if coarse == "satellite":
        return "tribonacci"
    norm = eisenstein_norm(b, e)
    for pair, fam in _NORM_PAIR.items():
        if norm in pair:
            return fam
    raise ValueError(f"cosmos ({b},{e}) norm {norm} unmapped")

# Precompute the fold: every lattice cell -> family symbol
FAMILIES = ["fibonacci", "lucas", "phibonacci", "tribonacci", "ninbonacci"]
FAM_IDX = {f: i for i, f in enumerate(FAMILIES)}
CELL_FAMILY = {(b, e): orbit_family_s9(b, e) for b in range(1, M + 1) for e in range(1, M + 1)}

# ---------------------------------------------------------------------------
# Continuous sources (observer projection layer)
# ---------------------------------------------------------------------------
def lorenz(n=60000, dt=0.01, s=10.0, r=28.0, bta=8.0 / 3.0, transient=5000):
    xyz = np.array([1.0, 1.0, 1.0])
    out = np.empty((n, 3))
    def f(v):
        x, y, z = v
        return np.array([s * (y - x), x * (r - z) - y, x * y - bta * z])
    for _ in range(transient):
        k1 = f(xyz); k2 = f(xyz + 0.5 * dt * k1)
        k3 = f(xyz + 0.5 * dt * k2); k4 = f(xyz + dt * k3)
        xyz = xyz + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for i in range(n):
        k1 = f(xyz); k2 = f(xyz + 0.5 * dt * k1)
        k3 = f(xyz + 0.5 * dt * k2); k4 = f(xyz + dt * k3)
        xyz = xyz + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = xyz
    return out[:, 0]  # observe x-coordinate only

def lorenz_lyapunov(n=40000, dt=0.01, s=10.0, r=28.0, bta=8.0/3.0, d0=1e-8):
    """Benettin largest-Lyapunov estimate (nats per unit time)."""
    def f(v):
        x, y, z = v
        return np.array([s*(y-x), x*(r-z)-y, x*y-bta*z])
    def rk4(v):
        k1=f(v); k2=f(v+0.5*dt*k1); k3=f(v+0.5*dt*k2); k4=f(v+dt*k3)
        return v + dt/6*(k1+2*k2+2*k3+k4)
    xa = np.array([1.0,1.0,1.0])
    for _ in range(5000): xa = rk4(xa)
    xb = xa + np.array([d0,0,0]); acc = 0.0
    for _ in range(n):
        xa = rk4(xa); xb = rk4(xb)
        d = np.linalg.norm(xb-xa); acc += np.log(d/d0)
        xb = xa + (xb-xa)*(d0/d)
    return acc / (n*dt)

def white_noise(n=60000):
    return RNG.standard_normal(n)

def periodic(n=60000):
    t = np.arange(n) * 0.05
    return np.sin(t) + 0.5*np.sin(2.3*t)  # quasi-periodic, low-complexity

# ---------------------------------------------------------------------------
# Firewall quantization: continuous series -> (b,e) lattice via delay embed
# ---------------------------------------------------------------------------
def to_bins(x, m=M, mode="equal_occ"):
    """Quantize to {1..m} (A1 No-Zero).
    equal_occ: rank bins -> uniform marginal, isolates DYNAMICAL structure.
    fixed_w:   equal-width bins -> keeps attractor amplitude density."""
    if mode == "equal_occ":
        ranks = np.argsort(np.argsort(x))
        return (ranks * m // len(x)) + 1
    lo, hi = x.min(), x.max()
    idx = np.clip(((x - lo) / (hi - lo + 1e-12) * m).astype(int), 0, m - 1)
    return idx + 1

def embed_to_states(x, tau=8, mode="equal_occ"):
    b = to_bins(x, mode=mode)
    e = np.roll(b, -tau)
    n = len(x) - tau
    return list(zip(b[:n].tolist(), e[:n].tolist()))

# ---------------------------------------------------------------------------
# Entropy measures
# ---------------------------------------------------------------------------
def shannon(counts):
    tot = sum(counts)
    if tot == 0: return 0.0
    return -sum((c/tot)*log2(c/tot) for c in counts if c > 0)

def perm_entropy(x, d=4, tau=1):
    """Bandt-Pompe permutation entropy (normalized, bits)."""
    from itertools import permutations
    patt = Counter()
    for i in range(len(x) - (d-1)*tau):
        window = x[i:i+(d-1)*tau+1:tau]
        patt[tuple(np.argsort(window))] += 1
    H = shannon(list(patt.values()))
    return H / log2(factorial(d))

def family_entropy(states):
    c = Counter(CELL_FAMILY[s] for s in states)
    return shannon([c.get(f, 0) for f in FAMILIES]), c

def null_partition_entropy(states, n_trials=3000):
    """Size-matched random partition of OCCUPIED cells into classes with the
    same sizes as the QA family partition. Returns null H distribution."""
    cell_counts = Counter(states)
    occupied = list(cell_counts.keys())
    # QA class sizes measured in number of *occupied cells* per family
    qa_class_cells = Counter(CELL_FAMILY[c] for c in occupied)
    sizes = [qa_class_cells.get(f, 0) for f in FAMILIES]
    weights = np.array([cell_counts[c] for c in occupied], dtype=float)
    Hs = []
    for _ in range(n_trials):
        perm = RNG.permutation(len(occupied))
        idx, start, class_mass = perm, 0, []
        for sz in sizes:
            grp = idx[start:start+sz]; start += sz
            class_mass.append(weights[grp].sum())
        Hs.append(shannon(class_mass))
    return np.array(Hs)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def analyze(name, x, tau=8, mode="equal_occ"):
    states = embed_to_states(x, tau=tau, mode=mode)
    H_pe = perm_entropy(x)
    H_qa, fam_counts = family_entropy(states)
    null = null_partition_entropy(states)
    z = (H_qa - null.mean()) / (null.std() + 1e-12)
    p = (np.sum(null <= H_qa) + 1) / (len(null) + 1)  # one-sided: QA MORE concentrated
    dist = {f: fam_counts.get(f, 0) for f in FAMILIES}
    return dict(name=name, H_pe=H_pe, H_qa=H_qa, H_max=log2(5),
                null_mean=null.mean(), null_std=null.std(),
                excess=null.mean()-H_qa, z=z, p=p, dist=dist)

if __name__ == "__main__":
    print("Estimating Lorenz largest Lyapunov exponent (Benettin)...")
    lam = lorenz_lyapunov()
    print(f"  lambda_1 ~ {lam:.4f} nats/time  (Pesin: h_KS = lambda_1 for Lorenz)")
    print(f"  literature reference ~ 0.906 nats/time\n")

    sources = {
        "Lorenz (chaotic)": lorenz(),
        "white noise": white_noise(),
        "periodic": periodic(),
    }
    print(f"{'source':22s} {'H_pe':>6s} {'H_qa':>6s} {'null_mu':>8s} "
          f"{'excess':>7s} {'z':>7s} {'p':>8s}")
    print("-"*72)
    results = {}
    for name, x in sources.items():
        r = analyze(name, x); results[name] = r
        print(f"{name:22s} {r['H_pe']:6.3f} {r['H_qa']:6.3f} {r['null_mean']:8.3f} "
              f"{r['excess']:+7.3f} {r['z']:+7.2f} {r['p']:8.4f}")
    print("-"*72)
    print(f"(H_max over 5 families = {log2(5):.3f} bits)\n")
    for name, r in results.items():
        print(f"{name}: family occupancy = {r['dist']}")

    print("\n=== ROBUSTNESS: does the Lorenz null hold across tau + binning? ===")
    print(f"{'mode':12s} {'tau':>4s} {'H_qa':>6s} {'null_mu':>8s} {'excess':>7s} {'z':>7s} {'p':>8s}")
    print("-"*56)
    lx = sources["Lorenz (chaotic)"]
    for mode in ("equal_occ", "fixed_w"):
        for tau in (1, 4, 8, 16, 32):
            r = analyze("L", lx, tau=tau, mode=mode)
            print(f"{mode:12s} {tau:4d} {r['H_qa']:6.3f} {r['null_mean']:8.3f} "
                  f"{r['excess']:+7.3f} {r['z']:+7.2f} {r['p']:8.4f}")
