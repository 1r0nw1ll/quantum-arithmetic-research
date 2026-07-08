#!/usr/bin/env python3
# RT1_OBSERVER_FILE: continuous-signal analysis (Lorenz/Rossler/sine/noise, permutation entropy, Lyapunov); trig/sqrt act on observer-layer signals, never QA state.
"""
Syntropic Projector — DYNAMICS-AWARE fold (candidate cert [518], v2).

v1 (static Eisenstein-norm fold) gave a robust NULL: the norm partition is a
lattice invariant blind to temporal structure. v2 replaces the fold with the
actual QA DYNAMICS operator and asks a sharper, dynamical question.

Key idea: under a tau=1 delay embedding, b_t=bin(x_t), e_t=bin(x_{t+1}).
QA-native flow qa_step(b,e)=(e,(b+e-1)%m+1) then PREDICTS the next bin via a
QA additive (Fibonacci-type) recurrence:
      bin(x_{t+2})  ?=  (bin(x_t)+bin(x_{t+1})-1) % m + 1
The b-component (next b = current e) is automatic for a tau=1 embedding; the
e-component is the non-trivial content. So we test:

  TEST A (QA-flow match): does a deterministic chaotic trajectory obey the QA
    additive recurrence more than chance / more than a time-shuffled null?

  TEST B (resonance-flow concentration): quantize the canonical einsum
    resonance <T_t,T_{t+1}> between consecutive QA 4-tuples; does its
    distribution concentrate for chaos vs a time-shuffle null?

Null = time-shuffle of the SAME series (identical marginal, destroyed
dynamics) -> the proper null for a dynamical claim. Match rate above the
shuffle null = QA dynamics genuinely aligned with attractor dynamics.

Boundary crossings respected (Theorem NT): continuous -> bins [crossing 1];
all QA ops integer (T2); symbol/statistic out [crossing 2].
A1/S1 compliant.
"""
from __future__ import annotations

QA_COMPLIANCE = "observer=continuous_signal_to_phase_bins, state_alphabet=mod{9,24}_A1; negative-result companion to cert [518], see docs/theory/QA_SYNTROPY_PHASE_CONJUGATE_INVESTIGATION.md"
import numpy as np
from collections import Counter
from math import log2

RNG = np.random.default_rng(42)
M = 9

def qa_mod(x, m=M): return ((int(x) - 1) % m) + 1
def qa_step(b, e, m=M): return e, ((b + e - 1) % m) + 1
def qa_tuple(b, e, m=M):
    d = ((b + e - 1) % m) + 1
    a = ((b + 2 * e - 1) % m) + 1
    return (b, e, d, a)

def lorenz(n=60000, dt=0.01, s=10.0, r=28.0, bta=8.0/3.0, transient=5000):
    xyz = np.array([1.0, 1.0, 1.0])
    def f(v):
        x, y, z = v
        return np.array([s*(y-x), x*(r-z)-y, x*y-bta*z])
    for _ in range(transient):
        k1=f(xyz); k2=f(xyz+0.5*dt*k1); k3=f(xyz+0.5*dt*k2); k4=f(xyz+dt*k3)
        xyz = xyz + dt/6*(k1+2*k2+2*k3+k4)
    out = np.empty(n)
    for i in range(n):
        k1=f(xyz); k2=f(xyz+0.5*dt*k1); k3=f(xyz+0.5*dt*k2); k4=f(xyz+dt*k3)
        xyz = xyz + dt/6*(k1+2*k2+2*k3+k4)
        out[i] = xyz[0]
    return out

def rossler(n=60000, dt=0.05, a=0.2, b=0.2, c=5.7, transient=5000):
    xyz = np.array([1.0, 1.0, 1.0])
    def f(v):
        x, y, z = v
        return np.array([-y - z, x + a*y, b + z*(x - c)])
    for _ in range(transient):
        k1=f(xyz); k2=f(xyz+0.5*dt*k1); k3=f(xyz+0.5*dt*k2); k4=f(xyz+dt*k3)
        xyz = xyz + dt/6*(k1+2*k2+2*k3+k4)
    out = np.empty(n)
    for i in range(n):
        k1=f(xyz); k2=f(xyz+0.5*dt*k1); k3=f(xyz+0.5*dt*k2); k4=f(xyz+dt*k3)
        xyz = xyz + dt/6*(k1+2*k2+2*k3+k4)
        out[i] = xyz[0]
    return out

def logistic(n=60000, r=3.99, transient=2000):
    x = 0.4; out = np.empty(n)
    for _ in range(transient): x = r*x*(1-x)
    for i in range(n): x = r*x*(1-x); out[i] = x
    return out

def white_noise(n=60000): return RNG.standard_normal(n)
def red_noise(n=60000, phi=0.95):
    """AR(1): smooth + autocorrelated but NOT chaotic (no positive Lyapunov)."""
    x = np.empty(n); x[0] = 0.0
    for i in range(1, n):
        x[i] = phi*x[i-1] + RNG.standard_normal()
    return x
def periodic(n=60000):
    t = np.arange(n)*0.05
    return np.sin(t) + 0.5*np.sin(2.3*t)

def to_bins(x, m=M, mode="equal_occ"):
    if mode == "equal_occ":
        ranks = np.argsort(np.argsort(x))
        return (ranks*m//len(x)) + 1
    lo, hi = x.min(), x.max()
    return np.clip(((x-lo)/(hi-lo+1e-12)*m).astype(int), 0, m-1) + 1

def shannon(counts):
    tot = sum(counts)
    return -sum((c/tot)*log2(c/tot) for c in counts if c > 0) if tot else 0.0

# --- TEST A: QA additive-recurrence match rate -----------------------------
def qa_flow_match_rate(bins, m=M):
    """Fraction of t where bin[t+2] == (bin[t]+bin[t+1]-1)%m+1 (QA e-recurrence)."""
    b = bins[:-2]; e = bins[1:-1]; nxt = bins[2:]
    pred = ((b + e - 1) % m) + 1
    return np.mean(pred == nxt)

# --- TEST B: resonance-flow concentration ----------------------------------
def resonance_stream(bins, m=M, nbins=12):
    """Quantized canonical einsum resonance <T_t, T_{t+1}> between consecutive
    QA 4-tuples along the trajectory."""
    T = np.array([qa_tuple(int(bins[i]), int(bins[i+1]), m) for i in range(len(bins)-1)])
    # consecutive coupling via einsum('ik,jk->ij') diagonal-shifted = row dot next row
    res = np.einsum('ik,ik->i', T[:-1], T[1:])
    lo, hi = res.min(), res.max()
    q = np.clip(((res-lo)/(hi-lo+1e-12)*nbins).astype(int), 0, nbins-1)
    return q

def resonance_entropy(bins):
    q = resonance_stream(bins)
    return shannon(list(Counter(q.tolist()).values()))

def analyze(name, x, mode="equal_occ", n_shuffle=500):
    bins = to_bins(x, mode=mode)
    # TEST A
    match = qa_flow_match_rate(bins)
    sh_match = np.array([qa_flow_match_rate(RNG.permutation(bins)) for _ in range(n_shuffle)])
    zA = (match - sh_match.mean())/(sh_match.std()+1e-12)
    pA = (np.sum(sh_match >= match)+1)/(n_shuffle+1)  # one-sided: MORE match than shuffle
    # TEST B
    Hres = resonance_entropy(bins)
    sh_H = np.array([resonance_entropy(RNG.permutation(bins)) for _ in range(n_shuffle)])
    zB = (Hres - sh_H.mean())/(sh_H.std()+1e-12)
    pB = (np.sum(sh_H <= Hres)+1)/(n_shuffle+1)  # one-sided: MORE concentrated
    chance = 1.0/M
    return dict(name=name, match=match, chance=chance, sh_match=sh_match.mean(),
                zA=zA, pA=pA, Hres=Hres, sh_H=sh_H.mean(), zB=zB, pB=pB)

if __name__ == "__main__":
    sources = {
        "Lorenz (chaotic)": lorenz(),
        "Rossler (chaotic)": rossler(),
        "logistic r=3.99": logistic(),
        "white noise": white_noise(),
        "red noise AR(1)": red_noise(),
        "periodic": periodic(),
    }
    print("TEST A — QA additive-recurrence match rate  (chance=1/9=0.1111)")
    print("TEST B — resonance-flow entropy vs time-shuffle null\n")
    print(f"{'source':20s} | {'matchA':>7s} {'shuf':>6s} {'zA':>7s} {'pA':>7s} "
          f"| {'Hres':>6s} {'shuf':>6s} {'zB':>7s} {'pB':>7s}")
    print("-"*92)
    for name, x in sources.items():
        r = analyze(name, x)
        print(f"{name:20s} | {r['match']:7.4f} {r['sh_match']:6.4f} {r['zA']:+7.2f} "
              f"{r['pA']:7.4f} | {r['Hres']:6.3f} {r['sh_H']:6.3f} {r['zB']:+7.2f} {r['pB']:7.4f}")
    print("-"*92)
    print("zA>0 & pA<0.05  -> chaotic dynamics obey QA additive recurrence above shuffle")
    print("zB<0 (Hres<shuf, pB small) -> QA resonance flow concentrates vs shuffle")
