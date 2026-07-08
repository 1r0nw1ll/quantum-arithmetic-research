#!/usr/bin/env python3
# RT1_OBSERVER_FILE: continuous-signal analysis (coherent/Lorenz/noise driving the QA engine); trig acts on observer-layer signals, never QA state.
"""
QA Syntropy as "order out of noise" — QA-native order metric.

Facet under test (source material): syntropy = "spontaneous self-organization,
creation of usable signal or coherence out of noise". QA-native reading: when
the self-organizing QASystem is driven by a source, does it converge to a MORE
ORDERED attractor (higher E8 alignment / harmonic index) than when free-running
or driven by noise? And does the ordering reflect the source's structure?

Metric = QA's own emergent-order scores (E8 alignment, harmonic index),
steady-state mean over the last window, across seeds. This is the QA-native
order measure, not a foreign linear decode.

Sources: coherent (deterministic tones), Lorenz (deterministic chaos),
red noise AR(1) (autocorrelated stochastic), white noise (structureless),
free-run (no drive). Falsifiable prediction if QA is syntropic:
structured drives -> higher steady order than white noise / free-run.
"""
from __future__ import annotations

QA_COMPLIANCE = "observer=continuous_signal_injection_into_QASystem_b_state, state_alphabet=mod24_A1; negative-result companion to cert [518], see docs/theory/QA_SYNTROPY_PHASE_CONJUGATE_INVESTIGATION.md"
import numpy as np
from scipy.stats import ttest_ind
from qa_core.engine import QASystem

M, N, T = 24, 24, 4000
WIN = 500

def coherent(n):
    tt = np.arange(n)
    s = np.sin(2*np.pi*tt/220) + 0.6*np.sin(2*np.pi*tt/90+0.5)
    return (s-s.mean())/s.std()

def lorenz(n, dt=0.01, s=10., r=28., b=8/3., tr=5000):
    xyz=np.array([1.,1.,1.])
    f=lambda v:np.array([s*(v[1]-v[0]), v[0]*(r-v[2])-v[1], v[0]*v[1]-b*v[2]])
    def rk4(v):
        k1=f(v);k2=f(v+.5*dt*k1);k3=f(v+.5*dt*k2);k4=f(v+dt*k3)
        return v+dt/6*(k1+2*k2+2*k3+k4)
    for _ in range(tr): xyz=rk4(xyz)
    o=np.empty(n)
    for i in range(n): xyz=rk4(xyz); o[i]=xyz[0]
    return (o-o.mean())/o.std()

def red(n, phi=0.95, rng=None):
    rng=rng or np.random.default_rng(0)
    x=np.empty(n);x[0]=0.
    for i in range(1,n): x[i]=phi*x[i-1]+rng.standard_normal()
    return (x-x.mean())/x.std()

def white(n, rng=None):
    rng=rng or np.random.default_rng(0)
    x=rng.standard_normal(n); return (x-x.mean())/x.std()

def steady_order(signal_data, seed, inj=0.1):
    np.random.seed(seed)
    sys=QASystem(num_nodes=N, modulus=M, coupling=0.15, noise_base=0.5,
                 noise_annealing=0.999, signal_injection_strength=inj,
                 signal_mode="final")
    sys.run_simulation(len(signal_data), signal_data, progress=False)
    e8=np.asarray(sys.history["e8_alignment"])
    hi=np.asarray(sys.history["hi"])
    return e8[-WIN:].mean(), hi[-WIN:].mean()

if __name__ == "__main__":
    seeds=range(15)
    srcs={}
    for seed in seeds:
        rng=np.random.default_rng(2000+seed)
        srcs.setdefault("coherent",[]);srcs.setdefault("lorenz",[])
        srcs.setdefault("red_noise",[]);srcs.setdefault("white_noise",[])
        srcs.setdefault("free_run",[])
    data={
        "coherent":  lambda seed: coherent(T),
        "lorenz":    lambda seed: lorenz(T),
        "red_noise": lambda seed: red(T, rng=np.random.default_rng(2000+seed)),
        "white_noise":lambda seed: white(T, rng=np.random.default_rng(2000+seed)),
        "free_run":  lambda seed: np.zeros(T),
    }
    print("QA syntropy / order-out-of-noise — injection-strength sweep")
    print(f"N={N}, T={T}, window={WIN}, {len(list(seeds))} seeds. Metric: steady E8 alignment.\n")
    print(f"{'inj':>5s} | {'coherent':>16s} {'lorenz':>16s} {'white_noise':>16s} "
          f"{'free_run':>16s} | {'coh-white p':>11s}")
    print("-"*92)
    for inj in (0.1, 0.5, 1.0, 2.0, 4.0):
        E8={}
        for k,gen in data.items():
            E8[k]=np.array([steady_order(gen(seed), seed, inj=inj)[0] for seed in seeds])
        _,p=ttest_ind(E8["coherent"],E8["white_noise"],equal_var=False)
        def ms(k): return f"{E8[k].mean():.4f}+/-{E8[k].std():.4f}"
        print(f"{inj:5.1f} | {ms('coherent'):>16s} {ms('lorenz'):>16s} "
              f"{ms('white_noise'):>16s} {ms('free_run'):>16s} | {p:11.4f}")
    print("-"*92)
    print("If order stays flat across injection -> drive can't move QA's intrinsic order.")
