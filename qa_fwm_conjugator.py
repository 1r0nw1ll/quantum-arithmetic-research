#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=intensity_to_phase_projection, state_alphabet=mod24_A1_compliant"
# RT1_OBSERVER_FILE: image-intensity synthesis (make_pattern/smooth_screen) is an
# observer projection (Theorem NT inbound crossing); trig is on continuous intensity,
# never on QA state. The QA layer (group/FWM/aberrate/recover) is pure integer.
"""
qa_fwm_conjugator.py — QA Four-Wave-Mixing Phase Conjugator (candidate cert [518]).

Builds the *explicit* conjugate-generating operator that emergent QA dynamics
(QASystem consensus coupling, rolling QCI) were shown NOT to implement. Grounded
in the distortion-correction theorem (Yariv/Zel'dovich; Agarwal-Friberg proof):
a phase-conjugated wave returned through the SAME distorting medium exactly
undoes the distortion; a different medium does not.

## The physics -> QA mapping

Degenerate four-wave mixing generates the conjugate via the PHASE-SUM relation
    theta_c = theta_f + theta_b - theta_s          (two pumps + signal, conjugated)
With conjugate pumps (theta_b = -theta_f) this collapses to theta_c = -theta_s.

QA realizes this in the modular additive group on the A1 alphabet {1,...,m}:
    qa_add(a,b) = qa_mod(a+b)      identity = m  (No-Zero representative of 0)
    qa_neg(a)   = qa_mod(-a)       phase conjugation (an involution)
    FWM(p_f,p_b,s) = qa_mod(p_f + p_b - s)                 # phase-sum relation
Conjugate pumps p_b = qa_neg(p_f)  =>  FWM = qa_neg(s) exactly, for any p_f.

Distorting medium = phase screen phi (per-site modular shift):
    aberrate(s, phi) = qa_add(s, phi)
Distortion-correction theorem, exact in QA:
    return_same = qa_add( FWM(aberrate(s,phi)), phi ) = qa_neg(s)   for ALL s, phi
    return_diff (phi' != phi) leaves residual qa_add(qa_neg(s), phi'-phi)

Everything in the QA layer is integer state in {1,...,m} (A1/S2). The observer
boundary is crossed exactly twice for the image demo: intensity -> phase state
[inbound], recovered phase state -> intensity [outbound] (Theorem NT). No **2 (S1).

Run:  python qa_fwm_conjugator.py
Outputs: qa_fwm_conjugator_demo.png (Original | Aberrated | Recovered | Wrong-medium)
"""
from __future__ import annotations
import numpy as np

M = 24  # applied QA modulus


# ---------------------------------------------------------------------------
# QA additive group on the A1 alphabet {1,...,M}  (identity = M, never 0)
# ---------------------------------------------------------------------------
def qa_mod(x: np.ndarray | int) -> np.ndarray | int:
    """A1-compliant modular reduction into {1,...,M}."""
    return ((np.asarray(x, dtype=np.int64) - 1) % M) + 1


def qa_add(a, b):
    """Phase addition (group op). Identity element is M."""
    return qa_mod(np.asarray(a, np.int64) + np.asarray(b, np.int64))


def qa_neg(a):
    """Phase conjugation theta -> -theta. Involution; two fixed points (M, M/2)."""
    return qa_mod(-np.asarray(a, np.int64))


def fwm_conjugate(p_f, p_b, s):
    """Four-wave-mixing phase-sum relation: theta_c = theta_f + theta_b - theta_s.
    With conjugate pumps p_b = qa_neg(p_f) this returns qa_neg(s) exactly."""
    return qa_mod(np.asarray(p_f, np.int64) + np.asarray(p_b, np.int64)
                  - np.asarray(s, np.int64))


def aberrate(s, phi):
    """Pass a wavefront through a phase-screen medium phi (per-site shift)."""
    return qa_add(s, phi)


# ---------------------------------------------------------------------------
# 1. Theorem verification + medium-mismatch sweep
# ---------------------------------------------------------------------------
def recover(s, phi_forward, phi_return, p_f, pump_err=0):
    """Full pipeline: aberrate -> FWM conjugate -> return through a medium.
    Returns the recovered wavefront (ideally qa_neg(s) when media match)."""
    p_b = qa_add(qa_neg(p_f), pump_err)          # conjugate pump (+ optional error)
    distorted = aberrate(s, phi_forward)         # forward pass through medium
    c = fwm_conjugate(p_f, p_b, distorted)       # phase-conjugate mirror
    return qa_add(c, phi_return)                 # return pass through a medium


def fidelity(recovered, target):
    return float(np.mean(recovered == target))


def mismatch_sweep(L=4000, seed=0):
    """Recovery fidelity vs fraction of the return screen that differs from the
    forward screen. Same medium -> 1.0 exact; fully different -> chance 1/M."""
    rng = np.random.default_rng(seed)
    s = rng.integers(1, M + 1, L)
    phi = rng.integers(1, M + 1, L)
    p_f = rng.integers(1, M + 1, L)
    target = qa_neg(s)                            # the theorem's exact output
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    out = []
    for f in fracs:
        phi_ret = phi.copy()
        k = int(f * L)
        idx = rng.choice(L, k, replace=False)
        phi_ret[idx] = rng.integers(1, M + 1, k)  # re-randomize a fraction
        rec = recover(s, phi, phi_ret, p_f)
        out.append((f, fidelity(rec, target)))
    return out


def robustness(L=4000, seed=1):
    """Same-medium recovery fidelity under imperfect pump conjugation and under
    additive medium phase-noise on the return pass."""
    rng = np.random.default_rng(seed)
    s = rng.integers(1, M + 1, L)
    phi = rng.integers(1, M + 1, L)
    p_f = rng.integers(1, M + 1, L)
    target = qa_neg(s)
    pump_rows, noise_rows = [], []
    for err_lvl in (0, 1, 2, 4):
        pe = rng.integers(0, err_lvl + 1, L) * rng.choice([-1, 1], L) if err_lvl else 0
        rec = recover(s, phi, phi, p_f, pump_err=pe)
        pump_rows.append((err_lvl, fidelity(rec, target)))
    for noise_lvl in (0, 1, 2, 4):
        nz = rng.integers(0, noise_lvl + 1, L) * rng.choice([-1, 1], L) if noise_lvl else 0
        phi_ret = qa_add(phi, nz)                 # jitter on the return medium
        rec = recover(s, phi, phi_ret, p_f)
        noise_rows.append((noise_lvl, fidelity(rec, target)))
    return pump_rows, noise_rows


def controls(L=4000, seed=2):
    """Show the conjugation is load-bearing: (a) no correction (just view the
    distorted wave), (b) a NON-conjugate mixer (pumps not conjugate)."""
    rng = np.random.default_rng(seed)
    s = rng.integers(1, M + 1, L)
    phi = rng.integers(1, M + 1, L)
    p_f = rng.integers(1, M + 1, L)
    target = qa_neg(s)
    no_correct = fidelity(aberrate(s, phi), s)                    # distorted vs clean
    p_b_bad = rng.integers(1, M + 1, L)                           # random 2nd pump
    c_bad = fwm_conjugate(p_f, p_b_bad, aberrate(s, phi))
    non_conj = fidelity(qa_add(c_bad, phi), target)              # non-conjugate mixer
    fwm_ok = fidelity(recover(s, phi, phi, p_f), target)          # proper FWM
    return no_correct, non_conj, fwm_ok


# ---------------------------------------------------------------------------
# 2. Image demo — the classic "recover the wavefront through frosted glass"
# ---------------------------------------------------------------------------
def make_pattern(n=72):
    """Synthetic recognizable pattern: 'QA' glyphs + rings, intensity in [0,1]."""
    img = np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2)          # noqa: S1 (float observer)
    img += 0.5 * (np.sin(r / 2.2) * 0.5 + 0.5)                  # concentric rings
    # block letters Q and A
    img[12:30, 12:24] = 1.0; img[16:26, 16:20] = 0.15           # Q body
    img[26:32, 22:30] = 1.0                                      # Q tail
    img[12:30, 44:48] = 1.0; img[12:30, 56:60] = 1.0            # A legs
    img[12:16, 44:60] = 1.0; img[20:24, 44:60] = 1.0            # A bars
    return np.minimum(img, 1.0)  # non-negative by construction; cap intensity at 1


def smooth_screen(n, seed):
    """Spatially-correlated random phase screen (atmospheric-turbulence-like),
    quantized to per-site modular shifts in {1,...,M}."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n, n))
    for _ in range(6):                                          # cheap smoothing
        field = (field
                 + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                 + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5.0
    field = (field - field.min()) / (np.ptp(field) + 1e-9)
    return qa_mod((field * M).astype(np.int64) + 1)


def encode(img):
    """Observer boundary crossing #1: intensity [0,1] -> phase state {1,...,M}."""
    return qa_mod((img * (M - 1)).astype(np.int64) + 1)


def decode(state):
    """Observer boundary crossing #2: phase state {1,...,M} -> intensity [0,1]."""
    return (np.asarray(state) - 1) / (M - 1)


def image_demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 72
    img = make_pattern(n)
    s = encode(img)
    phi = smooth_screen(n, seed=7)
    phi_wrong = smooth_screen(n, seed=8)                        # different medium
    p_f = np.random.default_rng(9).integers(1, M + 1, (n, n))

    distorted = aberrate(s, phi)
    recovered_same = qa_neg(recover(s, phi, phi, p_f))          # undo final conj -> s
    recovered_wrong = qa_neg(recover(s, phi, phi_wrong, p_f))

    fid_same = fidelity(recovered_same, s)
    fid_wrong = fidelity(recovered_wrong, s)

    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    for a in ax:
        a.axis("off")
    ax[0].imshow(decode(s), cmap="magma");            ax[0].set_title("Original\n(QA phase)")
    ax[1].imshow(decode(distorted), cmap="magma");    ax[1].set_title("Aberrated\n(phase screen)")
    ax[2].imshow(decode(recovered_same), cmap="magma")
    ax[2].set_title(f"Recovered — SAME medium\nfidelity {fid_same:.3f}")
    ax[3].imshow(decode(recovered_wrong), cmap="magma")
    ax[3].set_title(f"Wrong medium (control)\nfidelity {fid_wrong:.3f}")
    fig.suptitle("QA Four-Wave-Mixing Phase Conjugation — distortion-correction theorem",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("qa_fwm_conjugator_demo.png", dpi=120)
    return fid_same, fid_wrong


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("QA FOUR-WAVE-MIXING PHASE CONJUGATOR  (m=24)\n")

    print("[1] Distortion-correction theorem — medium-mismatch sweep")
    print("    (fraction of return screen re-randomized  ->  recovery fidelity)")
    for f, fid in mismatch_sweep():
        bar = "#" * round(fid * 40)
        print(f"      mismatch {f:4.0%}   fidelity {fid:6.3f}  {bar}")
    print(f"      chance level = 1/m = {1/M:.3f}\n")

    print("[2] Robustness (same medium)")
    pump_rows, noise_rows = robustness()
    print("    imperfect pump conjugation (max |error| in phase units):")
    for lvl, fid in pump_rows:
        print(f"      pump_err <= {lvl}   fidelity {fid:6.3f}")
    print("    return-medium phase jitter (max |noise|):")
    for lvl, fid in noise_rows:
        print(f"      noise    <= {lvl}   fidelity {fid:6.3f}\n")

    print("[3] Controls (conjugation is load-bearing)")
    no_correct, non_conj, fwm_ok = controls()
    print(f"      no correction (distorted vs clean) : {no_correct:6.3f}")
    print(f"      non-conjugate mixer                : {non_conj:6.3f}")
    print(f"      proper FWM conjugator              : {fwm_ok:6.3f}\n")

    print("[4] Image demo -> qa_fwm_conjugator_demo.png")
    fs, fw = image_demo()
    print(f"      recovered (same medium)  fidelity {fs:.3f}")
    print(f"      recovered (wrong medium) fidelity {fw:.3f}")
