#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=morphology_field_to_phase, state_alphabet=mod24_A1_compliant"
# RT1_OBSERVER_FILE: body-plan field synthesis (gradients/radial via sin/sqrt) is an
# observer projection onto the 2D morphology field; the QA layer (phase patterns,
# phase-conjugate memory) is integer {1..m}. Damage/perturbation are observer-layer.
"""
QA Phase-Conjugate Morphogenetic Memory (candidate cert [521]).

Michael Levin's bioelectric morphogenesis: the TARGET MORPHOLOGY (correct body
plan) is stored as an attractor, and the tissue navigates back to it from
perturbed / damaged starting states (regeneration). This builds that as the
phase-conjugate associative memory of cert [519] (on cert [518]'s exact
conjugator): body plans are 2D QA phase fields, stored as attractors; a DAMAGED
(amputated) field is regrown to the correct target by content-addressable recall.

Distinct from qa_brainca_morphogenesis_v3.py (organizer-cell CA on one target):
here MULTIPLE body plans are stored and a damaged one regenerates to the RIGHT
plan (no chimera), robust to systemic (global) perturbation via [518] phase-lock.

Falsifiable claims:
  - regeneration: a body plan regrows from localized amputation up to a basin
  - which-morphology: a damaged plan converges to its OWN stored plan, not another
  - no-chimera: the regrown field is a clean stored plan, not a mixture
  - systemic tolerance: a global bioelectric shift is undone by phase-lock

A1/S2/Theorem-NT: phase state integer in {1..m}; observer boundary crossed at
field->phase and phase->field only.
"""
from __future__ import annotations
import numpy as np
from qa_phase_conjugate_memory import QAPhaseConjugateMemory, qa_add, qa_neg, qa_mod, M

RNG = np.random.default_rng(42)
SIZE = 16                      # 16x16 morphology field
N = SIZE * SIZE


# ---------------------------------------------------------------------------
# Body plans — distinct 2D morphological phase fields (observer synthesis)
# ---------------------------------------------------------------------------
def _to_phase(field: np.ndarray) -> np.ndarray:
    """Observer boundary: continuous morphology field in [0,1] -> phase {1..M}."""
    return qa_mod((field * (M - 1)).astype(np.int64) + 1).ravel()


def body_plans() -> dict:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1)          # noqa: RT1 observer
    plans = {
        "anterior_posterior": xx,                            # head->tail gradient
        "dorsal_ventral": yy,                                # top->bottom gradient
        "radial_whole": 1 - np.sqrt((xx - .5) ** 2 + (yy - .5) ** 2) / 0.71,  # concentric
        "bipolar_twohead": np.abs(xx - 0.5) * 2,             # symmetric two-ended
    }
    return {name: _to_phase(np.clip(f, 0.0, 1.0)) for name, f in plans.items()}


def similar_plans(n_variants=6, jitter=0.25):
    """Correlated morphologies: noisy variants of the base body plans (shared
    structure) — the honest stress case where chimeras / wrong-plan could form."""
    base = body_plans()
    out = {}
    for name, p in base.items():
        for v in range(n_variants):
            x = p.copy()
            idx = RNG.choice(N, int(jitter * N), replace=False)
            x[idx] = RNG.integers(1, M + 1, len(idx))
            out[f"{name}_{v}"] = x
    return out


# ---------------------------------------------------------------------------
# Damage models (observer-layer perturbations of the morphology field)
# ---------------------------------------------------------------------------
def amputate(pattern: np.ndarray, frac: float, rng) -> np.ndarray:
    """Remove a CONTIGUOUS region (columns) — an amputation, not scattered noise.
    Removed cells are re-randomized (undifferentiated tissue)."""
    x = pattern.copy().reshape(SIZE, SIZE)
    n_cols = int(round(frac * SIZE))
    if n_cols > 0:
        start = rng.integers(0, SIZE - n_cols + 1)
        x[:, start:start + n_cols] = rng.integers(1, M + 1, (SIZE, n_cols))
    return x.ravel()


def systemic_shift(pattern: np.ndarray, phi: int) -> np.ndarray:
    """Global bioelectric offset applied to the whole field (a systemic perturbation)."""
    return qa_add(pattern, phi)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def regen_fidelity(recovered, target):
    return float(np.mean(recovered == target))


def which_plan(recovered, plans):
    names = list(plans)
    d = [np.sum(recovered != plans[n]) for n in names]
    return names[int(np.argmin(d))], min(d)


def run():
    plans = body_plans()
    names = list(plans)
    P = np.stack([plans[n] for n in names])
    mem = QAPhaseConjugateMemory(P, sharpen=6.0)
    print(f"QA PHASE-CONJUGATE MORPHOGENETIC MEMORY  ({len(names)} body plans, {SIZE}x{SIZE} field, m={M})\n")

    # [1] Regeneration from amputation: regrow the correct plan from a contiguous cut
    print("[1] Regeneration from amputation (contiguous region removed -> recall):")
    print(f"{'amputated':>10s} {'regen fidelity':>15s} {'correct plan':>13s}")
    for frac in (0.25, 0.5, 0.625, 0.75, 0.875):
        fid, correct = [], 0
        for _ in range(200):
            s = RNG.integers(len(names))
            damaged = amputate(P[s], frac, RNG)
            rec = mem.recall(damaged)
            fid.append(regen_fidelity(rec, P[s]))
            if which_plan(rec, plans)[0] == names[s]:
                correct += 1
        print(f"{frac:10.1%} {np.mean(fid):15.3f} {correct/200:13.3f}")

    # [1b] Honest stress: SIMILAR morphologies (noisy variants sharing structure)
    sim = similar_plans()
    Ps = np.stack([sim[n] for n in sim])
    mem_s = QAPhaseConjugateMemory(Ps, sharpen=6.0)
    sim_names = list(sim)
    print(f"\n[1b] Correlated-morphology stress ({len(sim_names)} variants of 4 plans, 75% shared):")
    print(f"{'amputated':>10s} {'regen fidelity':>15s} {'exact plan':>11s}")
    for frac in (0.25, 0.5, 0.625):
        fid, exact = [], 0
        for _ in range(200):
            s = RNG.integers(len(sim_names))
            rec = mem_s.recall(amputate(Ps[s], frac, RNG))
            fid.append(regen_fidelity(rec, Ps[s]))
            if np.array_equal(rec, Ps[s]):
                exact += 1
        print(f"{frac:10.1%} {np.mean(fid):15.3f} {exact/200:11.3f}")

    # [2] No-chimera: the regrown field must be an EXACT stored plan
    print("\n[2] No-chimera check (regrown field is an exact stored body plan?):")
    exact = 0
    for _ in range(300):
        s = RNG.integers(len(names))
        rec = mem.recall(amputate(P[s], 0.375, RNG))
        if any(np.array_equal(rec, P[k]) for k in range(len(names))):
            exact += 1
    print(f"      exact-stored-plan rate: {exact/300:.3f}  (fraction that are clean plans, not mixtures)")

    # [3] Systemic perturbation (global bioelectric shift) -> phase-locked regeneration
    print("\n[3] Systemic perturbation — global bioelectric shift (phi):")
    print(f"{'phi':>5s} {'naive regen':>12s} {'phase-locked':>13s}")
    for phi in (0, 3, 6, 12):
        naive_ok = pl_ok = 0
        for _ in range(200):
            s = RNG.integers(len(names))
            probe = systemic_shift(amputate(P[s], 0.25, RNG), phi)
            # correct target in the shifted frame:
            target_shifted = systemic_shift(P[s], phi)
            if np.array_equal(mem.recall(probe), target_shifted):
                naive_ok += 1
            if np.array_equal(mem.recall_phase_locked(probe), target_shifted):
                pl_ok += 1
        print(f"{phi:5d} {naive_ok/200:12.3f} {pl_ok/200:13.3f}")

    # [4] Basin control: random-guess baseline for "correct plan"
    print(f"\n(chance of picking correct plan = 1/{len(names)} = {1/len(names):.3f})")


if __name__ == "__main__":
    run()
