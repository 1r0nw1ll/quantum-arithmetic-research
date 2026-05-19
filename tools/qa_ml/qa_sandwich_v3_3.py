"""QA-ML v3.3 — sandwich-product features (E1) and QA-ReLU activation (E1.5).

Per docs/specs/QA_ML_PEPE_MAPPING_CATALOG.md.

E1 — sandwich-product (conjugate action) features:
  Pepe (2025) §1.3 / §4.4: the GA "sandwich" `g · x · g⁻¹` is
  conjugation in the group algebra. Applied to a vector x, it rotates
  /reflects x by the rotor g. Applied to ANOTHER generator h, it gives
  the conjugate `g h g⁻¹` — a rotated copy of h.

  The QA analog is conjugation of generator-h-action by generator-g:

      conjugate_action(g, h, x) := g(h(g⁻¹(x)))

  For (g, h) in the 4×4 grid {σ, μ, λ₂, ν}², this yields 16 conjugate
  actions per state. σ and μ are bijections on {1..m}² (globally
  invertible). λ₂ and ν are partial inverses of each other within
  their domains. When g⁻¹ or h is undefined at the relevant point,
  the conjugate returns (0, 0, 0, 0) — distinct from any in-domain
  output per A1.

  Diagonal pairs (g, g): when g is bijective this collapses to h(x).
  Off-diagonals carry the non-commutativity of the generators, which
  is exactly the structural information Pepe's sandwich pattern is
  designed to expose. σ and μ do not commute (verified by direct
  computation), so the off-diagonal conjugates are non-degenerate.

E1.5 — QA-ReLU:
  Phase-attenuated nonlinearity that respects QA orbit class. Pepe 2025
  §5.3 defines GA-ReLU as composition of (1) coefficient-wise ReLU and
  (2) phase-dependent cardioid attenuation. The QA analog uses the
  discrete orbit class (singularity / satellite / cosmos) as the gating
  signal and the position-within-8-cycle (for satellite states) as the
  discrete phase.

  Theorem NT compliance: QA-ReLU operates on the float activation in
  the ML head (observer-projection layer), gated by INTEGER QA
  classifications (orbit_family, position). The continuous side never
  feeds back into QA logic — it only shapes the loss gradient. This is
  the same boundary as the v3.x experiments already cross.

QA_COMPLIANCE = "qa_ml_sandwich_v3_3 — A1/A2/S1/S2 compliant; observer-layer activation"
"""

from __future__ import annotations

from math import cos, pi

from typing import Optional

from .qa_generators import sigma, mu, lambda_2, nu
from qa_orbit_rules import orbit_family, qa_step


def sigma_inv(b: int, e: int, m: int) -> tuple[int, int]:
    """Inverse of sigma. sigma(b, e) = (e, ((b+e-1) mod m) + 1).
    The inverse maps (b', e') = (e, ((b+e-1) mod m)+1) back to (b, e),
    so given (b', e') we recover b = ((e' - b' - 1) mod m) + 1, e = b'.
    Bijection on {1..m}^2.
    """
    assert 1 <= b <= m and 1 <= e <= m, f"A1: ({b},{e}) out of {{1,...,{m}}}"
    new_b = ((e - b - 1) % m) + 1
    new_e = b
    return (new_b, new_e)


def mu_inv(b: int, e: int, m: int) -> tuple[int, int]:
    """Inverse of mu (involution)."""
    return mu(b, e, m)


def lambda_2_inv(b: int, e: int, m: int) -> Optional[tuple[int, int]]:
    """Inverse of lambda_2 = nu within domain."""
    return nu(b, e, m)


def nu_inv(b: int, e: int, m: int) -> Optional[tuple[int, int]]:
    """Inverse of nu = lambda_2 within domain."""
    return lambda_2(b, e, m)


GEN_FWD = {"sigma": sigma, "mu": mu, "lambda_2": lambda_2, "nu": nu}
GEN_INV = {"sigma": sigma_inv, "mu": mu_inv, "lambda_2": lambda_2_inv, "nu": nu_inv}
GEN_ORDER = ("sigma", "mu", "lambda_2", "nu")


# Per qa/core.py QAState: the canonical QA state is the 4-tuple
# (b, e, d, a) with d = b+e and a = b+2e. Each conjugate action
# produces a 4-tuple of (b', e', d', a'). 4 generators g × 4 generators
# h × 4 coords = 64 features. The (d', a') projections expose
# information CART cannot compute from (b', e') alone (axis-aligned
# trees can't sum features).
SANDWICH_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"sw_{g}_{h}_{coord}"
    for g in GEN_ORDER
    for h in GEN_ORDER
    for coord in ("b", "e", "d", "a")
)


def conjugate_action(g_name: str, h_name: str, b: int, e: int, m: int) -> tuple[int, int, int, int]:
    """g(h(g_inv(x))) — Pepe sandwich conjugation. Returns the FULL
    4-tuple (b', e', d', a') of the conjugated state, where d' = b'+e'
    and a' = b'+2e' (raw, per A2; the tree gets all four coordinates
    rather than having to reconstruct d, a from b, e which it cannot
    do being axis-aligned).

    Returns (0, 0, 0, 0) when any step is out of domain (A1 reserves
    0 outside {1..m}).
    """
    g_inv = GEN_INV[g_name]
    h_fn = GEN_FWD[h_name]
    g_fn = GEN_FWD[g_name]

    step1 = g_inv(b, e, m)
    if step1 is None:
        return (0, 0, 0, 0)
    step2 = h_fn(step1[0], step1[1], m)
    if step2 is None:
        return (0, 0, 0, 0)
    step3 = g_fn(step2[0], step2[1], m)
    if step3 is None:
        return (0, 0, 0, 0)
    b2, e2 = step3
    d2 = b2 + e2          # A2 raw derivation (no mod reduction; T-operator only)
    a2 = b2 + 2 * e2
    return (b2, e2, d2, a2)


def sandwich_features(b: int, e: int, m: int) -> tuple[int, ...]:
    """64 integer features: full 4x4 conjugate-action grid, each
    conjugate exposing the 4-tuple (b', e', d', a').

    Per Will Dale's correction (2026-05-15): QA state is genuinely 4D
    (qa/core.py QAState.tuple() = (b, e, d, a)). Exposing only (b', e')
    of each conjugate hides the (d', a') projections, which an
    axis-aligned tree cannot compute from (b', e') alone. Pepe-style
    sandwich features must expose the full algebraic state.
    """
    assert isinstance(b, int) and isinstance(e, int) and isinstance(m, int), (
        f"S2: b={b!r}, e={e!r}, m={m!r} must be Python int"
    )
    assert 1 <= b <= m and 1 <= e <= m, f"A1: ({b},{e}) out of {{1,...,{m}}}"

    out = []
    for g_name in GEN_ORDER:
        for h_name in GEN_ORDER:
            r = conjugate_action(g_name, h_name, b, e, m)
            out.extend(r)
    return tuple(out)


def orbit_position_in_satellite(b: int, e: int, m: int) -> int:
    """For satellite states, return position k in {0,...,7} along the
    8-cycle from the lexicographically least state in the cycle. For
    non-satellites returns -1.

    Used by QA-ReLU's phase-dependent attenuation.
    """
    family = orbit_family(b, e, m)
    if family != "satellite":
        return -1

    orbit: list[tuple[int, int]] = []
    cur = (b, e)
    for k in range(8):
        orbit.append(cur)
        cur = qa_step(*cur, m)

    anchor = min(orbit)
    cur = anchor
    for k in range(8):
        if cur == (b, e):
            return k
        cur = qa_step(*cur, m)
    return 0


def qa_relu_scalar(activation: float, b: int, e: int, m: int) -> float:
    """QA-ReLU on a single scalar activation, gated by (b, e, m).

    - singularity: pass through (identity) — orbit is fixed, no
      attenuation needed.
    - satellite: scale by (1 + cos(2 · pi · k / 8)) / 2 where k is the
      position in the 8-cycle. Cardioid envelope in [0, 1].
    - cosmos: standard ReLU (max(activation, 0)).
    """
    family = orbit_family(b, e, m)
    if family == "singularity":
        return float(activation)
    if family == "satellite":
        k = orbit_position_in_satellite(b, e, m)
        envelope = (1.0 + cos(2.0 * pi * k / 8.0)) / 2.0
        return float(activation) * envelope
    return max(float(activation), 0.0)
