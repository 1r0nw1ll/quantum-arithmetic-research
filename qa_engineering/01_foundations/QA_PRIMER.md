# QA Primer: What It Is and Why It Matters for SVP Engineering

## The One-Paragraph Version

Quantum Arithmetic (QA) is a **modular arithmetic system** where every configuration is described by a pair of numbers `(b, e)` and every transition is performed by one of four **generators** (sigma, mu, lambda, nu). The system has three natural orbit types — singularity, satellite, and cosmos — and every reachable state belongs to one. The key discovery is that these orbit types appear across physically distinct domains: sound resonance, seismic propagation, neural network training, and Pythagorean geometry all instantiate the same underlying structure. This is not analogy — it is a proved theorem, certified by formal validators.

---

## The SVP Connection

Dale Pond's SVP research establishes that sympathetic resonance follows laws: certain frequencies attract, certain transitions are lawful, certain states are attainable from certain starting conditions. QA is the **arithmetic shadow** of those laws.

Where SVP observes:
- A Chladni plate moves from no pattern → stripes → hexagons as frequency increases
- A guitar string can reach harmonics but not arbitrary frequencies
- Resonance is not continuous — it is discrete, orbit-governed

QA formalizes:
- `flat → stripes → hexagons` maps exactly to `singularity → satellite → cosmos`
- The path length is k=2 generator steps (proved, not assumed)
- The transition is governed by specific generators (`increase_amplitude`, `set_frequency`)
- If you apply the wrong generator in the wrong order, you get a **classified failure** — not chaos, but a named, deterministic error

---

## Why This Matters for Engineers

**1. You can specify what you want.**
Instead of tuning by intuition, you specify a target orbit family and ask: what generator sequence gets me there from my current state? The system can plan the path (BFS over the orbit graph) and certify the plan.

**2. You can know when something is impossible.**
QA has an **obstruction spine**: arithmetic properties of the target state (specifically, the p-adic valuation `v_p(r)`) determine whether it is reachable at all. If `v_p(r) = 1` for an inert prime `p`, the target is arithmetically forbidden and no search is needed — the planner prunes it with zero nodes expanded.

**3. The same laws work across domains.**
Once you understand the orbit trajectory `singularity → satellite → cosmos` in cymatics, you already understand it in seismology, signal processing, and any other domain you map onto QA. The compiler law is structural, not domain-specific.

**4. Every result is verifiable.**
All claims in QA engineering are backed by machine-checkable certificates. When you build something with QA, you produce a certificate that any AI or human can independently verify.

---

## The Three Orbit Families

Under mod-24 arithmetic (the standard for applied QA):

| Orbit Family | State count | Character | SVP analogue |
|-------------|-------------|-----------|--------------|
| **Singularity** | 1 state | Fixed point: no generator moves you | Pure unison, no harmonic structure |
| **Satellite** | 8 states | 3D symmetric structure, 2-cycle under μ | Partial/transitional resonance |
| **Cosmos** | 72 states (3 orbits of 24) | 1D linear structure, full dynamics | Full harmonic resonance achieved |

Most engineering work involves navigating from Singularity (initial quiescence) through Satellite (transition) to Cosmos (target resonance achieved).

---

## The Modular Arithmetic

QA uses **mod 9** (theoretical/Pythagorean work) or **mod 24** (applied experiments).

The "no zero element" rule: QA uses `{1, 2, ..., 9}` or `{1, 2, ..., 24}`, not `{0, 1, ..., N-1}`. This is not a quirk — it reflects the SVP principle that the zero state (pure void) is outside the system.

The state `(b, e)` generates a 4-tuple `(b, e, d, a)`:
- `d = b + e` (sum)
- `a = b + 2e` (extended sum)
- `d` and `a` are **derived** — they are never independent variables

This 4-tuple is the fundamental object. All invariants, all metrics, all orbit classifications operate on it.

---

## The Key Invariant: Q(√5) Norm

The function `f(b, e) = b² + be - e²` is the norm in **Q(√5)** (the golden ratio number field). This is the invariant that classifies orbits:

- `v₃(f) ≥ 2` → degenerate orbit (length 1 or 4)
- `v₃(f) = 0` → cosmos orbit (length 12 under mod-9, 24 under mod-24)

The golden ratio φ appears here structurally: `T = F² = ×φ²` in Z[φ]. The QA generator `F` is the Fibonacci-like matrix `[[0,1],[1,1]]`, and the five Pythagorean families are exactly its orbits in `(Z/9Z)²`.

This is why QA connects to Dale Pond's harmonic geometry: the golden ratio and Fibonacci sequence are not decoration — they are the arithmetic backbone of the orbit structure.

---

## Next Steps

- For the formal axioms: → `QA_AXIOMS.md`
- For the state space and failure types: → `QA_STATE_SPACE.md`
- For the most intuitive applied example: → `../03_applied_domains/CYMATICS_EXAMPLE.md`
