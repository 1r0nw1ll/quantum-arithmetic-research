# Family [80] — QA Energy Cert v1.1 (CAPS_TR cognitive domain)

**Machine tract:** `qa_energy_cert_v1_1/`
**Schema version:** `QA_ENERGY_CERT.v1.1`
**Domain:** `CAPS_TR` (primary) · `CAPS_BE` supported

---

## What it certifies

Generator-relative **energy** in a discrete state space:

> **Energy(s)** = minimal BFS path length from `reference_state` to `s`
> under the declared `generator_set`.

This is structurally identical to the QA Time Axiom (minimal generator
distance) and provides a quantitative foundation for the Power metric
**ΔE(g, s) = E(g(s)) − E(s)**.

---

## CAPS_TR domain

States **(T, R)** where:
- **T** ∈ {0..N} — Threat level (0 = quiescent, N = maximum)
- **R** ∈ {0..N} — Regulation level (0 = dysregulated, N = fully regulated)

### Generators

| Name | Rule | Family | Interpretation |
|---|---|---|---|
| `fear_up` | (T,R)→(T+1,R) | `fear` | Threat escalates |
| `fear_down` | (T,R)→(T-1,R) | `fear` | Threat de-escalates |
| `fear_lock` | (T,R)→(T,R-1) | `fear` | Regulation depleted by fear |
| `love_soothe` | (T,R)→(T-1,R+1) | `love` | Threat down + regulation up |
| `love_support` | (T,R)→(T,R+1) | `love` | Pure regulation increase |
| `love_reframe` | (T,R)→(T-1,R) | `love` | Threat reduced without changing regulation |

`family_tag` is a **generator operator-class label**, not a person label.
All cert claims are structural (state-transition topology), not narrative.

---

## Gate summary

| Gate | What it checks |
|---|---|
| Gate 1 | JSON schema validity · domain in {CAPS_BE, CAPS_TR} · reference_state in domain · `interaction_horizon` hard-locked to 2 |
| Gate 2 | Recompute `energy_map` via BFS from `reference_state` |
| Gate 3 | Recompute `return_energy_map` via reverse BFS |
| Gate 4 | Energy monotonicity: 1-Lipschitz along edges · tight predecessor · E(ref)=0 |
| Gate 5 | `return_in_k_tests` match computed reverse distances |
| Gate 6 | SCC recompute · `power_stats` · `family_power_stats` · `family_interaction_stats` · optional `power_tests` qualitative assertions |

---

## Family statistics

**`power_stats`** — per-generator ΔE over reachable edges.

**`family_power_stats`** — aggregate ΔE by `family_tag` (fear / love).

**`family_interaction_stats`** — 2-step paths **s →(g₁)→ t →(g₂)→ u**,
ΔE₂ = E(u) − E(s), keyed by ordered `(from_family, to_family)` pair.
`interaction_horizon` is declared in the cert (default 2); v1.1 validator
hard-locks it to 2.

### Theoretical meaning

| Observation | Interpretation |
|---|---|
| fear family: mean ΔE > 0 | Fear generators increase distance from equilibrium |
| love family: mean ΔE < 0 | Love generators decrease distance (recovery operators) |
| love→love interaction ΔE < fear→fear | Sustained love application is more efficient than fear cycling |
| love→fear cross-pair ΔE | Cost of escalation after a regulatory gain |

---

## Fixtures

| File | Type | Notes |
|---|---|---|
| `PASS_FEAR.json` | PASS | N=2, ref=(0,2), generators: fear only |
| `PASS_LOVE.json` | PASS | N=2, ref=(2,0), generators: love only |
| `PASS_MIXED.json` | PASS | N=2, ref=(0,2), all 6 generators; 1 large SCC |
| `FAIL_POWER.json` | FAIL | `POWER_TESTS_VIOLATION` — impossible mean_sign constraint |
| `FAIL_INTERACTION.json` | FAIL | `POWER_STATS_MISMATCH` — wrong family_interaction_stats entry |
| `FAIL_HORIZON.json` | FAIL | `DOMAIN_INVALID` — interaction_horizon=3 rejected |

---

## Research context

This cert originated from a QA analysis of **generator-dependent reachability
in cognitive/behavioral science** (see `Documents/QA_cognative_behavioral_science.odt`).
The key insight: similar functional patterns in cognitive systems arise not from
shared lineage but from **shared generator topology** — if the same environmental
generators act on a state space, partial attractor convergence follows regardless
of initial state differences.

The CAPS_TR domain formalizes this: `reference_state` = equilibrium/recovery
baseline; energy = effort required to reach a state FROM baseline under the
current generator set; ΔE = generator power.

---

## Self-test

```bash
python3 qa_energy_cert_v1_1/validator.py --self-test
```

Expected: `RESULT: PASS` (6/6 fixtures).
