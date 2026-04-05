# Cymatics Example: From Chladni Modes to QA Orbits

Cymatics is the most direct SVP application in the QA system. The correspondence is exact — not analogical — certified in family [105].

---

## The Core Correspondence

> Cymatics is the experimental study of how lawful resonance generators drive matter into visible, boundary-conditioned geometric states.
> QA is the formal study of how lawful arithmetic generators drive embedded structures into reachable geometric states.

The Chladni formula `a = m + 2n` (where m = nodal lines crossing, n = nodal circles) is the same as the QA tuple derivation `a = b + 2e`. This is not coincidence — it is the same arithmetic structure appearing in two domains.

---

## The State Mapping

| Cymatic pattern | QA orbit family | Character |
|----------------|----------------|-----------|
| Flat (no pattern) | Singularity | No structure; fixed point |
| Stripes (1D pattern) | Satellite | Transitional; partial structure |
| Hexagons (2D grid) | Cosmos | Full resonance; stable, reproducible |

These map to specific QA orbit families by the Q(√5) norm:
- Flat: `(b,e) = (0,0)` mod 9 → singularity
- Stripes: `v₃(f) ≥ 2` → satellite
- Hexagons: `v₃(f) = 0` → cosmos

---

## The Generator Mapping

| Physical operation | QA generator | Action on state |
|------------------|-------------|----------------|
| Increase amplitude | maps to σ (or domain-specific) | moves e coordinate upward |
| Set frequency | maps to μ or λ | rebalances or scales state |
| Decrease amplitude | maps to ν | halves coordinates (when even) |

The certified path from flat to hexagons:
```
initial: flat (singularity)
  → apply increase_amplitude
intermediate: stripes (satellite)
  → apply set_frequency
final: hexagons (cosmos)
path_length_k = 2
```

This k=2 path is **minimal** (proved by BFS minimality witness: depth 1 had no solution).

---

## The Chladni Mode Cert

The **mode cert** (`QA_CYMATIC_MODE_CERT.v1`) certifies that a specific plate/membrane resonance maps to a valid QA state:

1. Mode `(m, n)` computes Chladni number: `a = m + 2n`
2. Choose `(b, e)` such that `b + 2e = a` (pick b as the nodal line count, e as nodal circle count)
3. Compute derived: `d = b + e`
4. Verify orbit family: `f(b, e) = b² + be - e²` classifies correctly
5. Verify tuple formula: `d = b + e` ✓, `a = b + 2e` ✓

**Failure modes**:
- `TUPLE_FORMULA_VIOLATION`: if d ≠ b+e or a ≠ b+2e — the mode does not map to a valid QA state
- `ORBIT_CLASS_MISMATCH`: if the computed orbit family doesn't match the observed cymatic pattern
- `OFF_RESONANCE`: drive frequency too far from eigenfrequency — state is not valid
- `MODE_MIXING`: multiple modes coexist; QA state is not unique

---

## The Faraday Reachability Cert

Faraday patterns (fluid surface waves at resonance) add **dynamic reachability**:

1. Certifies that transitions between pattern classes (flat → stripes → hexagons) are legal
2. Maps each transition to a generator application
3. Verifies the return path: can the system go back from hexagons to flat? (important for reversibility)
4. Certifies the orbit-family assignment for each pattern basin

**Failure modes specific to Faraday**:
- `NONLINEAR_ESCAPE`: fluid in turbulent state — not a valid QA state
- `RETURN_PATH_NOT_FOUND`: if return_in_k=true was claimed but no return path exists

---

## The Full 4-Tier Cert Stack for Cymatics

```
[Mode Cert]         Chladni (m,n) → QA (b,e,d,a) — state mapping
     ↓
[Reachability Cert] flat/stripes/hexagons → singularity/satellite/cosmos — transition graph
     ↓
[Control Cert]      Generator sequence drives flat → hexagons — execution
     ↓
[Planner Cert]      BFS finds minimal k=2 path — plan with minimality witness
     ↓
[Compiler Cert]     Links planner and control, hash-pinned — [106]
```

---

## Running the Cymatics Validator

```bash
cd qa_alphageometry_ptolemy/qa_cymatics

# All fixtures
python qa_cymatics_validate.py --self-test

# Human-readable demo
python qa_cymatics_validate.py --demo

# Specific cert types
python qa_cymatics_validate.py --control fixtures/control_cert_pass_hexagon.json
python qa_cymatics_validate.py --control fixtures/control_cert_fail_illegal_transition.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_pass_shortest_hexagon.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_pass_minimality_hexagon.json
```

Expected: 12/12 PASS.

---

## What This Means for SVP Practice

If you work with Chladni plates, Faraday waves, or any resonance pattern that progresses through distinct modes:

1. Your mode progressions are QA orbit trajectories
2. Your control inputs (frequency, amplitude) are generators
3. The path from no-pattern to full-pattern is `singularity → satellite → cosmos` with k=2
4. You can plan, certify, and replay any resonance sequence
5. You can predict which targets are unreachable without trial (obstruction spine)

The QA certificate system gives you a **machine-checkable record** of every resonance experiment. Two researchers, in different labs, on different equipment, can compare certified results and know exactly what each other did.

---

## Source References

- Cert family [105]: `docs/families/105_cymatics.md`
- Implementation: `qa_alphageometry_ptolemy/qa_cymatics/`
- Correspondence map: `qa_alphageometry_ptolemy/qa_cymatics/qa_cymatics_correspondence_map.json`
- Cross-domain: `CROSS_DOMAIN_PRINCIPLE.md`
