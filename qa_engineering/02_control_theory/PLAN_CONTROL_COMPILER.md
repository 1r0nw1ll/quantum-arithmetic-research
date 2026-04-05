# The Plan-Control Compiler

The plan-control compiler is the formal bridge between **planning** (finding a path) and **control** (executing it). This is cert family [106] `QA_PLAN_CONTROL_COMPILER_CERT.v1`.

---

## The Core Claim

> Whenever a QA planner cert emits a plan over a lawful generator algebra and a QA control cert instantiates that plan, a certifiable compilation edge exists between them: same generator sequence + same target invariant + same orbit-family witness + hash-pinned on both ends.

This is the abstraction upward from cymatics: the search → plan → control → replay chain is a **general QA law**, not a cymatics-specific artifact.

---

## What the Compiler Certifies

The compiler cert verifies nine claims (CC1–CC9):

| Check | What is verified | Failure type |
|-------|-----------------|-------------|
| CC1 | Source planner cert found | `SOURCE_CERT_MISSING` |
| CC2 | Source cert hash matches declared hash | `COMPILATION_HASH_MISMATCH` |
| CC3 | Target control cert found | `TARGET_CERT_MISSING` |
| CC4 | Target cert hash matches declared hash | `COMPILATION_HASH_MISMATCH` |
| CC5 | initial_pattern_class consistent across planner, control, and compiler claims | `TARGET_INVARIANT_MISMATCH` |
| CC6 | target_pattern_class consistent across all three | `TARGET_INVARIANT_MISMATCH` |
| CC7 | path_length_k consistent across all three | `PATH_LENGTH_MISMATCH` |
| CC8 | move_sequence consistent across all three | `GENERATOR_SEQUENCE_MISMATCH` |
| CC9 | final_orbit_family consistent across all three | `REPLAY_RESULT_MISMATCH` |

If all nine pass: the plan and controller are formally equivalent. The compilation edge is certified and hash-pinned.

---

## The Four-Tier Cert Stack

A complete QA control design has four tiers, each building on the previous:

```
Tier 1: Mode/Recognition Cert
  └── "This state corresponds to a valid QA (b,e) pair"
      Example: Chladni mode cert — plate at (m,n) resonance maps to QA state

Tier 2: Reachability Cert
  └── "Legal transitions exist in this domain connecting these states"
      Example: Faraday cert — flat/stripes/hexagons form a connected pattern graph

Tier 3: Control Cert
  └── "This generator sequence drives the system from initial to target"
      Example: apply increase_amplitude then set_frequency → reaches hexagons

Tier 4: Planner Cert
  └── "BFS finds this as a minimal path and can prove no shorter path exists"
      Plus: minimality_witness proving no shorter path was possible
```

The **compiler cert** ([106]) sits above Tiers 3 and 4, verifying they agree.

---

## Minimality Witness

A complete planner cert includes a **minimality_witness** proving the plan is as short as possible:

```json
{
  "minimality_witness": {
    "proved_no_path_shorter_than": 2,
    "excluded_shorter_lengths": [1],
    "frontier_sizes": {"depth_1": 2, "depth_2": 1}
  }
}
```

This says: BFS explored depth 1 (found 2 new states, none was the target) and depth 2 (found the target). Therefore k=2 is minimal.

Without a minimality witness, a plan of length k=5 is not provably optimal — it might just be what the planner happened to find first.

---

## The Cymatics Instantiation (Reference Example)

The first certified compiler cert links:

- **Source** (planner): `planner_cert_pass_replay_hexagon.v1`
  - BFS finds `flat → stripes → hexagons` in k=2
  - Moves: `[increase_amplitude, set_frequency]`
  - Initial: flat (singularity), Target: hexagons (cosmos)

- **Target** (control): `control_cert_pass_hexagon.v1`
  - Executes `flat → stripes → hexagons`
  - Final orbit: cosmos
  - All transitions legal

- **Compiler claims**:
  - initial = flat ✓ (all three agree)
  - target = hexagons ✓ (all three agree)
  - k = 2 ✓ (all three agree)
  - moves = [increase_amplitude, set_frequency] ✓ (all three agree)
  - orbit = cosmos ✓ (all three agree)
  - **PASS**

The FAIL fixture demonstrates `GENERATOR_SEQUENCE_MISMATCH`: both referenced certs agree on moves=[increase_amplitude, set_frequency] but the compilation claim declares the wrong second move.

---

## Adding a New Domain

To instantiate the compiler in a new domain:

1. Write a recognition/mode cert for your domain (Tier 1)
2. Write a reachability cert mapping your domain states to QA orbits (Tier 2)
3. Write a control cert certifying a specific generator sequence (Tier 3)
4. Write a planner cert with BFS evidence (Tier 4)
5. Register your domain in `DOMAIN_TO_FIXTURES` in `qa_plan_control_compiler_validate.py`
6. Write a compiler cert linking your Tier 4 to Tier 3 cert

Your domain cert will then be part of the certified QA ecosystem and will pass `python qa_meta_validator.py`.

---

## Running the Compiler

```bash
cd qa_alphageometry_ptolemy/qa_plan_control_compiler

# Self-test (all fixtures)
python qa_plan_control_compiler_validate.py --self-test

# Demo mode (human-readable)
python qa_plan_control_compiler_validate.py --demo

# Single cert
python qa_plan_control_compiler_validate.py \
  --cert fixtures/compiler_cert_pass_cymatics_hexagon.json

# Failure example
python qa_plan_control_compiler_validate.py \
  --cert fixtures/compiler_cert_fail_sequence_mismatch.json
```

---

## Source References

- Cert family [106]: `qa_alphageometry_ptolemy/qa_plan_control_compiler/`
- Schema: `qa_alphageometry_ptolemy/qa_plan_control_compiler/schemas/qa_plan_control_compiler_cert.schema.json`
- Validator: `qa_alphageometry_ptolemy/qa_plan_control_compiler/qa_plan_control_compiler_validate.py`
- Fixtures: `qa_alphageometry_ptolemy/qa_plan_control_compiler/fixtures/`
- Cymatics 4-tier: cert family [105], `qa_alphageometry_ptolemy/qa_cymatics/`
