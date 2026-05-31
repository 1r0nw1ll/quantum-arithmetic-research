<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary source stance and non-claims are enforced by validator and mapping_protocol_ref.json -->

# [284] QA Wave Co-processor Boundary Cert

**Cert family**: `qa_wave_coprocessor_boundary_cert_v1`

## Purpose

This family certifies the boundary discipline for using a physical wave
interference layer as a QA observer co-processor. It is the first registered
foothold for the wave-parallel computation roadmap: exact QA phase packets may
be encoded into continuous wave parameters, continuous amplitudes may interfere
inside the declared co-processor boundary, and readout must return to finite
declared bins.

The cert does not claim optical speedup, neural mechanism, Maxwell physics,
reservoir universality, physical implementation, unlimited parallelism, or
computational complexity bypass.

## Schema Fields

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_WAVE_COPROCESSOR_BOUNDARY_CERT.v1` |
| `cert_slug` | Must be `qa_wave_coprocessor_boundary_cert_v1` |
| `family_id` | Must be `284` |
| `source_attribution` | Must mention Iverson, Synchronous Harmonics, and boundary |
| `non_claims` | Explicit list of rejected high-risk claims |
| `claim_policy` | Boolean overclaim switches, all false in v1 |
| `coprocessor_boundary` | Declares the physical-wave observer co-processor boundary |
| `pipeline_stages` | Stages and state types; continuous wave state must be boundary-local |
| `wave_packets` | Exact modular phase packets and rational wave parameters |
| `interference_witnesses` | Declared pairwise phase deltas and expected relations |
| `readout_policy` | Declared finite-bin readout and ambiguity rejection |

## Validator Checks

| Gate | Check |
|---|---|
| `WCB_0` | Schema version, slug, and family id |
| `WCB_1` | Packet phase equals `phase_index / modulus`; amplitude/frequency rational checks |
| `WCB_2` | Co-processor boundary kind, role, input/output direction, and not-QA-core flag |
| `WCB_3` | Continuous wave state appears only inside `coprocessor_boundary` |
| `WCB_4` | Phase-delta relation recomputed exactly as support/oppose/neutral |
| `WCB_5` | Readout policy has declared bins and rejects ambiguity |
| `WCB_6` | Source attribution and `mapping_protocol_ref.json` are well-formed |
| `WCB_7` | Overclaims are rejected |

## Fixtures

| Fixture | Verdict | What it tests |
|---|---|---|
| `pass_wcb_three_packet_interference.json` | PASS | Three packets with exact support, oppose, and neutral phase witnesses |
| `fail_wcb_continuous_state_as_core.json` | FAIL | Rejects treating continuous wave state as QA core state |
| `fail_wcb_wrong_phase_relation.json` | FAIL | Rejects a wrong support/oppose relation |
| `fail_wcb_overclaims_speedup.json` | FAIL | Rejects unlimited-parallelism and physical-speedup overclaims |
| `fail_wcb_ambiguous_readout.json` | FAIL | Rejects readout policy that allows ambiguity |

## Relationships

- Builds on [147] Synchronous Harmonics as the QA-side motivation for support,
  oppose, and synchronization language.
- Complements [257] Integer-State Pipeline by allowing continuous state only
  in a declared observer co-processor boundary, not inside QA core state.
- Anchors `docs/specs/QA_WAVE_PARALLEL_COMPUTATION_ROADMAP.md`.
- Provides the boundary prerequisite for future wave encoding, interference
  readout, wave-parallel benchmark, and reservoir observer certs.

