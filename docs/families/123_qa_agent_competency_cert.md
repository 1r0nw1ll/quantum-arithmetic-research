# Family [123] — QA Agent Competency Cert

**Schema:** `QA_AGENT_COMPETENCY_CERT.v1`
**Validator:** `qa_alphageometry_ptolemy/qa_agent_competency_cert/qa_agent_competency_cert_validate.py`
**Parent cert:** `QA_CORE_SPEC.v1`

## Purpose

Formalizes the **Levin morphogenetic agent architecture** for QA Lab.

An agent competency profile certifies:
- What the agent achieves (goal)
- How far it looks (cognitive horizon)
- Whether it converges (convergence type)
- Which QA orbit family characterizes it (orbit signature)
- What kind of developmental cell it is (Levin cell type)
- When it should fail gracefully (failure modes)
- How it composes with other agents (composition rules)
- When to dedifferentiate back to stem state
- When to recommit to a different competency

## The Levin Cell Mapping

| QA Orbit | Levin Cell Type | Meaning |
|---|---|---|
| singularity | stem | totipotent, no specialization, infinite potential |
| satellite | progenitor | cycling, differentiating, loops until committed |
| cosmos | differentiated | specialized, productive, convergent |
| mixed | progenitor | context-dependent, routes to cosmos or satellite |

## Consistency Rule (V8 — CELL_ORBIT_MISMATCH)

The cell type and orbit signature must be consistent:
- `differentiated` requires `cosmos`
- `progenitor` requires `satellite` or `mixed`
- `stem` requires `singularity`

This prevents invalid profiles: a "differentiated" agent cannot be declared to run in a looping satellite orbit.

## Organ Composition Invariant

Every multi-agent organ must have at least one `differentiated` agent as its **spine** (cosmos orbit). Satellite/progenitor agents must be paired with a cosmos guardian to guarantee convergence.

## Validated Examples

| Agent | orbit_signature | levin_cell_type | convergence |
|---|---|---|---|
| merge_sort | cosmos | differentiated | guaranteed |
| gradient_descent | mixed | progenitor | conditional |

## Fail Types

| Code | Meaning |
|---|---|
| `SCHEMA_VERSION_MISMATCH` | schema_version field incorrect |
| `UNKNOWN_COGNITIVE_HORIZON` | not one of: local/bounded/global/adaptive |
| `UNKNOWN_CONVERGENCE_TYPE` | not one of: guaranteed/probabilistic/conditional/none |
| `UNKNOWN_ORBIT_SIGNATURE` | not one of: cosmos/satellite/singularity/mixed |
| `UNKNOWN_LEVIN_CELL_TYPE` | not one of: stem/progenitor/differentiated |
| `EMPTY_FAILURE_MODES` | at least one structural failure condition required |
| `EMPTY_DEDIFFERENTIATION_COND` | at least one dedifferentiation condition required |
| `CELL_ORBIT_MISMATCH` | orbit_signature inconsistent with levin_cell_type |
| `GOAL_TOO_SHORT` | goal string < 10 characters |
| `EMPTY_COMPOSITION_RULES` | at least one composition rule required |

## Lineage

This family extends [107] QA_CORE_SPEC.v1 and [121] QA_Engineering_Core_Cert
into the agent architecture domain. It is the machine-checkable spec that
turns the Levin algorithm competency study (`qa_algorithm_competency.py`)
from an observational analysis into a certifiable formal claim.
