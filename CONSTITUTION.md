# QA Project Constitution

**Version:** 1.0 | **2026-03-25**

---

## Mission

QA is a formal specification language for reality-constrained computation — a state-transition
algebra over Z[φ]/mZ[φ] whose orbit structure is simultaneously a mathematical object,
a convergence metric, and a coherence detector. Every artifact in this project either
advances that claim or does not belong here.

---

## The Positioning

| System | Specifies | Advantage |
|---|---|---|
| DeepMind (AlphaGeometry, etc.) | What should happen (search / inference) | Powerful search |
| Anthropic (Constitutional AI) | What can happen within constraints (behavior / safety) | Behavioral guarantees |
| **QA** | **The geometry of possibility itself** — what IS reachable, what CANNOT happen, WHY it fails | **First-class failure algebra + reachability semantics** |

QA sits in the formal specification lineage:
**Hoare (1969) → Dijkstra → Goguen → Pnueli / Lamport → Lawvere → QA (2025)**

---

## The Four Research Tracks

| Track | Scope |
|---|---|
| **A — Algebraic Foundations** | Z[φ]/mZ[φ] orbit structure, inert primes, Pythagorean families, modular dynamics |
| **B — Specification Infrastructure** | QA_CORE_SPEC [107], cert ecosystem [18]–[122], meta-validator, QA-ORBIT benchmark |
| **C — Convergence Theory** | Finite-Orbit Descent Theorem, Łojasiewicz orbit descent, κ as universal convergence metric |
| **D — Coherence Detection** | Audio, EEG, finance, seismic — empirical projections of the formal structure |

---

## Five Non-Negotiables

1. **Every empirical claim has pre-declared success criteria.** Post-hoc criteria are not accepted.
2. **Negative results are first-class.** A CONTRADICTS cert that PASSes is as valuable as a CONSISTENT one.
3. **The meta-validator is always green.** No cert ships without PASS + FAIL fixtures and a human tract.
4. **Open Brain captures happen at the moment of result.** Never batched at session end.
5. **Every new artifact maps to a track.** If it doesn't map, it goes to archive/.

---

## Gate Hierarchy

| Gate | Check |
|---|---|
| 0 | Mapping protocol present (`mapping_protocol.json` or `mapping_protocol_ref.json`) |
| 1 | Schema valid |
| 2 | Recompute deterministic |
| 3 | Invariants hold |
| 4 | `invariant_diff` check |
| 5 | Merkle / hash verification |

---

## Current Health (2026-03-25)

- Meta-validator: **128/128 PASS**
- Papers arXiv-ready: Pythagorean Families, Unified Curvature
- Papers complete draft: QA-ORBIT, Agency Decomp, Locality Dominance, Phase 2 EEG, Modular Dynamics (outline + proofs)
- Empirical: r(κ, loss) = −0.845 | EEG HI 2.0 +6.7pp | Crypto singularity p=0.0025

**Full strategic audit:** `docs/specs/VISION.md`
**Role specifications:** `AGENTS.md`
**Project spec:** `docs/specs/PROJECT_SPEC.md`
