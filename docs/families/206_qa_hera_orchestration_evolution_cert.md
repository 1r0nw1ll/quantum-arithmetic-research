# Family [206] QA_HERA_ORCHESTRATION_EVOLUTION_CERT.v1

## One-line summary

HERA's multi-agent orchestration evolution (Li & Ramakrishnan, Virginia Tech 2026) independently reproduces QA orbit dynamics: the dual-axes prompt evolution maps onto Bateson Learning Levels [191] (op=Level I, bp=Level II), the four-phase topology evolution matches orbit descent (Satellite→Cosmos→Satellite, never Singularity), and the entropy plateau at intermediate level is exactly Satellite convergence — structured flexibility between maximum entropy (Cosmos) and zero entropy (Singularity).

## Background

**HERA** (Hierarchical Evolution for RAG Agents) is a framework for multi-agent Retrieval-Augmented Generation that evolves both the global orchestration (which agents to activate and how they interact) and individual agent behavior (role-specific prompt refinement). Published results show 38.69% average improvement over SOTA on six benchmarks.

The key empirical finding: the system self-organizes into simpler, sparser structures rather than growing more complex — and single-agent (Singularity) baselines are beaten by 38.69%.

## Mathematical content

### ROPE: Dual-axes prompt evolution = Bateson Learning Levels

HERA decomposes prompt updates into:
- **Operational rules** (Δρ^op): short-term corrective behaviors
- **Behavioral principles** (Δρ^bp): long-term strategies

| HERA Mechanism | Bateson Level | QA Formalization |
|---|---|---|
| Fixed prompt execution | Level 0 | Identity operator, fixed point |
| Operational rule update | Level I | Within-orbit adaptation |
| Behavioral principle update | Level II-a | Orbit change, family preserved |
| Agent replacement (topology) | Level II-b | Family change |
| LLM backbone change (not in HERA) | Level III | Modulus change |

The paper's credit assignment ("agent is underperforming if orchestrator identifies it as primary contributor to failure") is double-bind detection — an agent trapped at Level 0 cannot escape without Level I+ operators.

### PHASE: Four-phase topology evolution = orbit descent

| Phase | Name | QA Orbit | Description |
|---|---|---|---|
| 1 | Initial | Satellite-like | Narrow, shallow, low agent count |
| 2 | Exploration | Cosmos-like | Recruits agents, increases diameter/cycles |
| 3 | Refinement | Descent | Prunes redundant agents, efficiency rises |
| 4 | Optimization | Satellite-like | Compact chains, minimal nodes, high utility |

Critical: the system does NOT converge to Singularity (single agent). The "Direct" single-agent baseline is beaten by 38.69%. The system finds the **minimal orbit that solves the task** — Satellite, not Singularity.

### ENTROPY: Intermediate plateau = Satellite convergence

"After an initial rise, entropy stabilizes at an intermediate level" — not maximum entropy (Cosmos, uniform exploration) and not zero (Singularity, fixed behavior). This is Satellite: structured flexibility that "prevents premature convergence to suboptimal strategies while allowing the integration of novel cooperation pathways."

### SPARSE: Orbit discovery

"Sparse exploration leads to compact, high-performing networks." QA's 81 mod-9 pairs partition into 3 orbits — the orbit structure IS sparse. HERA discovers that most agent topologies are redundant, converging to invariant subsets (= orbit discovery).

### NT: Theorem NT compliance

The framework naturally separates:
- **E-step** (topology sampling): discrete, QA layer
- **M-step** (semantic insight extraction): continuous/linguistic, observer layer

Frozen LLMs ensure the observer layer doesn't corrupt the generative layer. The experience library stores (query_chars, NL_insight, utility_float) — the NL and float components are observer projections per T2.

## Qualified claims (honest)

- **Multi-agent → RNS/CRT**: WEAK. Architecture is sequential pipeline, not parallel-modular.
- **Agent graph → QA coupling**: MODERATE. Transition matrix resembles resonance matrix but has no algebraic symmetry.

## Checks

| ID | Description |
|----|-------------|
| HOE_1 | schema_version matches |
| HOE_ROPE | Dual-axes present with Bateson level mapping |
| HOE_PHASE | 4 phases present; final NOT Singularity |
| HOE_ENT | Entropy plateau at intermediate level |
| HOE_PERF | Improvement > 0% over baselines |
| HOE_W | >= 4 witnesses |
| HOE_F | Falsifier: Singularity convergence would break the analogy |

## Source grounding

- **Li & Ramakrishnan** (2026): arXiv:2604.00901. HERA framework.
- **QA cert [191]**: Bateson Learning Levels — strict invariant filtration.
- **QA cert [205]**: Grid Cell RNS — modular arithmetic in neural systems.

## Connection to other families

- **[191] Bateson Learning Levels**: RoPE = L1/L2a filtration
- **[205] Grid Cell RNS**: Modular encoding in multi-agent coordination
- **[199] Grokking Eigenvalue**: Phase transition from memorization to generalization
- **[122] Empirical Observation**: Experience library as structured observation

## Fixture files

- `fixtures/hoe_pass_core.json` — 6 claims + 2 qualified + witnesses
- `fixtures/hoe_pass_numerical.json` — performance improvement, phase count, entropy behavior
- `fixtures/hoe_fail_singularity.json` — Singularity convergence rejected (beaten by 38.69%)
