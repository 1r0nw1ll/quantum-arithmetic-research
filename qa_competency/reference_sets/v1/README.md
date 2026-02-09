# QA Competency Reference Pack v1

Curated exemplar bundles across bio/AI/hybrid domains for the
QA Competency Detection framework (family [26]).

> **Disclaimer:** These are reference exemplars illustrating framework
> capabilities. They are not claims about specific wet-lab experiments,
> deployed AI systems, or clinical protocols.

## Bundle inventory

### Bio domain (`bio/`)

| Bundle | Agency | Plasticity | Entropy | Edge case |
|--------|--------|------------|---------|-----------|
| `planarian_regen` | 0.71 | 0.85 | 1.34 | 2 certs: regeneration + memory retention through decapitation |
| `xenopus_patterning` | 0.88 | 0.72 | 1.58 | High agency over narrow goal space (GD=0.04) |
| `organoid_self_repair` | 0.62 | 0.55 | 1.94 | 7 generators, high entropy distributed decision-making |

### AI domain (`ai/`)

| Bundle | Agency | Plasticity | Entropy | Edge case |
|--------|--------|------------|---------|-----------|
| `tool_agent_debugger` | 0.64 | 0.30 | 1.34 | Low plasticity: competent but brittle |
| `retrieval_agent_arag` | 0.82 | 0.68 | 1.38 | Citation-fidelity invariant |
| `multi_agent_coordination` | 0.75 | 0.70 | 1.94 | 7 generators, 3 components, distributed agency |

### Hybrid domain (`hybrid/`)

| Bundle | Agency | Plasticity | Entropy | Edge case |
|--------|--------|------------|---------|-----------|
| `bioelectric_controller_with_llm` | 0.68 | 0.78 | 1.38 | Two-substrate (bio + digital) |
| `lab_robot_closed_loop` | 0.55 | 0.35 | 1.21 | Low entropy, mostly deterministic protocol |
| `human_in_the_loop_protocol` | 0.91 | 0.82 | 1.55 | Highest agency, constrained by human response time |

## Entropy constraint

Control entropy = -sum p ln p, max = ln(N) for N generators.

- 4 generators: max 1.386
- 5 generators: max 1.609
- 7 generators: max 1.946

Bundles targeting entropy >= 1.8 use >= 7 generators (`organoid_self_repair`,
`multi_agent_coordination`).

## Validation

```bash
# Validate all reference sets
python qa_competency/qa_competency_validator.py --reference-sets

# Rehash manifests (includes reference sets)
python qa_competency/qa_competency_validator.py --rehash
```

## Schema

Each `.bundle.json` conforms to `QA_COMPETENCY_CERT_BUNDLE.v1.schema.json`.
Metrics are deterministically recomputed from `metric_inputs` during validation.
