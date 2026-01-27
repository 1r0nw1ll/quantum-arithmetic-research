# QA Certificate Tetrad Index

**Theorem**: `Capability = Reachability(S, G, I)` and `Intelligence = K = log10(tau_blind / tau_agent)`

This holds whether G operates on bits, populations, electromagnetic fields, or biological cells.

---

## The Four Certificates

| # | Certificate | Direction | Domain | Source Papers |
|---|-------------|-----------|--------|---------------|
| 1 | `GENERATOR_INJECTION` | Reach EXPANDS | Agents, proofs | LLM-in-Sandbox, Axiom Putnam 2025 |
| 2 | `DIVERSITY_COLLAPSE_OBSTRUCTION` | Reach CONTRACTS | Search, RL | Execution-Grounded (Stanford 2026) |
| 3 | `FIELD_COMPUTATION_CERT` | Reach REALIZED BY PHYSICS | RF, photonic, analog | WISE (Science Advances) |
| 4 | `BEYOND_NEURONS_INTELLIGENCE_CERT` | Intelligence SUBSTRATE-NEUTRAL | Biology, silicon, collective | Levin & Chis-Ciure 2024 |

---

## Shared Structure

Every certificate contains:

```
certificate_id          Deterministic ID with prefix + UTC timestamp
certificate_type        One of the four registered types
timestamp               ISO 8601 UTC
barrier                 Obstruction object: type, source, target, required mechanism
before_witness          Proof of unreachability under baseline configuration
after_witness           Proof of reachability under extended configuration
result                  BARRIER_CROSSED | CONSTRAINT_EDITED | GOAL_DECOUPLED | ...
```

All use `qa_cert_core.py` for:
- Exact scalars (`int | Fraction`, no floats)
- Deterministic JSON (`canonical_json` with sorted keys)
- Certificate hashing (`SHA-256`, 16-char short hash)
- Failure-complete validation (`ValidationResult`)

---

## Tetrad Table

```
              Gen. Injection     Div. Collapse      Field Computation   Beyond Neurons
              ---------------    ---------------    -----------------   ---------------
Direction     Reach EXPANDS      Reach CONTRACTS    Reach REALIZED      Intelligence =
                                                                        SUBSTRATE-NEUTRAL
Mechanism     Add generators     Violate invariant  Physics = generators Constraint editing,
                                                                        goal decoupling,
                                                                        horizon expansion
Barrier       Crossed            Erected            Crossed (physical)  Crossed / Erected
Invariants    Preserved          Violated (I_div)   Preserved (I_field) Edited (C)
Witness       ReachabilityW      CollapseWitness    FieldReachabilityW  IntelligenceWitness
Evidence      Constructive path  Mean/best/div      Trajectory + error  K = log10(blind/agent)
Fix           (injection)        Add diversity gen  Add sync/calibrate  Edit constraints /
                                                                        realign goals
Substrate     Any (agents)       Any (search)       Physical fields     ANY (substrate-neutral)
Scale         Single             Single             Single              Multi-scale (nested)
```

---

## Canonical Examples

### 1. Generator Injection: LLM-in-Sandbox

```
Before:   {text_generation}
After:    {text_generation, execute_bash, file_read, file_write}
Injected: {execute_bash, file_read, file_write}
Barrier:  CONTEXT_LENGTH (100k tokens > 32k window)
Fix:      Store context to files, grep relevant sections
Result:   BARRIER_CROSSED
```

### 2. Diversity Collapse: RL vs Evolution

```
Strategy:   RL policy gradient (10 steps, 64 population)
Signature:  mean_improved=True, best_improved=False, diversity_collapsed=True
Collapse:   Step 6 (diversity < 30 for 3 consecutive steps)
Plateau:    Steps 0-3 (max_reward flat within epsilon=2)
Missing:    entropy_bonus_regularizer
Comparison: Evolutionary search preserved diversity (min=73 > threshold=30)
Result:     COLLAPSE_DETECTED
```

### 3. Field Computation: WISE Desync

```
Task:       4x4 matrix-vector multiply via RF superposition
Domain:     rf_inphysics
Barrier:    DESYNC (no pilot synchronization)
Injected:   sync_align (pilot-based timing/phase alignment)
Trajectory: encode -> phase_shift -> gain -> propagate -> mix -> sync_align -> measure
Error:      3/100 (tolerance: 1/20)
Invariants: power_budget, bandwidth_limit, sync_lock, linearity
Result:     BARRIER_CROSSED
```

### 4a. Beyond Neurons: Planaria Regeneration (Constraint Editing)

```
Substrate:  biological_non_neural
Scale:      organism
Operators:  {cell_division, cell_migration, apoptosis, differentiation, bioelectric_signaling}
Mechanism:  Constraint editing (gap junction manipulation inverts A-P polarity)
Barrier:    CONSTRAINT_LOCK (two-headed morphology unreachable under normal polarity)
Edited:     anterior_posterior_polarity constraint
K:          9 (regeneration is 10^9x faster than blind morphospace search)
Result:     CONSTRAINT_EDITED
```

### 4b. Beyond Neurons: Cancer Goal Decoupling

```
Substrate:  biological_non_neural
Scale:      cellular (intelligent at its own scale)
Operators:  {proliferate, evade_apoptosis, angiogenesis_signal, immune_evasion}
Mechanism:  Goal decoupling (E_cell diverges from E_organism)
Barrier:    GOAL_DECOUPLING (organism homeostasis unreachable when cells defect)
K:          4 at cellular scale (cancer IS intelligent -- problem is misalignment)
Architecture: 2-level (cellular -> organism) with decoupled evaluation
Result:     GOAL_DECOUPLED
```

### 4c. Beyond Neurons: Non-Neural AI (Substrate Neutrality)

```
Substrate:  silicon_digital
Scale:      artificial
Operators:  {text_generation, code_execution, retrieval, chain_of_thought}
Mechanism:  Operator application (same generators as ROI #1, different substrate)
K:          17 on math competition problems (10^20 blind / 10^3 agent)
Link:       Directly connects to Generator Injection certificate
Result:     SUBSTRATE_NEUTRAL_CONFIRMED
```

---

## New Mechanisms (Direction 4)

```
Constraint Editing:
  Same operators O, modified constraints C -> new states reachable
  Dual to Generator Injection (change O vs change C)
  Example: Planaria bioelectric manipulation

Goal Decoupling:
  Component evaluation E_i diverges from collective E
  Generalizes Diversity Collapse (RL E decouples from exploration E)
  Example: Cancer cells maximize proliferation, ignore homeostasis

Horizon Expansion:
  Increase planning depth H -> multi-step strategies become accessible
  Same O, same C, deeper composition
  Example: Tool use requires H >= 2 (use_tool THEN solve)

Substrate Neutrality:
  K is invariant under substrate substitution
  Neurons are ONE implementation, not the definition
  Example: LLM achieves K=17 without neurons
```

---

## File Inventory

```
qa_alphageometry_ptolemy/
  qa_cert_core.py                        Shared plumbing (scalars, JSON, hash, validation)
  qa_meta_validator.py                   Cross-certificate validator (tetrad)
  qa_generator_injection_certificate.py  Direction 1: Injection
  qa_diversity_collapse_certificate.py   Direction 2: Collapse
  qa_field_computation_certificate.py    Direction 3: Field
  qa_beyond_neurons_certificate.py       Direction 4: Beyond Neurons
  QA_MAP__GENERATOR_INJECTION.yaml       Direction 1 spine
  QA_MAP__DIVERSITY_COLLAPSE.yaml        Direction 2 spine
  QA_MAP__FIELD_COMPUTATION.yaml         Direction 3 spine
  QA_MAP__BEYOND_NEURONS.yaml            Direction 4 spine
  TRIAD_INDEX.md                         This file (now tetrad)
```

---

## General Theorem

```
Reachability is controlled by the configuration (S, O, C, E, H).

  Direction 1 (Injection):
    O1 subset O2, C preserved  =>  Reach(S, O2, C) contains Reach(S, O1, C)
    Witness: GeneratorInjectionCertificate

  Direction 2 (Collapse):
    O fixed, I_div violated  =>  Reach contracts
    Witness: DiversityCollapseObstruction

  Direction 3 (Field):
    O = physical operators, C = field constraints  =>  Reach realized by physics
    Witness: FieldComputationCertificate

  Direction 4 (Beyond Neurons):
    P = <S, O, C, E, H>  =>  Intelligence is substrate-neutral
    K = log10(tau_blind / tau_agent) measures intelligence
    Constraint editing, goal decoupling, horizon expansion are
    new reachability mechanisms beyond generator injection.
    Witness: BeyondNeuronsCertificate

  Corollaries:
    Barriers are generator-relative.
    Collapse is invariant-relative.
    Physics is just another generator set.
    Intelligence is just search efficiency. Neurons optional.
```
