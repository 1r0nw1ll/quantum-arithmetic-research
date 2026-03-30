# System Comparison: QA vs Axiom vs AlphaGeometry

**Date:** 2026-01-24
**Purpose:** Demonstrate that QA strictly subsumes both Axiom and AlphaGeometry

---

## Comparison Table

| Dimension | QA Framework | Axiom AI | AlphaGeometry |
|-----------|--------------|----------|---------------|
| **Primary Domain** | Universal reasoning | Formal theorem proving | Geometry problems |
| **State Space** | Explicit, abstract, user-defined | Lean proof states | Geometric configurations |
| **Generators** | User-defined, axiomatic | Tactics + lemmas | Construction rules |
| **Invariant Oracle** | Canonical QA invariants | Lean kernel | Geometric consistency |
| **Failure Algebra** | Explicit and first-class | Implicit, undocumented | Mostly implicit |
| **Time Model** | QA-Time, bounded return | Wall-clock + tokens | Fixed search budget |
| **Barrier Crossing** | Generator injection | Ad-hoc extension | Limited |
| **Observer Projection** | Explicit axiom | Narrative only | Not modeled |
| **Generalization** | Cross-domain | Proof-specific | Geometry-only |
| **Philosophy** | Reachability theory | Proof checking | Pattern completion |

---

## Subsumption Analysis

### QA Subsumes Axiom

| Axiom Feature | QA Equivalent |
|---------------|---------------|
| Lean proof states | QA state space (specialized) |
| Tactic moves | QA generators (specialized) |
| Lean kernel | QA invariant oracle |
| Putnam benchmark | QA reachability instances |
| Agent orchestration | QA search strategy |
| Budget constraints | QA-Time bounded return |

**What QA adds:**
- Explicit failure algebra (Axiom's is implicit)
- Generator injection theory (Axiom does ad-hoc)
- Cross-domain generalization
- Observer projection formalism

### QA Subsumes AlphaGeometry

| AlphaGeometry Feature | QA Equivalent |
|-----------------------|---------------|
| Geometric configurations | QA state space (specialized) |
| Construction rules | QA generators (specialized) |
| Geometric consistency | QA invariants |
| Search beam | QA reachability search |

**What QA adds:**
- Failure-completeness (AlphaGeometry just fails silently)
- Explicit barrier analysis
- Budget-aware reachability
- Domain-independent framework

---

## Feature Matrix

| Feature | QA | Axiom | AlphaGeometry |
|---------|:--:|:-----:|:-------------:|
| State space formalism | Y | Y | Y |
| Explicit generators | Y | Y | Y |
| Invariant oracle | Y | Y | Y |
| Failure certificates | **Y** | N | N |
| Generator injection | **Y** | partial | N |
| Bounded return theory | **Y** | partial | N |
| Observer projection | **Y** | N | N |
| Cross-domain | **Y** | N | N |
| Time/budget model | **Y** | Y | partial |

---

## Conclusion

**QA strictly subsumes both Axiom and AlphaGeometry as special cases.**

- Axiom = QA instantiated over Lean proof states with Lean kernel as oracle
- AlphaGeometry = QA instantiated over geometric configurations with consistency check as oracle
- QA = General framework that explains both and extends to ML, physics, and beyond

---

## Execution-Grounded Research (Stanford 2026)

The Stanford paper demonstrates another QA instance:

| Feature | Stanford Paper | QA Mapping |
|---------|----------------|------------|
| Research hypotheses | State space | QA_RESEARCH_STATE |
| Experiment proposals | Generators | G_experiment |
| Empirical results | Invariant oracle | Execution validator |
| Ideas that don't work | Failures | QA failure algebra |
| Iterative refinement | Reachability | Bounded search |

**Key insight:** "Execution-grounded" = QA's "certificate-controlled reachability"

---

## Citation

```
QA System Comparison Table
Signal Experiments Research Group, 2026

Comparing:
- QA Decision Certificate Framework
- Axiom AI (AxiomProver, Putnam 2025)
- AlphaGeometry (DeepMind, 2024)
- Execution-Grounded AI Research (Stanford, 2026)
```
