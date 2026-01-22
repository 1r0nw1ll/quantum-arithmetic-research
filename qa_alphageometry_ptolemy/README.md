# QA Decision Certificate Spine

**Machine-checkable witnesses for sequential decision making**

[![Tests](https://img.shields.io/badge/tests-295%20passing-brightgreen)]()
[![Version](https://img.shields.io/badge/version-1.0.0-blue)]()

## Quick Start

```bash
# Verify all demo outputs (THE FIRST THING TO RUN)
python qa_verify.py --demo
```

Expected:
```
✔ ALL CHECKS PASSED
```

## What Is This?

A unified certificate framework for decision-making where **every computation produces a machine-checkable witness**:

- **Success** → verifiable proof
- **Failure** → constructive obstruction evidence

This covers 7 decision layers: inference, planning, MCTS, exploration, filtering, RL, and imitation learning.

## Key Features

| Feature | Description |
|---------|-------------|
| **Failure-Completeness** | Every failure is a first-class mathematical object |
| **Exact Arithmetic** | All values in `Fraction`, no floating-point ambiguity |
| **Recompute Hooks** | VE, Kalman, Q-learning updates are replayable |
| **Bundle Coherence** | Cross-certificate consistency validation |
| **CLI Verifier** | `qa_verify.py` for artifact-grade checking |

## Certificate Types

| Type | Chapter | QA-Native Feature |
|------|---------|-------------------|
| Inference | 3-4 | VE = graph reduction |
| Policy | 7 | BFS = shortest path |
| MCTS | 8 | SCC pruning witness |
| Exploration | 9 | Regret = steps − BFS |
| Filter | 9-11 | Kalman/particle |
| RL | 12 | Reward = distance_delta |
| Imitation | 13 | IRL = target inference |

## Running

```bash
# Run tests (295 passing)
python -m pytest test_understanding_certificate.py -q

# Run spine demo (5×5 gridworld through all 7 layers)
python ../demos/decision_spine_demo.py

# Run benchmark demo (Gym-style comparison)
python ../demos/gym_gridworld_certificate_demo.py

# Verify any certificate bundle
python qa_verify.py spine_bundle.json
```

## Files

| File | Description |
|------|-------------|
| `qa_certificate.py` | Core certificate dataclasses and validators |
| `qa_verify.py` | CLI verifier |
| `QACertificateSpine.tla` | TLA+ formal specification |
| `QA_DECISION_CERTIFICATE_SPINE.md` | Full documentation |
| `ARTIFACT_MANIFEST.md` | Release manifest with SHA-256 hashes |

## Paper

See `papers/qa_decision_certificate_spine_paper.pdf` (10 pages).

**Key theorem**: For every admitted decision process, exactly one of:
1. Success certificate with verifiable witness, or
2. Failure certificate with constructive obstruction.

---

## Legacy: AlphaGeometry / Physics Adapters

For backwards compatibility with earlier adapters:

```python
# AlphaGeometry adapter
from qa_alphageometry.adapters.certificate_adapter import wrap_searchresult_to_certificate
cert = wrap_searchresult_to_certificate(sr, theorem_id="ptolemy_quadrance", max_depth_limit=50)

# Physics adapter
from qa_physics.adapters.certificate_adapter import wrap_reflection_result_to_certificate
cert = wrap_reflection_result_to_certificate(result, observer_id="GeometryAngleObserver")
```

---

## Citation

```bibtex
@misc{qa_decision_spine_2026,
  title={The QA Decision Certificate Spine},
  author={Signal Experiments Research Group},
  year={2026}
}
```
