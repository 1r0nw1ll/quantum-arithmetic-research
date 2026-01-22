# Release Notes: v1.0.0-qa-decision-certificate-spine

**Release Date**: 2026-01-21

---

## Summary

First stable release of the QA Decision Certificate Spine — a unified framework for machine-checkable decision-making with constructive failure witnesses.

---

## Key Features

- **Deterministic verifier** (`qa_verify.py`) — CLI tool for certificate validation
- **Failure-Completeness theorem** — every decision yields success witness OR obstruction proof
- **TLA+ executable spec** — formal model of certificate semantics
- **Reproducible RL benchmark** — Gym gridworld with obstruction witnesses
- **Exact arithmetic** — all values in `Fraction`, no floating-point ambiguity

---

## What's Included

### Core Artifacts
| File | Lines | Description |
|------|-------|-------------|
| `qa_certificate.py` | ~6200 | Certificate dataclasses, validators, recompute hooks |
| `qa_verify.py` | ~700 | CLI verifier |
| `QACertificateSpine.tla` | ~350 | TLA+ formal specification |

### Documentation
- `QA_DECISION_CERTIFICATE_SPINE.md` — full technical spec
- `ARTIFACT_MANIFEST.md` — SHA-256 hashes for all artifacts
- `REPRODUCIBILITY.md` — 30-second verification guide

### Paper
- `papers/qa_decision_certificate_spine_paper.pdf` — 10 pages, LaTeX source included

### Demos
- `demos/decision_spine_demo.py` — end-to-end 5×5 gridworld
- `demos/gym_gridworld_certificate_demo.py` — Gym benchmark comparison

---

## Certificate Types (7)

| Type | MIT Chapter | Recompute Hook |
|------|-------------|----------------|
| Inference | 3-4 | ✓ VE marginal |
| Policy | 7 | — |
| MCTS | 8 | — |
| Exploration | 9 | — |
| Filter | 9-11 | ✓ Kalman update |
| RL | 12 | ✓ Q-learning TD |
| Imitation | 13 | — |

---

## Verification

```bash
python qa_verify.py --demo
# ✔ ALL CHECKS PASSED

python -m pytest test_understanding_certificate.py -q
# 295 passed
```

---

## Breaking Changes

None. This is the initial stable release.

---

## Known Limitations

- Exact arithmetic may cause rational explosion on large factor graphs
- SCC pruning requires explicit state graph (not for continuous spaces)
- Single-threaded implementation

---

## Upgrade Path

For users of earlier `qa_certificate.py` versions:
- All existing certificate types are preserved
- New decision certificates extend (don't replace) the schema
- Legacy adapters (AlphaGeometry, Physics) remain compatible

---

## Citation

```bibtex
@misc{qa_decision_spine_2026,
  title={The QA Decision Certificate Spine: Machine-Checkable Witnesses for Sequential Decision Making},
  author={Signal Experiments Research Group},
  year={2026},
  note={v1.0.0}
}
```
