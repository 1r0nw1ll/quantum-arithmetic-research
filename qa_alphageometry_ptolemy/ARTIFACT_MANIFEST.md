# QA Decision Certificate Spine — Artifact Manifest

**Version**: 1.0.0
**Release Date**: 2026-01-21
**Tag**: `v1.0.0-qa-decision-certificate-spine`

---

## Quick Verification

```bash
python qa_verify.py --demo
```

Expected output:
```
✔ Passed:   22
✘ Failed:   0
⚠ Warnings: 4

✔ ALL CHECKS PASSED
```

---

## Core Artifacts (Frozen)

| File | Description | SHA-256 |
|------|-------------|---------|
| `qa_certificate.py` | Certificate dataclasses, validators, recompute hooks | `70a315af34e6cb9b...` |
| `qa_verify.py` | CLI verifier for certificates and bundles | `19b5afd6b0457e73...` |
| `QACertificateSpine.tla` | TLA+ formal specification | `5b000c3b1c0725ae...` |
| `QA_DECISION_CERTIFICATE_SPINE.md` | Documentation | `8e056a1f43816356...` |

### Full Hashes

```
70a315af34e6cb9bb4ef8cc62a94665c4a901302cb13a189b461f674cdf00ac1  qa_certificate.py
19b5afd6b0457e736acff359ae964f1e678dedcb5f115bcfa33ecd689550f040  qa_verify.py
5b000c3b1c0725aed2f379089ea48203b804958ee3a3075249e31d1c6fd95faf  QACertificateSpine.tla
8e056a1f4381635e87b0df8e6eea590c6fca2f21d87f5bf99c362783032317cc  QA_DECISION_CERTIFICATE_SPINE.md
```

---

## Paper Artifacts

| File | Description | SHA-256 |
|------|-------------|---------|
| `papers/qa_decision_certificate_spine_paper.tex` | LaTeX source (10 pages) | `fa414c1228f26aed...` |
| `papers/qa_decision_certificate_spine_paper.pdf` | Compiled PDF | `d0971487bfe8c8ea...` |

### Full Hashes

```
fa414c1228f26aed1b37e5757894c6bf612a0da81f4a8098356494aa97d9192d  papers/qa_decision_certificate_spine_paper.tex
d0971487bfe8c8ea159d9f717ac645d62587fa56abb79f13363e40d946886589  papers/qa_decision_certificate_spine_paper.pdf
```

---

## Demo Outputs

| File | Description | SHA-256 |
|------|-------------|---------|
| `demos/spine_bundle.json` | End-to-end 5×5 gridworld certificate bundle | `749352b5993d30b7...` |
| `demos/gym_benchmark_results.json` | Gym gridworld benchmark certificates | `c6a70dc69d228f30...` |

### Full Hashes

```
749352b5993d30b759bbca3cd3deab01fa0c051c045801d31c4476822dd884a0  demos/spine_bundle.json
c6a70dc69d228f301b92f1745e47dd4f97f745dff65122236b7880d93115cb09  demos/gym_benchmark_results.json
```

---

## Test Suite

- **295 tests passing**
- Test file: `test_understanding_certificate.py`
- Run: `python -m pytest test_understanding_certificate.py -q`

---

## Certificate Types (7 Total)

| Certificate | MIT Chapter | QA-Native Feature | Recompute Hook |
|-------------|-------------|-------------------|----------------|
| InferenceCertificate | 3-4 | VE = graph reduction | ✓ |
| PolicyCertificate | 7 | BFS = shortest path | — |
| MCTSCertificate | 8 | SCC pruning witness | — |
| ExplorationCertificate | 9 | Regret = steps − BFS | — |
| FilterCertificate | 9-11 | Kalman/particle | ✓ |
| RLCertificate | 12 | Reward = distance_delta | ✓ |
| ImitationCertificate | 13 | IRL = target inference | — |

---

## Key Theorem

**Failure-Completeness Theorem**: For every decision process admitted by the certificate spine, exactly one of:

1. A **success certificate** exists with verifiable witness, or
2. A **failure certificate** exists with constructive obstruction evidence.

---

## Reproducibility

```bash
# Clone repository
git clone <repo-url>
cd qa_alphageometry_ptolemy

# Run tests
python -m pytest test_understanding_certificate.py -q

# Verify demo outputs
python qa_verify.py --demo

# Run spine demo
python ../demos/decision_spine_demo.py

# Run benchmark demo
python ../demos/gym_gridworld_certificate_demo.py
```

---

## Citation

```bibtex
@misc{qa_decision_certificate_spine_2026,
  title={The QA Decision Certificate Spine: Machine-Checkable Witnesses for Sequential Decision Making},
  author={Signal Experiments Research Group},
  year={2026},
  note={Version 1.0.0, Artifact Release}
}
```

---

## License

Research use permitted. Contact authors for commercial licensing.

---

## Changelog

### v1.0.0 (2026-01-21)
- Initial artifact release
- 7 certificate types complete
- 295 tests passing
- TLA+ formal specification
- CLI verifier
- Paper (10 pages) with Failure-Completeness theorem
