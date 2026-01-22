# Reproducibility Guide

## 30-Second Verification

```bash
# Step 1: Verify all certificates
python qa_verify.py --demo

# Expected output:
# ✔ ALL CHECKS PASSED
```

That's it. If you see `ALL CHECKS PASSED`, the artifact is verified.

---

## What the Verifier Checks

| Check | Description |
|-------|-------------|
| Schema compliance | Required fields present |
| Success/failure consistency | Success has proof, failure has obstruction |
| Regret computation | `actual - optimal = claimed_regret` |
| Method constraints | Q-learning has γ, Kalman needs linear+Gaussian |
| Bundle coherence | Cross-certificate consistency |

---

## Tamper Detection Demo

Try modifying any certificate field:

```bash
# 1. Copy a certificate
cp demos/spine_bundle.json /tmp/test_bundle.json

# 2. Corrupt it (change regret value)
sed -i 's/"cumulative_regret": 50/"cumulative_regret": 999/' /tmp/test_bundle.json

# 3. Verify — should FAIL
python qa_verify.py /tmp/test_bundle.json
```

Expected: `✘ exploration.regret_mismatch`

---

## Full Test Suite

```bash
# Run all 295 tests
python -m pytest test_understanding_certificate.py -q

# Expected: 295 passed
```

---

## Demo Regeneration

```bash
# Regenerate spine demo (5×5 gridworld, 7 certificates)
python ../demos/decision_spine_demo.py

# Regenerate benchmark demo (Gym gridworld)
python ../demos/gym_gridworld_certificate_demo.py

# Re-verify
python qa_verify.py --demo
```

---

## Hash Verification

All artifact hashes are in `ARTIFACT_MANIFEST.md`. To verify:

```bash
sha256sum qa_certificate.py qa_verify.py QACertificateSpine.tla QA_DECISION_CERTIFICATE_SPINE.md
```

Compare against manifest. Any mismatch indicates modification.

---

## Dependencies

```bash
pip install pytest  # For test suite only
```

Core code requires only Python 3.8+ standard library (`fractions`, `dataclasses`, `json`).

---

## Contact

For artifact evaluation questions, see `ARTIFACT_MANIFEST.md` for citation and contact info.
