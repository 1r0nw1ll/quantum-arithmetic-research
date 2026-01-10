# Rule 30 Center Column Bounded Non-Periodicity Certificate

**Submission to:** Wolfram Research Rule 30 Prize
**Date:** January 2026
**Author:** Will Dale

---

## Quick Summary

This package contains a **bounded non-periodicity certificate** for the Rule 30 Elementary Cellular Automaton center column, with **explicit counterexamples for all 1024 periods** (p = 1..1024) within time horizon T = 16384.

**Key Result:** For every period p ∈ [1, 1024], we provide a concrete witness (t, center(t), center(t+p)) showing center(t) ≠ center(t+p).

**Scope:** This is a bounded result with conservative claims. No asymptotic or infinite claims are made.

---

## Files in This Package

### Primary Documents

1. **rule30_proof.md** (6KB)
   - Formal proof document
   - Bounded claim statement
   - Model specification
   - Light cone argument
   - Verification notes
   - Scope and limitations

2. **rule30_certificate.json** (3.4KB)
   - Machine-readable certificate
   - Parameters and witness references
   - SHA256 hashes
   - Verification metadata

3. **rule30_submission_email.txt** (3.3KB)
   - Cover letter for submission
   - Claim scope
   - Verification instructions

### Witness Data

4. **witness_rule30_center_P1024_T16384.csv** (10KB)
   - Explicit counterexamples (1024 rows)
   - Format: p,t,center_t,center_t_plus_p
   - SHA256: `47c99fcc3edc45ac28ef3b15a296572a3e07a79c16c905fc1217d33783aef95b`

5. **witness_rule30_center_P1024_T16384.json** (92KB)
   - Structured JSON witness data
   - Includes summary metadata
   - SHA256: `572de7f23251f93bff6634b53f39fd1d795629ef3bef6bd17a4f60c6b9e54c21`

6. **center_rule30_T16384.txt** (33KB)
   - Complete center column sequence
   - Format: space-separated bits (0 or 1)
   - Used for independent verification

### Verification Tools

7. **rule30_witness_generator.py** (11KB)
   - Complete Python implementation
   - Generates all witness data from scratch
   - Includes sanity tests
   - Runtime: ~2 minutes

8. **computation_summary.txt** (723 bytes)
   - Verification summary
   - SHA256 hashes
   - Parameters
   - Success confirmation

---

## Verification Instructions

To independently verify this result:

### Method 1: Check Existing Witnesses

```bash
# Verify SHA256 hashes
sha256sum witness_rule30_center_P1024_T16384.csv
sha256sum witness_rule30_center_P1024_T16384.json

# Check a few witnesses manually:
# For period p=1, witness claims: at t=1, center(1)=1 ≠ center(2)=0
# For period p=2, witness claims: at t=0, center(0)=1 ≠ center(2)=0
```

### Method 2: Regenerate All Data

```bash
# Run the witness generator (takes ~2 minutes)
python3 rule30_witness_generator.py

# Compare output hashes with provided hashes
sha256sum witness_rule30_center_P1024_T16384.csv
# Should match: 47c99fcc3edc45ac28ef3b15a296572a3e07a79c16c905fc1217d33783aef95b
```

### Method 3: Spot Check Specific Witnesses

Implement Rule 30 in any language and verify individual witnesses:

```python
# Example: Verify witness for period p=1
# Witness claims: t=1, center(1)=1, center(2)=0

# 1. Implement Rule 30 with single-1 initial condition
# 2. Evolve to t=1: compute center(1)
# 3. Evolve to t=2: compute center(2)
# 4. Verify: center(1) ≠ center(2)
```

---

## What This Proves

### ✓ Proven Claims

- **1024 explicit counterexamples** for periods 1-1024
- **No cycles** of length ≤ 1024 in time window [0, 16384]
- **Deterministic witnesses** that anyone can verify
- **Bounded non-periodicity** within stated parameters

### ✗ NOT Claimed

- Infinite non-periodicity
- Asymptotic behavior beyond T = 16384
- Computational irreducibility
- Topological or complexity properties
- Universal behavior

---

## Key Parameters

- **Rule:** 30 (Elementary Cellular Automaton)
- **Initial condition:** Single 1 at position 0
- **Projection:** Center column (position 0 over time)
- **Time horizon:** T ∈ [0, 16384]
- **Periods tested:** p ∈ [1, 1024]
- **Window size:** 32769 cells ([-16384, +16384])
- **Success rate:** 1024/1024 (100%)
- **Failures:** 0

---

## Technical Details

### Correctness Guarantee

The finite window [-T, +T] is provably sufficient because:
- Rule 30 is a nearest-neighbor automaton
- Information propagates ≤ 1 cell per time step
- Therefore center(t) depends only on initial positions in [-t, +t] ⊆ [-T, +T]

This **light cone argument** ensures our finite computation captures exact behavior.

### Witness Format

Each witness is a tuple (p, t, center_t, center_t_plus_p) where:
- **p:** Period being tested
- **t:** Time index where inequality occurs
- **center_t:** Value at position 0 at time t
- **center_t_plus_p:** Value at position 0 at time t+p
- **Inequality:** center_t ≠ center_t_plus_p

### Reproducibility

All computation is deterministic and reproducible:
- Rule 30 truth table is standard
- Initial condition is explicit
- Evolution algorithm is provided
- All results can be independently regenerated

---

## File Sizes and Hashes

```
File                                        Size    SHA256
────────────────────────────────────────────────────────────────────
witness_rule30_center_P1024_T16384.csv      10KB    47c99fcc...
witness_rule30_center_P1024_T16384.json     92KB    572de7f2...
center_rule30_T16384.txt                    33KB    (N/A)
rule30_witness_generator.py                 11KB    (N/A)
rule30_proof.md                             6KB     (N/A)
rule30_certificate.json                     3.4KB   (N/A)
────────────────────────────────────────────────────────────────────
Total package:                              ~160KB
```

---

## Contact

For questions or clarification:
- **Author:** Will Dale
- **Submission:** Wolfram Research Rule 30 Prize
- **Date:** January 2026

---

## License

This submission and all included data/code is provided for verification purposes in connection with the Wolfram Research Rule 30 Prize. All rights reserved by the author.

---

**Ready for submission to Wolfram Research**
