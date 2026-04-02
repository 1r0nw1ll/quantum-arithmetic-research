# QA Canonical Reference System

**Status:** Production-ready  
**Version:** 1.0  
**Date:** December 29, 2025

---

## What This Is

A **versioned, importable axiom system** for Quantum Arithmetic (QA-RML) that eliminates definition drift across AI agents, code implementations, and papers.

**Core Principle:** QA is not fuzzy context to be "remembered" — it's a formal system to be **imported**.

---

## Files in This Package

### 1. `qa_canonical.md`
**THE SINGLE SOURCE OF TRUTH**

Contains:
- ✅ Exact 21-element invariant packet formulas
- ✅ Generator algebra (σ, μ, λ₂, ν) with legality rules
- ✅ Phase system (φ₉, φ₂₄)
- ✅ Failure taxonomy (OUT_OF_BOUNDS, PARITY, etc.)
- ✅ Key theorems from Paper 1
- ✅ Caps(30,30) checksums for validation
- ✅ Usage protocol and citation format

**Immutability:** This file is LAW. Changes require version bump + changelog.

### 2. `validate_canonical_v2.py`
**CANON-FAITHFUL VALIDATOR**

Tests implementations against the canonical spec:
- Recomputes from canonical equations (no hardcoded expectations)
- Validates internal consistency
- Checks Caps(30,30) topology against §12 checksums
- Verifies generator semantics

Run: `python validate_canonical_v2.py`

---

## How to Use This System

### For AI Sessions (Claude, ChatGPT, Claude Code)

**At the start of EVERY QA conversation, paste this:**

```
You must follow qa_canonical.md (QA Canonical Reference v1.0).
Do not redefine symbols, simplify formulas, or infer missing constraints.
If a needed definition is absent from qa_canonical.md, stop and ask.
All results must be consistent with the canonical checksums in §12.
```

Then upload `qa_canonical.md` to the conversation.

**Result:** AI agents now have **deterministic, identical definitions** — no drift, no approximation.

---

### For Code Implementation

#### Step 1: Import canonical definitions
```python
from qa_canonical import construct_qa_state, sigma, mu, lambda2, nu
```

#### Step 2: Validate against checksums
```bash
python validate_canonical_v2.py
```

Expected output:
```
✅ All invariant consistency checks pass
✅ Caps(30,30) Σ₃ checksums match canonical reference
✅ All generator semantics match canonical reference
🎉 Implementation is CANONICAL-COMPLIANT
```

#### Step 3: Make validation a CI gate
```yaml
# .github/workflows/ci.yml
- name: Validate against canonical spec
  run: python validate_canonical_v2.py
```

---

### For Papers & Publications

In your LaTeX:

```latex
All QA definitions follow the canonical reference \cite{qa_canonical}.
State space is defined in \S1, generators in \S2.
```

Or inline:
```latex
From the 21-element invariant packet (qa_canonical.md \S1.3),
we have $C = 2ed$ and $F = ba$ with $C \neq F$ for all primitive states.
```

---

## Version Control Protocol

### Current Version
```
Version: 1.0
Date: December 29, 2025
Status: IMMUTABLE
```

### Change Protocol
Any modification requires:
1. ✅ Version bump (e.g., 1.0 → 1.1)
2. ✅ Explicit changelog entry in §11
3. ✅ Re-validation: `python validate_canonical_v2.py`
4. ✅ Verification that Papers 1-2 still hold

### Git Workflow
```bash
git add qa_canonical.md
git commit -m "Canonical QA reference v1.0"
git tag qa-canon-v1.0
```

---

## Why This Matters

### The Problem We're Solving

**Before:**
- "Use the 21-element invariant packet"
- AI: "Which formulas exactly?"
- User: [re-states everything]
- **Result:** Definition drift, non-reproducible results

**After:**
- "Load qa_canonical.md"
- AI: [reads exact definitions]
- **Result:** Deterministic, identical formulas across all agents

### What Makes QA Special

QA is **not symbolic math** — it's:
- A **geometric control theory**
- A **reachability algebra**
- A **topology of failure modes**

That means:
- ❌ Definition drift = changing the manifold
- ❌ "Close enough" = false theorem
- ❌ Implicit assumptions = non-reproducible results

This canonical reference ensures:
- ✅ Deterministic definitions
- ✅ Reproducible experiments
- ✅ Auditable claims
- ✅ Reviewer-proof rigor

---

## Current Validation Status

**All systems validated:**

✅ **Invariant Packet**: All 21 formulas consistent  
✅ **Generators**: σ, μ, λ₂, ν semantics match spec  
✅ **Caps(30,30)**: Topology checksums exact match
- States: 900 ✓
- Edges: 2220 ✓
- Failures: 1380 ✓
- SCCs: 1 ✓
- Max SCC: 900 ✓

---

## Cross-Tool Consistency

You can give `qa_canonical.md` to:
- ✅ Claude (this conversation)
- ✅ ChatGPT (other browser session)
- ✅ Claude Code (local workstation)
- ✅ Local LLMs
- ✅ Human collaborators

**Everyone works from identical definitions.**

---

## Quick Start Checklist

For your next QA session:

- [ ] Upload `qa_canonical.md` to the AI agent
- [ ] Paste the session header (see "For AI Sessions" above)
- [ ] Confirm: "Does your current implementation pass validate_canonical_v2.py?"
- [ ] If not, identify mismatches and fix implementation (not the canonical spec)

---

## Repository Structure (Recommended)

```
your-repo/
├── docs/
│   └── qa_canonical.md          # THE LAW
├── src/
│   ├── qa_oracle.py             # Implementation
│   └── benchmark_suite.py
├── tests/
│   └── validate_canonical_v2.py # Validator
└── papers/
    ├── paper1_qa_control.tex
    └── paper2_qawm.tex
```

---

## FAQ

**Q: Can I modify `qa_canonical.md`?**  
A: Only with explicit version bump, changelog, and full re-validation. Treat it like a programming language spec.

**Q: What if I find an error in the canonical spec?**  
A: If it's truly an error (not just inconvenient), fix it with version bump. But verify Papers 1-2 still hold.

**Q: Can I have multiple versions?**  
A: Yes — use git tags. But each paper should cite a specific version (e.g., "qa_canonical.md v1.0").

**Q: What if an AI agent drifts from the spec?**  
A: Re-paste the session header and re-upload `qa_canonical.md`. The drift is in the agent's working memory, not the spec.

---

## Contact

For questions about QA canonical definitions:
- See `qa_canonical.md` first
- If ambiguous, file an issue with the specific section reference
- Proposed changes require justification + validation results

---

**Remember: QA is not remembered. QA is imported.**

🎉 You now have reviewer-proof, AI-aligned, production-grade QA semantics.
