# QA Capture Templates

Templates for saving QA results to your AI memory system (Open Brain or any MCP-connected memory). These are structured to make QA knowledge **searchable and retrievable** across sessions and across AI platforms.

---

## Why QA Captures Need Special Structure

General note-taking captures "what happened." QA captures need to capture:
1. **What the mathematical claim is** (precise, no drift)
2. **Whether it is proved or conjectured** (epistemic status)
3. **Which cert family verifies it** (machine-checkable link)
4. **Which domains it applies to** (cross-domain utility)
5. **What the correction was if it was wrong** (retraction is as important as proof)

A QA capture that says "orbit structure works for cymatics" is useless. A capture that says "Control Stack Theorem (cert [117]): singularity→satellite→cosmos with k=2 is domain-generic; proved for cymatics [105] and seismology [110]; validator: qa_control_stack_validate.py --self-test" is retrievable and verifiable.

---

## Template 1: Theorem Proved

Use immediately when a theorem is established (not at session end).

```
QA THEOREM: [formal statement, be precise]
Status: PROVED
Cert family: [[ID] if applicable]
Proof method: [BFS | algebraic | induction | validator]
Verification: [X/Y checks pass | self-test PASS | not yet machine-checked]
Domains: [which physical/mathematical domains this theorem applies to]
Key: [1-sentence non-obvious implication for practitioners]
Date: [YYYY-MM-DD]
```

**Example**:
```
QA THEOREM: Control Stack Theorem — orbit trajectory singularity→satellite→cosmos with path_length_k=2 is domain-generic: preserved across cymatics (Faraday instability) and seismology (elastic wave propagation) under QA_PLAN_CONTROL_COMPILER_CERT.v1.
Status: PROVED
Cert family: [117] QA_CONTROL_STACK_CERT.v1
Proof method: Validator (CS1-CS11) + 2 cross-domain fixtures
Verification: 2/2 fixtures PASS; meta-validator 126/126
Domains: cymatics, seismology, any domain with lawful generator algebra
Key: Once you have a certified control design in one domain, it transfers structurally to any other domain with the same orbit trajectory.
Date: 2026-03-21
```

---

## Template 2: Claim Corrected / Retracted

Use immediately when a prior claim is found to be wrong.

```
QA CORRECTION: [what was wrong]
Original claim: [the incorrect statement]
Corrected claim: [the correct statement]
Why wrong: [root cause — approximation? wrong modulus? confused orbit family?]
Evidence: [what showed it was wrong]
Affected certs: [any certs that need updating]
Date: [YYYY-MM-DD]
```

**Example**:
```
QA CORRECTION: GF(9) remark
Original claim: Z[φ]/3Z[φ] is a field extension of GF(3)
Corrected claim: 3 is inert in Z[φ], so Z[φ]/3Z[φ] ≅ GF(9) (the degree-2 extension of GF(3))
Why wrong: confused "inert prime gives degree-2 extension" with "splits into two degree-1 ideals"
Evidence: checked: f(X) = X²+X-1 is irreducible mod 3 → GF(9) = GF(3)[φ]
Affected certs: pythagorean-families paper §4 GF(9) remark
Date: 2026-03-09
```

---

## Template 3: New Domain Mapping

Use when a new physical or conceptual domain is mapped to QA.

```
QA DOMAIN MAPPING: [domain name]
State mapping:
  - [physical state 1] → [QA orbit family]
  - [physical state 2] → [QA orbit family]
  - [...]
Generator mapping:
  - [physical operation 1] → [QA generator]
  - [physical operation 2] → [QA generator]
Failure mapping:
  - [physical failure mode] → [QA fail type]
Canonical path: [initial] → [intermediate] → [target], k=[N]
Cert status: [planned / Tier 1 only / full 4-tier / certified]
Cross-domain: same orbit trajectory as [existing domain]?
Date: [YYYY-MM-DD]
```

---

## Template 4: Paper / Cert Status Change

Use when a paper becomes arXiv-ready, frozen, or submitted.

```
QA PAPER STATUS: [paper title / cert family]
Change: [arXiv-ready | frozen | submitted | accepted | retracted]
Key results: [1-3 bullet points of what's in it]
Verify script: [path] — [N/N PASS]
File: [path to tex/md]
Notes: [venue, co-authors, any open items]
Date: [YYYY-MM-DD]
```

---

## Template 5: Experimental Result

Use when an experiment produces a QA-relevant finding.

```
QA EXPERIMENT: [experiment name / script]
Result: [key numerical result, be precise — r=-0.843, not "strong correlation"]
Setup: [modulus, N, parameter values that matter]
Interpretation: [what this means in QA terms]
Limitation: [what this doesn't prove or might be wrong about]
Script: [path/to/script.py]
Date: [YYYY-MM-DD]
```

**Example**:
```
QA EXPERIMENT: Empirical kappa Exp 1
Result: r(mean_κ, final_loss) = -0.843 (n=5 seeds, lr=0.5, gain=1)
Setup: mod-9, Caps(9,9), MNIST, lr=0.5, gain=1 (normalized), 5 seeds
Interpretation: Mean orbit curvature κ is a strong predictor of convergence. Directly reflects (1-κ_t)² per-step contraction from Finite-Orbit Descent Theorem.
Limitation: Single gain value; not yet tested on all architectures. Exp 3 replicates with different gains → r=-0.845 (robust).
Script: empirical_kappa_exp1.py
Date: 2026-03-07
```

---

## Quick Reference: When to Capture

| Event | When | Template |
|-------|------|----------|
| Theorem proved | Immediately, in-session | Template 1 |
| Claim corrected | Immediately, in-session | Template 2 |
| New domain mapped | After first cert runs clean | Template 3 |
| Paper arXiv-ready | When verify script passes | Template 4 |
| Significant experimental result | When r or p-value lands | Template 5 |

**Anti-pattern**: batch-capturing at the end of a session. Results get mixed together, epistemic status blurs, you lose the context of when each thing was established.

**Pattern**: capture at the moment of discovery. Keep a running capture habit. Search your captures before starting work to avoid re-deriving known results.

---

## Searching QA Captures

When starting a session on a QA topic, search first:

```
# In your memory system (Open Brain):
search: "QA orbit trajectory [domain]"
search: "QA correction [concept]"
search: "cert family [ID]"
search: "PROVED [theorem keyword]"
```

This prevents the common failure mode of asking an AI to re-derive a result that was already established (and possibly corrected) in a previous session.
