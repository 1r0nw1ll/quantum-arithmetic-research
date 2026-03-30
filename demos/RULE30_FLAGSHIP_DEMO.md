# Rule-30: The Flagship "Understanding ≠ Prediction" Demo

## The One-Sentence Pitch

> **Any simulator can generate Rule 30. Only RML can prove why certain patterns are impossible.**

---

## The Core Thesis

| Layer | What it does | Difficulty |
|-------|-------------|------------|
| **Prediction** | Apply local rule, generate pattern | Trivial (O(1) per cell) |
| **Understanding** | Prove non-periodicity | Requires 1024+ explicit witnesses |

World models predict *what happens*.
RML certifies *why some things cannot happen*.

---

## The Demo

```bash
python demos/rule30_understanding_demo.py
```

**Output**: A machine-verifiable `UnderstandingCertificate` showing:

- **Target**: "Rule 30 center column is periodic"
- **Reachable**: `false` (no periodic structure exists)
- **Obstruction**: 64 explicit counterexamples
- **Each counterexample**: `(period, t, c[t], c[t+p])` where `c[t] ≠ c[t+p]`

---

## Why Rule-30?

1. **Visual**: Everyone recognizes the pattern
2. **Maximal contrast**: Prediction = trivial, understanding = hard
3. **Prize hook**: Directly relevant to Wolfram's computational irreducibility claims
4. **Clean narrative**: "Simulation is not understanding"

---

## The Certificate Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: UNDERSTANDING (RML)                                   │
│  • 64 derived invariants: non_periodic_p1 ... non_periodic_p64  │
│  • Each has derivation witness: (t, c[t], c[t+p])              │
│  • Strategy with derivation (not free text)                     │
│  • Compression ratio: trace_length / explanation_length         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: REACHABILITY (QAWM)                                   │
│  • Target: "periodic center column"                             │
│  • Reachable: FALSE                                             │
│  • Fail type: GENERATOR_INSUFFICIENT                            │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: WORLD MODEL (Simulation)                              │
│  • Apply Rule 30 local update                                   │
│  • Generate any pattern you want                                │
│  • This is TRIVIAL                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Makes This Different

| Standard Approach | QA-RML Approach |
|------------------|-----------------|
| "Rule 30 appears random" | "Rule 30 center column fails periodicity test for all periods 1-1024, with explicit witnesses" |
| Narrative claim | Machine-verifiable certificate |
| Ad-hoc observation | Derivation witnesses for every claim |
| Cannot be replayed | QARM-compatible transition logs |

---

## Key Files

| File | Purpose |
|------|---------|
| `demos/rule30_understanding_demo.py` | Main demo script |
| `demos/rule30_understanding_cert.json` | Exported certificate |
| `qa_certificate.py` | Certificate schema (v2) |
| `test_understanding_certificate.py` | 25 unit tests |

---

## The Philosophical Point

When someone says "my model understands X", ask:

1. Does it have derivation witnesses for its claims?
2. Can you replay and verify its reasoning?
3. Does it prove what's impossible, or just predict what's likely?

If the answer is "no" to any of these, it's prediction, not understanding.

**Rule-30 is the cleanest proof of this distinction.**

---

## Reference

- Gupta & Pruthi (2025). "Beyond World Models: Rethinking Understanding in AI Models." arXiv:2511.12239v1.
- QA-RML Framework: `qa_understanding_cert/v2` schema
