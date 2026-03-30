# Checking Everything Is Not Enough

**A QA Manifesto**

---

Checking everything guarantees correctness.

It does not guarantee understanding.

It does not guarantee progress.

It does not guarantee reachability.

---

## What Formal Verification Answers

Formal verification answers one question:

> "Is this trace valid under the invariant oracle?"

This is important. This is non-negotiable. This is the kernel stratum.

---

## What Formal Verification Does NOT Answer

- **Why** the trace exists
- **Why** it was found
- **Why** others fail
- **What generator** would make it easier
- **Whether failure** is fundamental or budgeted
- **Where** the barriers are
- **How** to collapse components

A system that only checks is **blind to its own topology**.

---

## The QA Position

**Quantum Arithmetic asserts:**

- **Reasoning is reachability.** Every claim is a path through state space.
- **Understanding is geometry.** The shape of the space determines what's easy.
- **Failure is structure.** Obstructions are first-class mathematical objects.

---

## The Difference

| Approach | Focus | Limitation |
|----------|-------|------------|
| Verification | Is this proof correct? | Doesn't explain difficulty |
| Reachability | Can we get there? And why not? | Explains barriers |

**Checking is necessary.** Without it, we have no ground truth.

**Reachability theory is sufficient.** With it, we can navigate the space.

---

## What This Means for AI Research

### Axiom AI

Axiom proves: "We can check every step of every proof."

QA asks: "Why are some proofs hard to find? What generator would collapse the barrier?"

### AlphaGeometry

AlphaGeometry proves: "We can solve IMO geometry problems."

QA asks: "What's the topology of geometry problems? Why did this one fail?"

### Execution-Grounded Research (Stanford 2026)

Stanford proves: "We can run AI research ideas and learn from failures."

QA asks: "What's the structure of the failure space? How do we classify and exploit it?"

---

## The Thesis

> **Progress comes not from checking harder, but from mapping the space, classifying obstructions, and choosing generators that collapse components.**

---

## Practical Implications

1. **Build failure algebras** - Don't just log "failed"; classify why
2. **Study generator injection** - What new move would cross the barrier?
3. **Model time budgets** - Is this fundamentally hard or just expensive?
4. **Map component structure** - What's reachable from what?
5. **Expose observer projections** - Where does human intuition diverge from formal truth?

---

## The Bottom Line

**Checking is the floor.**

It tells you: "This is correct."

It does not tell you: "This is reachable."

**Reachability theory is the ceiling.**

It tells you: "Here is the map of what's possible, what's hard, and why."

---

## That is the difference between verification and science.

---

**Signal Experiments Research Group**
**2026-01-24**
