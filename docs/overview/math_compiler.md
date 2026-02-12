# The QA Math Compiler: Certified Mathematical Reasoning

## Overview

The QA Math Compiler converts informal mathematical reasoning into reproducible,
machine-verified, and auditable artifacts.

Instead of treating proofs as static text or opaque model outputs, the Math
Compiler treats reasoning as a compilation pipeline:

> Natural language -> formal proof trace -> deterministic replay -> certification

Every successful result is backed by executable traces. Every failure is typed
and preserved.

## Why This Exists

Modern AI systems can generate mathematical text, but outputs are often
non-reproducible, hard to verify, and difficult to audit after the fact.

The QA Math Compiler addresses this by imposing software-style correctness
guarantees on mathematical reasoning.

A proof is not considered valid unless it:

- replays deterministically,
- passes formal verification gates,
- is bound to its inputs and metadata,
- survives independent rerun checks.

## The Three Pillars

The current QA stack combines three certified subsystems.

### 1. Family [31]: Math Compiler Stack

Family [31] implements the core compilation pipeline:

| Stage | Function |
|-------|----------|
| Task | Standardized Lean problem definition |
| Trace | Stepwise proof construction record |
| Replay | Deterministic rerun verification |
| Pairing | Human <-> formal alignment certificate |
| Mining | Lemma extraction for compression |

Each stage yields either a verified witness or a typed failure record.

### 2. Family [34]: Rule 30 Certified Discovery

Family [34] demonstrates that the QA framework can certify open-ended
mathematical discovery workflows, not just replay known proofs.

It includes:

- bounded exploration,
- invariant tracking,
- typed counterexample/failure certification,
- independent replay verification.

### 3. Competency Certification Framework

The competency subsystem measures what a model can reliably do under explicit
constraints, using reproducible traces and certificate artifacts.

This shifts evaluation from one-off benchmark numbers to verifiable capability
records.

## End-to-End Flow

Together, the three pillars form a closed loop:

```text
Informal Claim
  ->
Math Compiler (Family 31)
  ->
Certified Proof Artifacts
  ->
Discovery / Exploration (Family 34)
  ->
Competency Evaluation
  ->
Certified Capability Profile
```

## What Makes This Different

### Replayability First

If a result cannot be reproduced from artifacts, it is not accepted.

### Failure is Data

Failures are classified, versioned, and preserved as first-class outputs.

### Artifact-Centric Design

Claims are carried by schemas, traces, manifests, validators, and docs. A claim
without artifacts is out of scope.

### External Verifiability

Third parties can run:

```bash
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

to reproduce repository validation gates locally.

## Relationship to \"AI Mathematicians\"

The QA Math Compiler is infrastructure-first. It does not try to imitate a
human mathematician's writing style; it enforces correctness constraints so AI
systems can participate in mathematics through certified traces.

Models become:

- trace generators,
- hypothesis explorers,
- lemma miners,

inside a deterministic validation environment.

## Practical Applications

### Research

- reproducible conjecture exploration,
- formalization of informal arguments,
- reusable proof-trace corpora.

### AI Development

- training on certified traces,
- capability evaluation via competency certificates,
- safer deployment for formal reasoning tasks.

### Governance

- auditable technical claims,
- reproducibility checks for high-stakes outputs,
- independent verification workflows.

## Current Status

At present:

- Family [31] enforces replay-first compilation gates,
- Family [34] supports certified discovery workflows,
- competency certificates are integrated,
- repository-level validation is executable via the meta-validator.

## Design Philosophy

The QA approach follows four rules:

1. No result without replay.
2. No replay without determinism.
3. No determinism without artifacts.
4. No artifacts without documentation.

## Getting Started

From the repository root, run:

```bash
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

This executes the integrated family validation sweep and reports PASS/FAIL per
family.

## Outlook

Near-term priorities include:

- expanding certified corpora,
- improving lemma mining quality and compression,
- scaling replay-based formalization workflows,
- integrating broader proof-assistant ecosystems.

The goal is not faster text generation. The goal is trustworthy mathematics.
