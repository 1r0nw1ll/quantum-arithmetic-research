# PHILOMATH Top-5 QA Crosswalk

## Purpose

This tract converts the highest-value PHILOMATH ingestion targets into **QA-native reduction work**.
It is not a chapter summary. It is a filter:

- what can be admitted into the QA-adjacent corpus
- how each claim must be rewritten
- what exact promotion test must pass before anything moves upward

Source machine files:

- `qa_ingestion_sources/qa_philomath_corpus_manifest.json`
- `qa_ingestion_sources/qa_philomath_ingestion_queue.json`
- `qa_ingestion_sources/qa_philomath_claim_ledger.json`
- `qa_ingestion_sources/qa_philomath_top5_crosswalk.json`

Executable harness:

- `tools/qa_philomath_top5_promotion_tests.py`
- `qa_ingestion_sources/qa_philomath_top5_fixtures.json`

## Operating Rule

For these chapters, PHILOMATH contributes only if the claim survives a rewrite into one of:

- exact modular structure
- exact factor or graph structure
- exact deterministic classification
- exact invertible witness

If a claim cannot survive that rewrite, it remains overlay material.

## Top 5

| Priority | Chapter | Claim class | QA-native reduction | Promotion test |
|----------|---------|-------------|---------------------|----------------|
| 1 | The Digital Root | modular | Rewrite entirely in `Z/9Z` | exact reduction to residue classes |
| 2 | Prime Numbers | modular | Rewrite as mod-24 wheel filter only | residue filter verification |
| 3 | Reciprocity of Numbers and Prime Factorization | structural analogy | Rewrite as factor-pair symmetry / factor graph / difference-of-squares | factor recovery |
| 4 | Number Classification | modular | Rewrite as exact membership predicates | membership testability |
| 5 | Semiprime Factorization - The Geometric Solutions | structural analogy | Rewrite as invertible semiprime witness geometry | invertible geometry |

## Chapter Notes

### 1. The Digital Root

Allowed rewrite:

- `rho_9(n) = n mod 9`
- digital-root patterns become residue-class patterns

Forbidden shortcuts:

- decimal mysticism
- "harmonic closure" without explicit modular content

Promotion condition:

- no semantic content is lost when the claim is stated purely in `Z/9Z`

### 2. Prime Numbers

Allowed rewrite:

- primes greater than `3` must lie in residue classes coprime to `24`
- the 24-wheel is a **necessary filter**

Forbidden shortcuts:

- spoke membership implies primality
- wheel picture as proof

Promotion condition:

- prime claims must be stated as residue filters with necessary/sufficient language kept explicit

### 3. Reciprocity of Numbers and Prime Factorization

Allowed rewrite:

- factor pairs mirrored around arithmetic invariants
- difference-of-squares encoding
- exact factor graphs

Forbidden shortcuts:

- purely pictorial reciprocity
- geometry with no factor recovery

Promotion condition:

- the construction must carry real factor information

### 4. Number Classification

Allowed rewrite:

- exact predicate classes
- congruence classes
- tuple families
- recursive generators with a deterministic rule

Forbidden shortcuts:

- family resemblance language without a decision rule
- recursive ancestry without a recursion law

Promotion condition:

- each class must have a deterministic membership test and boundary examples

### 5. Semiprime Geometry

Allowed rewrite:

- geometry as a witness layer over exact semiprime arithmetic
- invertible coordinate encoding
- mirror planes indexed by factor pairs

Forbidden shortcuts:

- geometry-first factor revelation
- non-invertible pictures

Promotion condition:

- the geometry must decode back to the factor data

## Executable Coverage

The promotion-test harness currently checks:

- mod-9 reduction examples
- mod-24 wheel filter examples
- factor recovery through midpoint/delta encoding
- deterministic classification predicates
- invertible semiprime witness geometry

These are not final validators. They are executable stubs that enforce the reduction discipline.

## Current Verdict

This top-5 set is the right forward path for PHILOMATH inside the QA corpus.

- strong entry: digital root, prime wheel, exact classification
- medium entry: factor symmetry once arithmetic recovery is explicit
- acceptable bridge: semiprime geometry only if invertible

Everything beyond this should continue to route through the manifest and queue before promotion.
