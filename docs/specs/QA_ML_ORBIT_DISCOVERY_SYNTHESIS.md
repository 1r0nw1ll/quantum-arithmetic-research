# QA-ML Orbit Discovery — Synthesis (2026-05-09)

> Status: **synthesis note**. Not a cert. Captures what the QA-ML →
> orbit-rule investigation discovered, fixed, certified, and parked
> over 2026-05-08 / 2026-05-09 so the chain can stop here without
> losing the result.

## What QA-ML discovered

A single experimental result that started in machine learning ended up
exposing a foundational bug in the canonical orbit classifier:

1. **QA-ML v1** (sample-efficiency benchmark, mod-24): polynomial QA
   structural features `(b, e, d, a, C, F, G)` add little signal beyond
   raw `(b, e)` for orbit prediction. **Modular-phase columns** (mod
   `m // 3`) carry the actual signal. *The QA expansion that mattered
   was the modular phase, not the algebraic packet.*

2. **QA-ML v2** (GCN over QA reachability graph): a 2-layer plain-torch
   GCN trained transductively on the symmetric-normalized adjacency
   built from the QA generators σ, μ, λ_2, ν beats an identity-adjacency
   ablation by **+0.14 to +0.27 macro F1** at small training counts.
   *QA generator topology carries information the features do not.*

3. **QA-ML v2 modulus sweep** (cert [276] generating sweep): graph
   advantage holds across `m ∈ {9, 12, 15, 18, 21, 24, 27, 30, 36}` at
   `train_fraction = 0.30`, 20 seeds, all 9 PASS. *The graph effect is
   not mod-24-specific.* During this sweep, the labeling oracle
   (`qa_orbit_rules.orbit_family`) silently disagreed with the
   period-based ground truth on `m ∈ {15, 30}` — flagged in cert [276]
   scope_note as a follow-up boundary observation.

## What got fixed in the code

`qa_orbit_rules.orbit_family` was using an **algebraic divisor
shortcut** `(m // 3) | b ∧ (m // 3) | e` as if it were the canonical
classifier. The shortcut works for `m ∈ {9, 24}` (the documented
KNOWN_MODULI) but diverges for other moduli.

**Code fix at commit `e7b2af0`:**

- `orbit_family(b, e, m)` is now canonical via `orbit_period`:
  `period 1 → singularity, period 8 → satellite, else → cosmos`. Works
  for any `m ≥ 2`.
- The algebraic shortcut moves to a named helper
  `orbit_family_divisor_shortcut(b, e, m)` — preserved for callers that
  explicitly want the O(1) form on verified moduli.
- Both decorated with `@lru_cache(maxsize=None)` so the price of going
  empirical is amortized.

Existing callers (~25 grep'd files) all run at `m ∈ {9, 24}` where
shortcut and canonical agree exactly. No downstream behavior change.

## What got certified

Two cert families that map the divisor shortcut's failure surface:

| Cert | Family | Failure mode | Bounded scope | Verified |
|---|---|---|---|---|
| **[277]** | `qa_orbit_pisano_5_factor_boundary_cert_v1` | shortcut **under-counts** real period-8 satellites by 32 | `m = 15k`, `k ∈ {1..12, 15, 20}` | 14 values |
| **[278]** | `qa_orbit_no_3_divisor_overclaim_cert_v1` | shortcut **over-claims** 9 false satellites in the 3×3 grid `{(a · m//3, b · m//3) : a, b ∈ {1, 2, 3}}` | `3 ∤ m, m ≥ 7, m ≠ 8` | 25+ values |

Both cite Wall (1960) Fibonacci-mod-m / Pisano-period framing as the
primary source. Both registered in `qa_meta_validator.py` and pass the
human-tract doc gate after `docs/families/README.md` was updated.

The boundary exception **`m = 8`** (4×4 grid → 15 overclaims, not 9) is
explicitly excluded from [278] v1 and exposed by a FAIL fixture rather
than buried.

## What is theoretical and parked

Three observations that the sweep surfaced but the certs do **not**
claim:

1. **`max(qa_step period at m) = π(m)`** for 21 of 22 tested moduli, with
   `m = 75` as a verified exception (`max = 200 = 2 · π(75)`). The
   doubling is presumably an effect of the A1 correction
   `((b + e − 1) mod m) + 1` shifting the orbit space relative to pure
   Fibonacci-mod-m, but that interaction is not formalized.

2. **CRT-style decomposition**: observed periods are a subset of
   divisors of `lcm` of prime-power Pisano periods, with `m = 75` as
   the same exception. Whether this holds universally (or whether more
   `pisano_doubled` cases exist for `m ∈ {25k, 5^a · k, ...}`) is open.

3. **Period 4 occurs iff `5 | m`** (in tested set). Empirical, not
   proven.

The **orbit-period spectrum draft cert ([279] candidate)** is **parked**
in `docs/specs/QA_ORBIT_PERIOD_SPECTRUM_CERT_DRAFT.md` until the theory
catches up. Promoting now would either certify the m=75 anomaly as a
fixture flag (Option A) or exclude it (Option B); either way, the cert
would document the divergence without explaining it. Better to land the
explanation first.

## What is the next ML experiment

The QA-ML v2 graph-topology result is the strongest signal in this
chain. The natural extension is **not** another classifier; it is a
**predictive structure-discovery** model that can re-derive the
boundary findings the cert chain just enumerated.

**Proposed QA-ML v3 question:**

> Can a QA-graph model predict shortcut-failure modes (under-count vs
> over-claim) and orbit-period classes (singularity / satellite /
> cosmos / period-`p` for any `p`) directly from `(b, e, m)`, with the
> QA generator graph as input topology?

Target tasks:

- **T1. Shortcut-failure mode prediction.** Three-class: `agrees`,
  `under-counts (and by how much)`, `over-claims (and by how much)`.
  Train across `m ∈ {6..120}`. Hold out `m=75` as a known anomaly and
  see whether the model treats it specially.

- **T2. Orbit-period classification.** Predict
  `period(b, e, m) ∈ {1, 4, 8, π(m), other}` from `(b, e, m)` plus QA
  generator graph features. The features that matter (per QA-ML v1+v2):
  - integer-derived features `(b, e, d, a, C, F, G, b mod m//3, e mod
    m//3)`
  - position in the QA reachability graph (degree, distance to
    singularity, σ-orbit length up to truncation)
  - `m`'s factorization vector (or just gcd's with small primes
    `{2, 3, 5, 7, 11}`)

- **T3. Symbolic extraction.** Once T1/T2 train clean, extract the
  decision boundaries the model uses and check whether they recover the
  certified rules ([277], [278]) automatically. If they do, that is
  evidence the QA-graph topology is sufficient to learn the shortcut's
  failure surface. If they extract a tighter rule, that is a new
  theorem candidate.

The shift here is from **ML-as-classifier** (which we already did) to
**ML-as-discoverer**: train the model, then read the model.

## What we are NOT doing

- Not promoting cert [279] (orbit-period spectrum) until the m=75
  anomaly is theoretically explained.
- Not extending the orbit-period sweep further (e.g. to m=150, 200, ...)
  pre-theory.
- Not opening a separate `qa_orbit_pisano_doubling_cert` family for
  m=75 alone — fragmenting the boundary surface across many small
  certs is the failure mode this synthesis is recording.
- Not adding more QA-ML cert families on the strength of the QA-ML v2
  result alone — the cert ecosystem is dense enough; the next ML work
  should produce *findings*, not *cert families*.

## Lineage commits (origin/main)

```text
9c75f55 feat(qa_ml): v1 sample-efficiency baseline
88395c8 feat(qa_ml): v2 GCN benchmark
564f7b0 feat(qa_ml): modulus sweep
5de8133 cert(276): register QA-ML orbit-topology family
e7b2af0 fix(qa_orbit_rules): canonical orbit_family via orbit_period
deebcd8 docs(orbit): draft Pisano 5-factor boundary cert
fe1c0ba docs(orbit): refine draft with gcd-signature theorem
6dc7ba4 cert(277): register orbit Pisano 5-factor boundary
73916f8 docs(orbit): draft adjacent overclaim cert
7c5d803 cert(278): register no-3 divisor overclaim boundary
8f53e7a docs(orbit): draft orbit-period spectrum cert
f725bef fix(meta_validator): clear ORBIT-5 lint trip
e54de7c docs(families): add README index entries for [276], [277], [278]
```

Plus this synthesis note.

## References

- `tools/qa_ml/` — feature extractor, dataset, generators, graph.
- `experiments/qa_ml/` — benchmarks, protocols, results, ledgers.
- `qa_orbit_rules.py` — canonical orbit_family + divisor shortcut
  helper; commit `e7b2af0`.
- `qa_alphageometry_ptolemy/qa_ml_orbit_topology_cert_v1/` — [276].
- `qa_alphageometry_ptolemy/qa_orbit_pisano_5_factor_boundary_cert_v1/` —
  [277].
- `qa_alphageometry_ptolemy/qa_orbit_no_3_divisor_overclaim_cert_v1/` —
  [278].
- `docs/specs/QA_ORBIT_PERIOD_SPECTRUM_CERT_DRAFT.md` — [279] draft,
  parked.
- `docs/specs/QA_ORBIT_PISANO_5_FACTOR_BOUNDARY_CERT_DRAFT.md` and
  `docs/specs/QA_ORBIT_5_FACTOR_NO_3_OVERCLAIM_CERT_DRAFT.md` — earlier
  design drafts for [277] and [278].
- Wall, D. D. (1960). *Fibonacci series modulo m*. Amer. Math. Monthly
  67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.

## Closing

```text
QA-ML found a labeling bug.
The classifier got fixed.
The shortcut's failure surface got certified ([277], [278]).
The full spectrum got tabulated but not understood.
Stop the cert cascade here. Resume QA-ML as a theory-discoverer, not a
cert factory.
```
