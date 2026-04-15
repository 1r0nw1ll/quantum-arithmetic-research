# Experiment & Benchmark Axioms Block (Canonical v1.0)

Authority over experiment design and benchmark design, mirroring
`QA_AXIOMS_BLOCK.md` (which governs code-level QA compliance).

Source harvesting: `MEMORY.md` Hard Rules + post-mortems from Open Brain
(2026-04-01 Adversarial Testing, 2026-04-02 No Stochastic, 2026-04-05
Bateson×ATT&CK NULL + Bearden text transfer, 2026-04-08 QA Always
Applies, 2026-04-09 Raw d=b+e, 2026-04-10 Primary Sources + Landscape,
2026-04-13 cmap calibration bug).

Enforcement: `tools/qa_axiom_linter.py` rule groups `EXP`, `BENCH`,
`NULL`. Pre-commit gate via `qa_axiom_linter --staged`.

---

## Part A — Experiment Axioms

**E1 (Declaration)**: Every empirical experiment script must declare
an `EXPERIMENT_DESIGN = {...}` block at module top. Rule `EXP-1`.

Required fields:
- `hypothesis` — what is being tested (one sentence, falsifiable).
- `null` — what the null model is.
- `null_independence` — how the null is statistically independent of
  any clustered/structured data. Addresses 2026-04-01 circularity rule.
- `pre_registered_seed` — integer, fixed before seeing data.
- `n_trials` — integer, ≥1.
- `accept_criterion` — decision rule for accepting the hypothesis.
- `reject_criterion` — decision rule for rejecting the hypothesis.
- `observer_projection` — how continuous inputs become (b,e) integers
  (same substance as `DECL-1`, but experiment-scoped).

**E2 (No rerun same config after NULL)**: If a prior run was declared
NULL, the same configuration is not to be rerun in search of a
positive result. A new configuration requires a new design block and
an explanation of what changed and why. Rule `EXP-2-W` (warning).

**E3 (Real data before synthetic)**: Synthetic-only experiments are
permitted but must declare `real_data_status` = `"pending"` or
`"<path/source>"`. A synthetic failure does not invalidate a real
result; the design block must make the distinction explicit. Rule
`EXP-3-W` (warning).

**E4 (Reproduction before interpretation)**: If an experiment claims
to reproduce a prior result, it must declare
`reproduces = "<prior_run_id_or_path>"` and verify match before
interpreting new data. Rule `EXP-4-W` (warning).

**E5 (Implementation-first on unsupportive outcomes)**: An unsupportive
outcome is evidence of an implementation bug or wrong observer
projection. Modular arithmetic applies everywhere integers exist, so
an unsupportive outcome reflects the (b,e) mapping, not the framework.
Design blocks must declare
`on_unsupportive = "investigate_observer | investigate_implementation | pre_registered_accept"`.
Rule `EXP-5` (error).

**E6 (Primary sources)**: When evaluating a fringe-labeled source, the
design must cite primary literature, not consensus priors. Field
`primary_sources` required if the hypothesis references contested
work. Rule `EXP-6-W` (warning).

---

## Part B — Benchmark Axioms

**B1 (Declaration)**: Every benchmark script must declare a
`BENCHMARK_DESIGN = {...}` block at module top. Rule `BENCH-1`.

Required fields:
- `qa_method` — name of the QA technique being benchmarked.
- `baselines` — non-empty list of baseline methods.
- `datasets` — list of dataset names with source identifier.
- `same_seed_all_methods` — boolean, must be `True` for comparative fairness.
- `calibration_provenance` — what calibration (cmap, quantile edges,
  feature pairing, etc.) was learned on the training set and how.
  Addresses 2026-04-13 cmap-tuned-for-finance bug where a hand-tuned
  finance cmap silently failed on every non-finance benchmark.
- `framework_inheritance` — one of:
  - `"inherit: <framework_name>"` — benchmark ports the observer
    framework (windows, feature streams, clustering) from a prior
    working cert;
  - `"ported: yes, diff: <list>"` — framework differs from prior
    working cert, differences enumerated;
  - `"novel"` — first benchmark of this observer framework.
  Addresses 2026-04-05 Bearden text-transfer lesson: the issue was the
  observer-framework port, not the target domain; the observer framework
  (windows, feature streams, clustering scheme) had to match the prior cert.
- `metrics` — list of scored metrics (AUROC, ARI, NMI, etc.).

**B2 (Baseline parity)**: Baselines must share the same seed, same
data split, and same preprocessing as the QA method. Rule `BENCH-2`.

**B3 (Calibration declared)**: Any cmap, quantile edges, feature
pairing, or observer calibration must have provenance in
`calibration_provenance`. A default carried from another domain must
either be recalibrated or declared as a known-wrong baseline. Rule
`BENCH-3`.

**B4 (Framework inheritance)**: If a benchmark ports QA from one
domain to another, the observer framework (windows, feature streams,
clustering scheme) is inherited by default. Deviations require
enumeration in `framework_inheritance`. Rule `BENCH-4-W` (warning).

---

## Part C — Null-Model Axioms

**N1 (Independence)**: Null models must be statistically independent
of any clustered or structured data used in the positive arm. A null
that inherits structure from the test data is not a null. Rule
`NULL-1`.

**N2 (Permutation parity)**: Permutation nulls must permute the
correct axis (rows vs columns vs time). Declaration required in
`EXPERIMENT_DESIGN["null_permutation_axis"]` if the null is a
permutation test. Rule `NULL-2-W` (warning).

**N3 (Surrogate type disclosure)**: Surrogate null models must declare
the surrogate type (row_permuted, phase_randomized, block_bootstrap,
etc.) in `EXPERIMENT_DESIGN["surrogate_types"]`. Rule `NULL-3-W`
(warning).

---

## Enforcement

EXP and BENCH rules are not suppressible. A script that performs an
empirical experiment or benchmark must either reference a valid protocol
JSON explicitly or carry a valid sibling protocol JSON in the same
directory. Undeclared exceptions are the primary failure mode these
rules guard against.

---

## Relationship to QA Axioms

The QA axioms (A1, A2, T2, S1, S2, T1, DECL, ELEM, ORBIT, FIREWALL)
govern **code correctness**. The Experiment & Benchmark axioms (E1–E6,
B1–B4, N1–N3) govern **scientific validity of empirical claims**. Code
that is QA-compliant can still produce invalid results if the
experiment design or benchmark construction is flawed. Both gates
must pass for a result to be trustworthy.
