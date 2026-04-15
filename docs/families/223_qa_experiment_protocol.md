# Family [223] QA_EXPERIMENT_PROTOCOL.v1

## One-line summary

An enforceable design contract for empirical QA studies — hypothesis,
surrogate model, source mapping, ablation, reproducibility,
pre-registration, decision rules, and observer projection — validated
by a nine-gate JSON-schema validator and
enforced at file level by the linter rule `EXP-1`.

## Why

`MEMORY.md` Hard Rules encoded scientific-validity discipline as
guidance only. Post-mortems from Open Brain showed the same failure
modes recurring: null models that share structure with the positive
arm (2026-04-01 Adversarial Testing), unsupportive-outcome responses
that dismiss the framework instead of investigating the observer
projection (2026-04-08 QA Always Applies), re-running the same
configuration after an unsupportive outcome in search of a positive
(2026-04-05 Bateson × ATT&CK). This family lifts those rules into a
machine-checkable contract every empirical script must reference.

## What it certifies

For every empirical QA script there exists a concrete object

`X = (H, N, P, D, O, R, S, A, M)`

with

- `H` hypothesis (falsifiable)
- `N` null model with generating process, held-fixed fields, permuted
  fields, and explicit independence argument
- `P` pre-registration (seed, UTC date, n_trials ≥ 1)
- `D` decision rules (accept, reject, `on_unsupportive` enum)
- `O` observer projection (description + state alphabet)
- `R` real-data status (path | "pending" | "synthetic_only")
- `S` source mapping (`theory_doc`, `primary_source`, rationale), with
  `primary_source` required to appear in `theory_doc`
- `A` ablation contract naming the callable and the QA structure destroyed
- `M` reproducibility manifest (seed, data hash/status, package versions,
  results ledger)

## Gates (validator)

1. Schema Validity — conforms to `qa_experiment_protocol/schema.json`
2. Null Design Defined — non-empty `generating_process`, `held_fixed`,
   `permuted`, and `independence_argument`
3. Pre-Registration Complete — seed + `date_utc` + `n_trials ≥ 1`
4. Decision Rules Complete — `accept_criterion` + `reject_criterion` + `on_unsupportive` in {`investigate_observer`, `investigate_implementation`, `pre_registered_accept`}
5. Observer Projection Declared — `description` + `state_alphabet` non-empty
6. Real Data Status — pending, synthetic-only, or existing data path
7. Source Mapping — declared primary source appears in the referenced theory doc
8. Ablation — callable, destroyed structure, and expected direction declared
9. Reproducibility — seed, data hash/status, package versions, and ledger path declared

## How to run

```bash
python qa_experiment_protocol/validator.py --self-test
python qa_experiment_protocol/validator.py path/to/experiment_protocol.json
```

## How scripts declare compliance

Inline: `EXPERIMENT_PROTOCOL_REF = "path/to/experiment_protocol.json"`
or place `experiment_protocol.json` next to the script. Linter rule
`EXP-1` gates both forms on any file containing a statistical-test
call site (ttest, ks_2samp, mannwhitneyu, permutation_test, wilcoxon,
kruskal, ranksums, pearsonr, spearmanr).

## Authority

`EXPERIMENT_AXIOMS_BLOCK.md` Part A (E1–E6) and Part C (N1–N3).
Mirrors the enforcement shape of family [35] QA Mapping Protocol.
