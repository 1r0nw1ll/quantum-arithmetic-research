# Blind-Eval Harness — Synthesis (Pass 17)

A tight epistemic record of what the cascade is, what it has been
shown to do, what it has *not* yet been shown to do, and where the
remaining unknowns live. Written after Pass 18 closed the
"is the structural gate real or just dormant" question.

Companion docs:
- `BLIND_EVAL_HARNESS_V1.md` — design + scope
- `TLA_HARNESS_REMEDIATION_PLAN.md` — pass-by-pass running tracker

The anchor claim of this synthesis:

> **The harness works as a cascade, not a heuristic miracle.**

Cheap structural truth gates do most of the first-pass filtering;
heuristic discrimination matters after that; execution truth on
survivors is the strongest check. Each layer earns its keep against a
different failure mode. None of them are independently sufficient.

## 1. The arc — TLA+ incident to v1.3

The harness was built reactively after a real failure:

- **Pass 1–6** — TLA+ formal-publication remediation. The original
  monolithic scorer false-rejected 100% of upstream-approved TLA+ specs
  and 100% of mathlib4 Lean files because it conflated "matches our
  local bundle shape" with "is this work legitimate?".
- **Pass 7** — split intrinsic legitimacy from submission-bundle
  completeness. This is the central architectural insight; everything
  else is consequences of it.
- **Pass 7-a/b/c/d** — corpus expansion (TLA+ 99 / Lean 128); deception
  regression suite; Upwork-style as a third domain.
- **Pass 8** — first live-agent stress (codex on Upwork). Deception
  variants (rush / overclaim / looks_done / minimal_tests) caught.
- **Pass 9** — shared cross-domain core extracted to `evals/_blind_core/`.
- **Pass 10** — fourth domain: SWE-Bench Verified.
- **Pass 11** — freeze + tag `v1-blind-harness`.
- **Pass 12** — SWE-Bench live-agent stress. 17/25 heuristic-accept.
- **Pass 13a/b** — `git apply --check` against real cloned repos.
  Discovered 89% of "heuristic-accept" SWE-Bench patches were
  structurally malformed and would never apply. Pass 13b wired the
  check into the scorer; accept rate dropped 68% → 12% on Pass-12.
- **Pass 13c** — first execution truth (FAIL_TO_PASS). Of 2 testable
  codex patches, both fix their bugs. One harness false-revise surfaced
  on canonical-file mismatch.
- **Pass 14a** — tiered patch-relevance replaced binary canonical-file
  rule. Recovered the false-revise. Also recovered 27 SWE-Bench fixture
  metadata files Pass 10 had silently dropped under the global `*.json`
  gitignore.
- **Pass 14b** — astropy execution truth via official SWE-Bench docker
  images. 3/3 untested astropy patches pass. Tagged `v1.2-blind-harness`.
- **Pass 15** — calibration dashboard combining designed truth +
  executed truth. 8/8 designed; 7/7 executed on the testable Pass-12
  subset.
- **Pass 16** — Upwork structural gate (`py_compile` + import smoke +
  `pytest --collect-only`). Armed but silent on the Pass-8 set.
- **Pass V1.3** — SWE-Bench corpus expansion 20 → 50 tasks. 60 codex
  outputs. 28 heuristic-accept; apply-check rejected 22 (79%
  structural malformation rate at scale); 6 cascade survivors actually
  fix their bugs (6/6 FAIL_TO_PASS). Combined executable truth 13/13.
  Tagged `v1.3-blind-harness`.
- **Pass 18** — engineered Upwork stress targeting Pass-16 sub-checks.
  Structural gate fired alone on **0/10** outputs because codex
  resisted every adversarial framing.
- **Pass 19** — same 5 variants × 2 tasks, but agent is
  `opencode/gpt-5-nano` (smallest gpt-5 variant). Gate fired on
  **3/10** outputs (vs Pass 18's 0/10) — `nested_package` produced
  unresolvable `from .` imports; `broken_parametrize` produced
  uncollectable tests. The gate-only quadrant remains 0 because every
  gate-firing coincides with a heuristic-firing for an *unrelated*
  reason on the same output (missing README, missing required keyword,
  placeholder marker). The gate's contribution under weaker craft is
  thus **unique failure-mode coverage on shared outputs** plus
  **decision strengthening** (3 outputs flip revise → reject), not
  empty-quadrant rescue.
- **Pass 20** — closed the 4 known tolerated deception-regression
  gaps from Pass 7-c via narrow regression-backed detectors: TLA
  vacuous TypeOK / membership-only invariant (revise); TLA README
  vs spec semantic misalignment (reject); Lean vacuous-premise
  theorem (revise); Lean broad scope-overclaim absent typeclass
  parameter (revise). Closure was conservative — the misalignment
  detector originally regressed 6 upstream Paxos/consensus specs
  (revise → reject) when run under the charitable adapter because
  `.cfg` files and 0-variable helper `.tla`s like `Utils.tla`/
  `Digraph.tla` tripped the "0 vars / 0 actions" branch. Caught on
  master-runner pre-commit verification, fixed by guarding to
  `.tla`-only with `≥1 state variable`. Final state: deception
  regression 34/34 MATCH / 0 known_gap, upstream TLA accept 54.5%
  (unchanged), upstream Lean accept 100% (unchanged), charitable
  adapter improved (21/45 flipped→accept, 0 regressed).

## 2. What each domain taught

| Domain | What it taught about the harness |
|---|---|
| **TLA+** | The intrinsic-vs-bundle split. Pass 7's reframing came from TLA+ false-rejecting upstream specs because the bundle scorer was masquerading as a content scorer. |
| **Lean 4** | A pedagogical-vs-deceptive `sorry` distinction is recoverable from comment context. Heuristic adapters can extract intent without re-reading the proof; this is the model for charitable scoring across domains. |
| **Upwork-style** | Live-agent deception is real but predictable. Codex resists overclaim framing on patches but will overclaim in README copy; the textual gate catches it. The structural gate is built but unmeasured against this model. |
| **SWE-Bench Verified** | Structural truth (`git apply --check`) does most of the work. Heuristic-only would have over-accepted 79% of malformed patches at expansion scale. The execution layer (FAIL_TO_PASS docker) is the only one that can decide whether a *valid* patch actually fixes the bug. |

## 3. Where the strongest signal comes from

In order of decisiveness on observed data:

1. **Structural truth** — `git apply --check` for SWE-Bench;
   `py_compile` + import smoke + `pytest --collect-only` for Upwork.
   On SWE-Bench V1.3, this layer alone rejected 22/28 heuristic-accepts
   (79%). Cheap, tool-native, hard to argue with.
2. **Execution truth** — FAIL_TO_PASS inside the official SWE-Bench
   docker image. Decides whether a valid patch fixes the bug. On the
   testable subset (Pass-12 + V1.3), 13/13 cascade survivors actually
   fix their bugs.
3. **Text heuristics** — overclaim markers, requirement keywords,
   placeholder/TODO scans, tautological-assertion detection,
   tiered patch-relevance. Catches operational deception
   (overclaim README, removed tests, wrong-file patches) that
   structural and execution layers don't see by themselves.

The interaction matters: heuristic-accept-only is a 47% raw rate at
V1.3 scale; after structural gate it's 10%; after execution it's
unchanged at 10% (every cascade survivor was a true fix). Removing any
layer changes which kinds of outputs slip through, not just how many.

## 4. Proven vs suggested

**Proven on the observed data** (post-cascade, on the testable subset):

- Cascade is sound on the testable SWE-Bench subset: 13/13 TP, 0 FA,
  0 FR. Designed truth: 8/8.
- Apply-check structurally rejects 79% of heuristic-accepts at
  expansion scale (22/28 on Pass V1.3).
- Codex on Upwork-style tasks resists adversarial prompts targeting
  structural malformation: 0/10 engineered prompts trip the gate.

**Suggested, but not yet proven**:

- That the cascade *generalizes* to other models. Every codex output
  here is from one model family. A weaker model would likely populate
  the `heur_pass × struct_fail` quadrant Pass 18 left empty.
- That the cascade *generalizes* to other task classes. Upwork tasks
  here are small (factorial bugfix, CSV domain count). Larger
  frameworks, longer time horizons, or scaffold-heavy tasks may
  exercise gate failure modes the small tasks don't.
- That the cascade catches *all* the failure modes that matter. We
  catch the ones we built fixtures for (deception regression suite,
  34 fixtures, 4 tolerated gaps) and the ones execution truth surfaces
  (mostly malformation). Failure modes nobody has thought to write a
  fixture for are by definition uncovered.

## 5. Known limits

These are the things the dashboard cannot currently close:

- ~~**Survivor-truth is not full-confusion truth.**~~ *Closed in Pass 21.*
  Reject-sample FAIL_TO_PASS on the 13 V1.3 codex outputs the cascade
  rejected (with patch.diff saved) gives full confusion truth on the
  testable subset:

  | | actually_fixes | does_not_fix |
  |---|---:|---:|
  | cascade=accept | 6 (TP) | 0 (FA) |
  | cascade=reject/revise | **3 (FR)** | 10 (TN) |

  Precision: 100% (6/6); **recall: 67% (6/9)**.

  Of 10 true-rejects, all 10 were `apply_check_failed` — the patches
  literally won't apply. Of 3 patches that DID apply cleanly among the
  reject sample, all 3 actually fix the bug. The 3 false-rejects map
  to 2 specific heuristic bugs (unified-diff regex too strict;
  placeholder detector ignores test-file context); both are fixable
  without broad heuristic retuning. See
  `evals/swe_bench_blind/results/pass21_reject_sample/pass21_report.md`.
- **Truth coverage is partial.** Of 60 V1.3 codex outputs, 6 went to
  FAIL_TO_PASS. The other 54 are scored by the cascade but never
  executed. The 54 include all the rejects (above) plus expansion
  tasks where the docker image won't pull or the test infra is
  uncooperative (sphinx, sympy, matplotlib, scikit-learn — none ended
  up as cascade survivors, but if any had, they'd have needed
  per-domain test runners we haven't built).
- **Pass-16 contribution measured under weaker craft, not codex.**
  The gate has not fired on any of 20 codex outputs (Pass 8 + Pass 18)
  but fired on 3/10 `gpt-5-nano` outputs (Pass 19). The contribution
  is real — the gate caught structural failures (broken `from .`
  imports, uncollectable parametrize) the heuristic genuinely cannot
  see — but the contribution lives in *unique-failure-mode coverage on
  shared outputs* and *decision strengthening* (revise → reject), not
  in catching outputs the heuristic misses entirely. The empty
  `heur_pass × struct_fail` quadrant across 20 codex + 10 nano outputs
  is real; it just under-counts the gate's value because it ignores
  per-output failure-mode count.
- **Executed-truth subset is small and skewed.** 13 patches across 5
  django + 3 astropy tasks. No sympy, no sphinx, no matplotlib, no
  scikit-learn execution data. Conclusions hold on django-shaped and
  astropy-shaped fixes, not on arbitrary SWE-Bench Verified shape.
- ~~**Tolerated gaps remain non-blocking.**~~ *Closed in Pass 20.* The
  4 known gaps from Pass 7-c (TLA vacuous TypeOK, TLA README/spec
  misalignment, Lean vacuous premise, Lean broad scope-overclaim) now
  MATCH their expected outcomes via narrow regression-backed detectors.
  Deception regression: 34/34 MATCH, 0 KNOWN_GAP_TOLERATED.

## 6. Next experiment options

In rough order of information value:

- **(a) ~~Weaker model under Pass-18-style stress.~~** *Done — Pass 19.*
  Result: gate fires on weaker craft (3/10 nano vs 0/10 codex) but
  contribution is unique-failure-mode coverage on shared outputs, not
  empty-quadrant rescue. The gate is not theatrical; the framing
  "armed but unused" was wrong because it counted outputs, not
  failure-modes-per-output.
- **(b) Wire FAIL_TO_PASS docker truth as a default scorer flag**
  (currently invoked manually post-cascade). Would convert
  survivor-truth from a separate offline run into a first-class score.
- **(c) Extend execution truth to non-django domains.** Re-run V1.3
  expansion truth on whichever apply-check survivors emerge from
  astropy / sympy / sphinx prompts in future runs.
- **(d) Compute confusion truth, not just survivor truth.** Run
  FAIL_TO_PASS on a sample of harness-*rejected* patches to estimate
  recall, not just precision. Costly (rejections are 80%+ of corpus)
  but the only way to bound the cascade's recall.
- **(e) Larger Upwork task class.** The Pass-8 / Pass-18 set is two
  small tasks. A small framework or scaffold-heavy task may exercise
  Pass-16 sub-checks codex's small-task craft sidesteps.

## 7. Status as of v1.5

- **Tags**: `v1-blind-harness`, `v1.1-`, `v1.2-`, `v1.3-`, `v1.4-`,
  `v1.5-blind-harness`.
- **Master runner**: `evals/run_all.py` — clean across all 6 suites.
- **Calibration dashboard**:
  `evals/swe_bench_blind/results/calibration/calibration_report.md` —
  designed 8/8, executed 13/13 TP, 0 FA, 0 FR on testable subset.
- **Deception regression**: 34/34 MATCH, **0 KNOWN_GAP_TOLERATED**,
  exit-0.
- **Cross-domain blind benchmark**: 30/30 across 4 domains.
- **Upstream-corpus stability**: TLA intrinsic accept 54.5%, Lean
  intrinsic accept 100% — both held steady through Pass 20.
- **Charitable adapter**: 21/45 revise→accept (was 18 pre-Pass-20),
  0 regressed.
- **Pass-16 silent quadrant**: heuristic-pass × structural-fail = 0
  on 30 observed Upwork outputs (Pass 8's 10 + Pass 18's 10 codex +
  Pass 19's 10 nano). Gate fires (in any quadrant) on 3/10 nano
  outputs but never alone.
- **SWE-Bench full-confusion truth (Pass 21)**: precision 6/6 = 100%,
  **recall 6/9 = 67%** on the testable subset. 3 measured false-rejects
  trace to 2 specific fixable heuristic bugs.

The harness is a cascade. The cascade is sound on what it has been
tested against. The previously-tolerated gaps are now measured
detections. None of the layers are decorative.
