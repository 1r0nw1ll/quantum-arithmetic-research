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

- **Survivor-truth is not full-confusion truth.** FAIL_TO_PASS is run
  on patches that pass apply-check. Patches the harness *rejects* are
  never executed. We can compute precision on the survivor set but not
  recall on the underlying corpus. A perfectly conservative harness
  that rejects every codex output would also score 100% precision on
  zero survivors.
- **Truth coverage is partial.** Of 60 V1.3 codex outputs, 6 went to
  FAIL_TO_PASS. The other 54 are scored by the cascade but never
  executed. The 54 include all the rejects (above) plus expansion
  tasks where the docker image won't pull or the test infra is
  uncooperative (sphinx, sympy, matplotlib, scikit-learn — none ended
  up as cascade survivors, but if any had, they'd have needed
  per-domain test runners we haven't built).
- **Pass-16 contribution unmeasured on real outputs.** The Upwork
  structural gate has not fired on any of 30 codex outputs (Pass 8 +
  Pass 18). It is built and verified on synthetic fixtures but its
  operational contribution is currently 0.
- **Executed-truth subset is small and skewed.** 13 patches across 5
  django + 3 astropy tasks. No sympy, no sphinx, no matplotlib, no
  scikit-learn execution data. Conclusions hold on django-shaped and
  astropy-shaped fixes, not on arbitrary SWE-Bench Verified shape.
- **Tolerated gaps remain non-blocking.** 4 known gaps from Pass 7-c
  (vacuous TypeOK in TLA+, scope-overclaim with no `sorry`, etc.) are
  documented and tolerated pending recurrence; their non-recurrence is
  empirical, not proved.

## 6. Next experiment options

In rough order of information value:

- **(a) Weaker model under Pass-18-style stress.** The Upwork
  structural gate's contribution is 0 on codex. A cheap local LLM
  would likely populate the `heur_pass × struct_fail` quadrant and
  give the gate measured operational value. This is the strongest way
  to prove the gate is not theatrical.
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

## 7. Status as of v1.3

- **Tags**: `v1-blind-harness`, `v1.1-blind-harness`,
  `v1.2-blind-harness`, `v1.3-blind-harness`.
- **Master runner**: `evals/run_all.py` — clean.
- **Calibration dashboard**:
  `evals/swe_bench_blind/results/calibration/calibration_report.md` —
  designed 8/8, executed 13/13 TP, 0 FA, 0 FR on testable subset.
- **Deception regression**: 34 fixtures, 4 tolerated gaps, exit-0.
- **Cross-domain blind benchmark**: 30/30 across 4 domains.
- **Known silent layer**: Pass-16 Upwork structural gate
  (heuristic-pass × structural-fail = 0 on every output observed to
  date — Pass 8's 10 outputs and Pass 18's 10 engineered outputs).

The harness is a cascade. The cascade is sound on what it has been
tested against. The gaps in coverage are documented and bounded. None
of the layers are decorative.
