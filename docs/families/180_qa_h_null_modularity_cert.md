# Family [180] QA_H_NULL_MODULARITY_CERT.v1

## One-line summary

H-null chromogeometric modularity: replacing the standard graph modularity null model with QA invariant H=C+F improves community detection on hub-dominated networks (Les Miserables ARI +0.050). Tier 2 — topology-specific, not universal.

## Mathematical content

### H-null model

Standard modularity uses null model P_ij = k_i * k_j / (2m) where k = degree. The QA H-null replaces this with H(k_i, k_j) = C + F where:
- C = 2 * k_j * (k_i + k_j) = green quadrance (Qg)
- F = k_i * (k_i + 2 * k_j) = red quadrance (Qr)
- H = C + F captures both symmetric AND asymmetric degree relationships

### Key identity

H / X = b/e + 4 + 2e/b grows linearly with degree asymmetry r = b/e. The standard null X = k_i * k_j is purely multiplicative; H adds the additive component through d = k_i + k_j.

### Chromogeometric connection

H = C + F = Qg + Qr (green + red quadrance from Wildberger chromogeometry). The identity C*C + F*F = G*G connects to the blue quadrance G = d*d + e*e.

### Benchmark results

| Graph | N | H-null ARI | Standard ARI | Delta |
|-------|---|-----------|-------------|-------|
| les_miserables | 77 | 0.638 | 0.588 | +0.050 |
| football | 115 | 0.824 | 0.824 | 0.000 |
| karate | 34 | 0.403 | 0.882 | -0.479 |

### Honest negatives

- Wins: 1/10 graphs (les_miserables)
- Ties: 4/10 (caveman_6x8, davis_women, football, relaxed_caveman)
- Losses: 5/10 (karate, powerlaw_100, powerlaw_200, powerlaw_300, windmill_5x4)
- Degree-CV vs delta(H−X) correlation: r = -0.25 across all 10 tested graphs (-0.30 excluding the karate outlier) — see Verification Note; this replaces an earlier "hub locality r=-0.36" claim that had no backing computation anywhere in the repo
- Effect is topology-specific to hub-dominated networks with community-internal hubs

### Mechanism

Hubs (high k_i) get penalized more heavily by the asymmetric F term, so the modularity residual A_ij - H/norm is better calibrated for networks where hubs concentrate within communities.

## Axiom compliance

- A1: Degree values are positive integers
- A2: d = b + e, a = b + 2e derived correctly
- T2: ARI scores and community assignments are observer projections
- S1: No `**2` — uses d*d, e*e
- S2: All QA state is integer

## Scripts

- `qa_lab/qa_graph/h_null_model.py` — H-null vs standard-null spectral clustering (`build_null_X`, `build_null_H`, `modularity_spectral`, `ari_score`)
- `qa_lab/qa_graph/h_null_extended_results.json` — 10-graph benchmark sweep (real, independently reproduced — see Verification Note)

## Sources

- Newman, M.E.J. & Girvan, M. (2004), "Finding and evaluating community structure in networks," *Phys. Rev. E* 69, 026113. DOI:10.1103/PhysRevE.69.026113 — the standard `k_i*k_j/(2m)` modularity null model being replaced here.
- Wildberger, N.J. (2005), *Divine Proportions*, Ch. 6 — green/red/blue quadrance chromogeometry (C, F, G).

## Verification Note (2026-07-06)

**Method label was wrong**: `primary_result.method` said "louvain with H-null
expected weight," but the actual backing code
(`qa_lab/qa_graph/h_null_model.py:modularity_spectral`) does spectral
clustering (eigendecomposition of the modularity matrix `B = A - null/norm`
followed by k-means), not Louvain (greedy local modularity optimization).
Fixed the fixture's method field.

**`h_null_nmi`/`standard_nmi` were unbacked**: no NMI computation exists
anywhere in `qa_lab/qa_graph/` — the real code only computes ARI. Removed
both fields rather than leave an unverifiable number in a PASS fixture.

**Found and fixed a fabricated benchmark list**: the fixture's
`honest_negatives` section named a specific 10-graph benchmark
(`powerlaw_cluster, karate, random_partition, stochastic_block_model,
dolphins, polbooks` as losses; `football, email_eu_core, celegans` as
ties) with counts 1 win / 3 ties / 6 losses. A repo-wide grep for every
one of those graph names (`polbooks`, `stochastic_block_model`,
`email_eu_core`, `celegans`, `random_partition`, `dolphins`) turned up
**zero** matching computations anywhere outside this one fixture — no
script in the repo has ever run H-null vs standard on any of them.

The *real* backing data (`qa_lab/qa_graph/h_null_extended_results.json`,
whose field names — `n, k, deg_cv, ari_X, ari_H, delta_H_vs_X, H_wins` —
match `h_null_model.py:analyze_graph`'s return dict exactly) covers a
different 10-graph set: `caveman_6x8, davis_women, football, karate,
les_miserables, powerlaw_100, powerlaw_200, powerlaw_300,
relaxed_caveman, windmill_5x4`. Independently reproduced this data from
scratch using the actual `build_null_X`/`build_null_H`/
`modularity_spectral`/`ari_score` functions against **real** networkx
datasets (`karate_club_graph()` with its real `club` attribute as ground
truth; `les_miserables_graph()` with `greedy_modularity_communities` as
ground truth) — got an **exact** match: les_miserables
ari_X=0.5877/ari_H=0.6377/delta=+0.0500, karate
ari_X=0.8823/ari_H=0.4028/delta=-0.4794. This confirms the headline
result and the extended-results file are real, even though the original
driver script that generated the other 8 graphs (caveman/powerlaw/
windmill variants) is no longer present in the repo.

Correctly tallied from the real 10-graph data: **1 win (les_miserables),
4 ties (caveman_6x8, davis_women, football, relaxed_caveman), 5 losses
(karate, powerlaw_100/200/300, windmill_5x4)** — not 1/3/6, and not the
fabricated graph names. Fixed the fixture and this doc.

**"Hub locality correlation r=-0.36" was also unbacked**: grepped the
entire repo for "locality"/"hub_locality" in this context — no script or
result file computes any such metric. Independently computed the
closest real analog (Pearson r between `deg_cv` and `delta_H_vs_X`
across the real 10-graph sweep): **r = -0.25** (-0.30 excluding the
karate outlier) — same negative direction, different exact number since
"hub locality" was never a real, defined, computed quantity. Replaced
the claim with this honestly-computed correlation.

**Validator hardened**: `HN_HONEST` previously only checked
`wins <= total_graphs` — a check the buggy fixture already passed
(1+3+6=10). Added two new checks: (1) `wins+ties+losses == total_graphs`
(trivially true here but would catch a totals bug); (2) named-list
composition check — `len(wins_on)/len(ties_on)/len(loses_on)` must equal
the declared `wins`/`ties`/`losses` counts. This *would* have caught the
original bug (3 named ties but the list only ever had... actually the
original list had exactly 3 tie-names and 6 loss-names, matching its own
wrong totals — the bug was that the *set of graphs* was fabricated, not
an internal arithmetic slip). The new check still adds real protection
against a future mislabeled-count regression and is a genuine hardening
even though it wasn't sufficient by itself to catch this specific
fabrication (only cross-referencing real backing data could).

Core claims independently reconfirmed correct: `H = C+F` chromogeometric
identity, `C*C+F*F=G*G`, and the headline les_miserables/football/karate
ARI numbers are all real and reproducible. `--self-test` passes on both
fixtures after the fixes.
