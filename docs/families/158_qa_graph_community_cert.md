# [158] QA Graph Community Cert

**Schema**: `QA_GRAPH_COMMUNITY_CERT.v1`
**Status**: PASS (1 PASS + 1 FAIL fixture)

## What it certifies

QA feature map dimensions and community detection quality on standard benchmark graphs.

### Feature map dimensions

| Mode | Dimension | Contents |
|------|-----------|----------|
| `qa21` | 21 | Canonical invariants: b,e,d,a + squares + triangle legs + composites + ellipse |
| `qa27` | 27 | qa21 + 6 expanded (ratios, energy, angle) |
| `qa83` | 83 | Full stack: canonical + derived + modular + physical + ML features |

### Chromogeometry identity

`C^2 + F^2 = G^2` (Wildberger Theorem 6) verified for all tested directions. This is the foundational identity: `C = 2de` (green quadrance), `F = b*a` (red quadrance), `G = d^2 + e^2` (blue quadrance).

### Benchmark graphs

| Graph | Nodes | Edges | Ground truth | Louvain ARI | Louvain NMI | Modularity |
|-------|-------|-------|--------------|-------------|-------------|------------|
| Karate (Zachary) | 34 | 78 | 2 communities | 0.4905 | 0.5942 | 0.4266 |
| Football (college) | 115 | 613 | 12 conferences | 0.8069 | 0.8903 | 0.6046 |
| Dolphins | 62 | 159 | 2 groups | — (not yet in fixture) | — | — |

Data at `codex_on_QA/data/` (the `qa_lab/qa_graph/data` symlink target is a
stale Linux path `/home/player2/...` on this machine — the real files live
directly at `codex_on_QA/data/`: `karate.graphml`, `football.graphml`,
`dolphins.graphml`, real classic datasets, not synthetic).

## How to run

```bash
# Unit tests (12 tests)
cd qa_lab && PYTHONPATH=. python -m pytest qa_graph/tests/ -v

# Validator self-test
cd qa_alphageometry_ptolemy/qa_graph_community_cert_v1
python qa_graph_community_cert_validate.py --self-test
```

## What breaks

- Feature dimensions wrong (qa21 != 21, qa27 != 27, qa83 != 83)
- C^2 + F^2 - G^2 != 0 for any direction
- Benchmark ARI outside [-1, 1] or NMI outside [0, 1]

## Sources

- Wildberger, N.J. (2005), *Divine Proportions*, Ch. 6 — green/red/blue quadrance chromogeometry (C, F, G).
- Blondel, V.D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. (2008), "Fast unfolding of communities in large networks," *J. Stat. Mech.* P10008 — the Louvain method used for the benchmark numbers below.

## Verification Note (2026-07-06)

**Feature dimensions confirmed real**: independently imported
`qa_lab/qa_graph/feature_map.py:qa_feature_vector` and called it live for
modes `qa21`/`qa27`/`qa83` — got exactly 21/27/83 elements each. No bug.

**Found and fixed a real chromogeometry arithmetic bug**: the fixture's
4th test row (b=8, e=13 — consecutive Fibonacci numbers) declared
`F=264` with a falsely-declared `C2_plus_F2_minus_G2=0`. Recomputing from
the stated formulas (d=b+e=21, a=b+2e=34, F=b*a) gives **F=272**, not
264; with the correct F, 546²+272²=372100=610² (residual genuinely 0).
The validator (`GC_CHROMO`) only checked the fixture's own declared
residual field, never recomputing C/F/G from b/e itself — a
fixture-trusting gap that let a wrong F pass with a stale "0" residual
that didn't actually correspond to the declared C/F/G values. Hardened
the validator to recompute C, F, G from (b,e) directly and compare
against every declared field; verified the hardened check now correctly
rejects a reintroduced F=264.

**Found and replaced fabricated benchmark numbers**: the fixture claimed
karate "louvain: ARI=0.68, NMI=0.69" and football "qa_spectral_X:
ARI=0.45, NMI=0.72, modularity=0.55". Neither matches anything
computable in this repo: (1) `qa_spectral_X` is not a real method name —
grepped all of `qa_lab/qa_graph/*.py` and found no such method anywhere;
(2) the real, existing `karate_hub_distance_qa_results.json` (a genuine
backing script's output) gives real Louvain ARI=0.4905/NMI=0.5942 on
karate, not 0.68/0.69; (3) every real result file touching football
(`qa_native_kernel_results.json`, `unified_graph_bench_results.json`,
`integration_bench_results.json`) shows baseline ARI clustering around
0.83-0.91, nowhere near the claimed 0.45.

Independently reproduced fresh, real numbers using
`networkx.algorithms.community.louvain_communities(seed=42)` against the
real `karate_club_graph()` (with its real `club` attribute as ground
truth) and the real `codex_on_QA/data/football.graphml` (with its real
`value` conference-membership ground truth): karate
ARI=0.4905/NMI=0.5942/modularity=0.4266, football
ARI=0.8069/NMI=0.8903/modularity=0.6046 (Louvain finds 10 communities on
football, not the ground-truth 12). Replaced the fixture's fabricated
entries with these, correctly labeled "louvain" for both (dropped the
unverifiable "qa_spectral_X" method name and the specific "QA
feature-weighted spectral clustering ARI=0.68" witness claim, since
neither could be substantiated against any real code or data in the
repo).

Same finding class as [180] (same `qa_lab/qa_graph/` family): a named
method/benchmark result with no corresponding real computation anywhere
in the codebase. `--self-test` passes on both fixtures after the fixes.
