# [28] QA Graph Structure Bundle

## What this is

Certifies QA structural feature runs for graph community detection under paired baseline controls. This family encodes deterministic feature extraction, baseline parity, phase diagnostics, and metric-delta proofs in a failure-complete certificate.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__GRAPH_STRUCTURE.yaml` |
| Validator | `qa_graph_structure_validator_v1.py` |
| Bundle emitter/validator | `qa_graph_structure_bundle_v1.py` |
| Reference cert | `certs/QA_GRAPH_STRUCTURE_CERT.v1.json` |
| Cert hash sidecar | `certs/QA_GRAPH_STRUCTURE_CERT.v1.sha256` |
| Bundle manifest | `certs/QA_GRAPH_STRUCTURE_BUNDLE.v1.json` |
| Cert schema | `schemas/QA_GRAPH_STRUCTURE_CERT.v1.schema.json` |
| Bundle schema | `schemas/QA_GRAPH_STRUCTURE_BUNDLE.v1.schema.json` |
| Success example | `examples/graph_structure/graph_structure_success.json` |
| Failure example | `examples/graph_structure/graph_structure_parity_failure.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate individual cert
python qa_graph_structure_validator_v1.py --demo

# Validate with recompute level
python qa_graph_structure_validator_v1.py --level recompute examples/graph_structure/graph_structure_success.json

# Emit + check bundle
python qa_graph_structure_bundle_v1.py --emit --check

# Or via meta-validator (runs as family [28])
python qa_meta_validator.py
```

## Semantics

- **Generator set**: `{sigma_feat_extract, sigma_qa_embed, sigma_cluster, sigma_eval, sigma_phase_analyze}`
- **Paired baselines**: dataset/algorithm/seeds must match across baseline and QA traces
- **Delta integrity**: `delta_metrics[k] == qa_metrics[k] - baseline_metrics[k]`
- **Phase diagnostics**: phase-24 and phase-9 witnesses are replay-checked from trace terminal states
- **Hard invariants**: tuple consistency, feature determinism, eval reproducibility, trace completeness, baseline pairing

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `out_of_bounds` | Invalid graph size/community bounds | Correct graph input/config bounds |
| `invariant` | Determinism/reproducibility contract broken | Repair invariant-producing stage |
| `phase_violation` | Declared phase lock broken | Fix generator or phase mapping |
| `parity` | QA/baseline seed or config mismatch | Re-run with strictly matched pairing |
| `reduction` | Claim emitted without baseline evidence | Include baseline metrics/trace and re-emit |

## Changelog

- **v1.0.0** (2026-02-09): First graph-structure cert emission + bundle manifest + meta-validator gate.
- **2026-07-06**: See Verification Note below.

## Verification Note (2026-07-06)

Follow-up to auditing [27] (Elliptic Correspondence) and [19] (Topology
Resonance), which shared a "meta-validator only checks bundle file-hash
integrity, never invokes the real Level 1-3 validator" gap. Confirmed
family [28] had the identical wiring gap and fixed it the same way (now
also runs `qa_graph_structure_validator_v1.py`'s real validator against
the reference certs after the hash check).

`validate_recompute` itself is genuinely implemented here (not a stub
like [27]): it recomputes `trace_digest`, replays `baseline_trace`/
`qa_trace` to terminal metrics, and — importantly — verifies
`delta_metrics[k] == qa_metrics[k] - baseline_metrics[k]` as real
arithmetic, not just presence. This is stronger than [19]'s validator.

**Independently spot-checked the shipped example's karate-club claim**
(`graph_context: {dataset_id: karate_club, algorithm: louvain,
node_count: 34, edge_count: 78, community_count: 4, split_seed: 17,
clustering_seed: 17}`) against real `networkx` (v3.6.1,
`louvain_communities`, seed=17, this repo's `.venv`):
- Node/edge counts (34/78) and community count (4) matched Zachary's
  real karate club graph exactly.
- Computed modularity (0.4449) did **not** match either the declared
  `baseline_metrics.modularity` (1/2=0.5) or `qa_metrics.modularity`
  (14/25=0.56).
- Computed ARI (0.4646) did not exactly match declared
  `qa_metrics.ari` (17/25=0.68); computed NMI (0.5878) was close to
  declared `qa_metrics.nmi` (29/50=0.58) but not exact.

**This is left as an open question, not resolved either way**: the
mismatch could mean the declared numbers are fabricated (matching the
pattern found in sibling graph-benchmark certs [158]/[180] earlier in
this audit cycle), or it could mean a different Louvain implementation/
library version was used at authoring time (`networkx`'s
`louvain_communities` vs. the separate `python-louvain`/`community`
package can give different partitions for the same seed), or that
"qa_metrics" reflects some QA-structural-packet augmentation over plain
Louvain that isn't concretely specified anywhere in this codebase (the
same "generator has no concrete implementation" issue found in [19]:
`sigma_feat_extract`/`sigma_qa_embed`/`sigma_cluster`/`sigma_eval`/
`sigma_phase_analyze` are bare labels, not implemented functions).
Flagging honestly rather than declaring either "confirmed fabricated"
or "confirmed clean" without further investigation pinning down the
exact algorithm/library version used.
