# [276] QA-ML Orbit Topology Cert

Certifies that the QA generator reachability graph (sigma, mu, lambda_2, nu)
improves orbit-class learning under sparse satellite support over a
feature-only baseline that holds everything else constant.

## Primary source

- Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph
  Convolutional Networks. *International Conference on Learning Representations*.
  arxiv:1609.02907.

The two-layer GCN architecture in the validator's `--smoke` path follows the
spectral formulation of (Kipf & Welling, 2017). The QA layer (generator
algebra, orbit classification) is taken from the canonical
`qa_orbit_rules.qa_step` and `tools.qa_ml.qa_generators`.

## Claim

Holding node features (qa_full packet `(b, e, d, a, C, F, G, b mod m//3, e mod m//3)`),
architecture (2-layer plain-torch GCN), training procedure, seeds, and
standardization fixed:

```text
graph_delta := mean_macro_f1(with_graph) - mean_macro_f1(without_graph) >= 0.10
```

where `with_graph` uses the symmetric-normalized QA generator adjacency and
`without_graph` replaces it with the identity matrix (collapsing the GCN to a
2-layer MLP on the same features).

Verified empirically for `m in {9, 12, 15, 18, 21, 24, 27, 30, 36}` at
`train_fraction=0.30`, `n_seeds=20`, `epochs=300`, `hidden=32`, `lr=0.01`,
`weight_decay=5e-4`. See `experiments/qa_ml/results_gnn_modulus_sweep.json`.

## Non-claims

- Does **not** prove orbit-class learnability for arbitrary `m`.
- Does **not** certify GCN training stability or convergence.
- Does **not** cover stochastic random graphs.
- Does **not** extend to non-period-8 satellite classes.

## Run

```bash
# Structural fixture validation (fast)
python qa_alphageometry_ptolemy/qa_ml_orbit_topology_cert_v1/qa_ml_orbit_topology_cert_validate.py

# Plus smoke re-run on m=9 (requires torch)
python qa_alphageometry_ptolemy/qa_ml_orbit_topology_cert_v1/qa_ml_orbit_topology_cert_validate.py --smoke

# Demo
python qa_alphageometry_ptolemy/qa_ml_orbit_topology_cert_v1/qa_ml_orbit_topology_cert_validate.py --demo

# Or via meta-validator
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

## References

- Kipf & Welling 2017 — arxiv:1609.02907
- `qa_orbit_rules.py` — canonical orbit family + period via qa_step
- `experiments/qa_ml/03_gnn_modulus_sweep.py` — generating sweep
- `experiments/qa_ml/benchmark_protocol_v2_modulus_sweep.json` — protocol
