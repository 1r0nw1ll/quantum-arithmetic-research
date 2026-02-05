# Repository Guidelines

## Project Structure & Module Organization
- Root-level Python scripts (`run_signal_experiments*.py`, `dynamic_coprocessor_test.py`, `statistical_validation_gauntlet.py`) hold runnable experiments; keep new modules adjacent and note entry points in header comments.
- `data/` stores MNIST and CIFAR downloads; regenerated plots or CSVs should stay beside their scripts and replace prior artifacts only when sources change.
- Archive long-form notes in `files/` or `QAnotes/` and surface publication-ready summaries through `Documents/` and `PAPER_SUBMISSION_README.md`.

## QA Canonical Reference
- For QA control-theorem, generator, or invariant work, load `Formalizing tuple drift in quantum-native learning/files/files(1)/qa_canonical.md` and use definitions verbatim.
- Theorem statements and proofs are extracted in `QA_CONTROL_THEOREMS.md`; README-ready axioms are in `QA_AXIOMS_BLOCK.md`.
- Pipeline drift notes live in `QA_PIPELINE_AXIOM_DRIFT.md`.

## Build, Test, and Development Commands
- Work inside a virtual environment (`python -m venv .venv && source .venv/bin/activate`) to isolate dependencies.
- Install the common stack with `pip install numpy matplotlib torch torchvision tqdm seaborn scipy scikit-learn`, adding extras noted in individual scripts before committing.
- Run the harmonic simulator via `python run_signal_experiments.py`; tweak the constant block up top for new scenarios and capture the emitted PNGs/logs in the repo root.
- Reproduce the MNIST and CIFAR hybrids with `python dynamic_coprocessor_test.py` and `python statistical_validation_gauntlet.py`; both populate `data/` automatically and emit comparison plots.

## Coding Style & Naming Conventions
- Target Python 3.10+, enforce 4-space indentation, snake_case functions, UpperCamelCase classes, and uppercase module constants to match existing files.
- Maintain the `# --- Section ---` dividers to signal pipeline phases and extend them when inserting preprocessing, training, or visualization steps.
- Group imports by origin (stdlib, third-party, local), favor vectorized NumPy/Torch logic over loops, and export any notebook prototypes to `.py` modules that run headlessly.

## Testing Guidelines
- Script-based diagnostics (`dynamic_coprocessor_test.py`, `symmetry_generative_test.py`, `final_generative_test.py`) double as the regression suite; run them before proposing changes and log headline metrics in PRs.
- Prefix new diagnostic runners with their focus (e.g., `qa_entropy_test.py`), ensure they print summary stats, and save plots to the working directory.
- Seed randomness near `__main__` (`np.random.seed`, `torch.manual_seed`) so experiment diffs stay reproducible, and stage future pytest suites under `tests/` with notes in `QAnotes/`.

## Commit & Pull Request Guidelines
- With no bundled Git log, default to short, imperative subjects ("Add CIFAR stress overlay") and describe parameter or data shifts in wrapped body text when relevant.
- Mention affected scripts, regenerated artifacts, and required reruns ("Re-run `statistical_validation_gauntlet.py` for updated plots") inside each commit or PR.
- PRs should summarize intent, list touched modules, share before/after accuracy or loss deltas, embed paths for new figures, and link supporting notes in `files/` or `QAnotes/`.

## QA Decision Certificate Spine (qa_alphageometry_ptolemy/)

The `qa_alphageometry_ptolemy/` directory contains the **QA Decision Certificate Spine**—a unified framework for machine-checkable decision-making with constructive failure witnesses.

### Quick Verification
```bash
cd qa_alphageometry_ptolemy
python qa_verify.py --demo  # Should output: ✔ ALL CHECKS PASSED
```

### Core Certificate Files
| File | Description |
|------|-------------|
| `qa_certificate.py` | Core certificate dataclasses (7 types), validators, recompute hooks |
| `qa_verify.py` | CLI verifier for certificates and bundles |
| `QACertificateSpine.tla` | TLA+ formal specification |
| `QA_DECISION_CERTIFICATE_SPINE.md` | Full technical documentation |

### Generalization Bounds Mapping (arXiv:2504.05695)
| File | Description |
|------|-------------|
| `qa_generalization_certificate.py` | Dataclasses for generalization bound certificates |
| `qa_generalization_validator_v3.py` | Strict v3 validator (schema/consistency/recompute) |
| `qa_generalization_hooks.py` | 4 recompute hooks for independent verification |
| `QA_MAP__ARCH_INDEP_RELU_GENERALIZATION.yaml` | Full concept mapping specification |
| `QA_MAP_CANONICAL.md` | Gold standard mapping registry |
| `QA_GENERALIZATION_APPENDIX.md` | Conceptual interpretation document |

### NeuralGCM Mapping (Physics-ML Weather Models)
| File | Description |
|------|-------------|
| `qa_neuralgcm_certificate.py` | Conservation witness dataclasses |
| `qa_neuralgcm_validator_v3.py` | Strict v3 validator for forecast certificates |
| `QA_MAP__NEURALGCM.yaml` | Full concept mapping specification |
| `examples/neuralgcm/10day_forecast_success.json` | Success certificate |
| `examples/neuralgcm/mass_violation_failure.json` | Failure certificate |

### Sparse Attention Mapping (Transformer Efficiency)
| File | Description |
|------|-------------|
| `qa_sparse_attention_certificate.py` | Entropy/rank witness dataclasses |
| `qa_sparse_attention_validator_v3.py` | Strict v3 validator for attention health |
| `QA_MAP__SPARSE_ATTENTION.yaml` | Full concept mapping specification |
| `examples/sparse_attention/bert_base_success.json` | Success certificate |
| `examples/sparse_attention/rank_collapse_failure.json` | Failure certificate |

### Axiom AI + Execution-Grounded Research Mapping
| File | Description |
|------|-------------|
| `QA_MAP__AXIOM_AI.yaml` | Full concept mapping specification |
| `appendix/QA_AXIOM_LEDGER.md` | Integrated ledger document |
| `appendix/QA_AXIOM_STRATIFICATION_THEOREM.md` | Formal stratification theorem |
| `appendix/QA_COMPARISON_TABLE.md` | QA vs Axiom vs AlphaGeometry |
| `appendix/CHECKING_IS_NOT_ENOUGH.md` | QA manifesto |
| `schemas/QA_FAILURE_ALGEBRA.json` | Unified failure type schema |

### Cross-Paper Unification
| File | Description |
|------|-------------|
| `QA_CROSS_PAPER_UNIFICATION.md` | Unified theory across all four Gold Standard mappings |

### Certificate Tetrad + Conjecture Ledger

The **tetrad** formalizes `Capability = Reachability(S, G, I)` across four directions, with a conjecture ledger for falsifiable claims. All use exact arithmetic (`int | Fraction`), canonical JSON serialization, and failure-complete validation.

**Core theorem**: `Capability = Reachability(S, G, I)`
**Intelligence metric**: `K = log_10(tau_blind / tau_agent)`

| File | Description |
|------|-------------|
| `qa_cert_core.py` | Shared plumbing: `Scalar`, `canonical_json`, `certificate_hash`, `ValidationResult` |
| `qa_generator_injection_certificate.py` | Direction 1: G1 subset G2 -> Reach expands |
| `qa_diversity_collapse_certificate.py` | Direction 2: I_div violated -> Reach contracts |
| `qa_field_computation_certificate.py` | Direction 3: G = physical ops -> Reach realized |
| `qa_beyond_neurons_certificate.py` | Direction 4: P = <S,O,C,E,H> -> Intelligence is substrate-neutral |
| `qa_conjecture_core.py` | Conjecture dataclass, factories, CLI (imports registry from meta-validator) |
| `qa_meta_validator.py` | Cross-type validator: 5 cert types, single registry authority, 12 self-tests |
| `TRIAD_INDEX.md` | Tetrad Index: comparison table, examples, general theorem |
| `QA_MAP__GENERATOR_INJECTION.yaml` | Direction 1 YAML spine |
| `QA_MAP__DIVERSITY_COLLAPSE.yaml` | Direction 2 YAML spine |
| `QA_MAP__FIELD_COMPUTATION.yaml` | Direction 3 YAML spine |
| `QA_MAP__BEYOND_NEURONS.yaml` | Direction 4 YAML spine (Levin & Chis-Ciure) |
| `qa_ledger/conjectures/*.json` | 3 canonical conjecture JSONs with validator contracts |

**Registry authority**: `KNOWN_CERT_TYPES` and `KNOWN_CONJECTURE_TYPES` are defined once in `qa_meta_validator.py`. All other modules import from there.

### Validation Commands
```bash
# Certificate Tetrad + Conjectures (run from qa_alphageometry_ptolemy/)
python qa_meta_validator.py          # 12 tests: 9 valid, 3 invalid
python qa_conjecture_core.py         # 5 checks: factories, ledger, guards

# Decision certificate spine
python qa_verify.py --demo
python -m pytest test_understanding_certificate.py -q  # 295 tests

# Generalization bounds
python qa_generalization_validator_v3.py --demo
python qa_generalization_validator_v3.py examples/generalization/mnist_mlp_success.json
python qa_generalization_validator_v3.py --bundle examples/generalization/complete_bundle.json

# NeuralGCM
python qa_neuralgcm_validator_v3.py --demo
python qa_neuralgcm_validator_v3.py examples/neuralgcm/10day_forecast_success.json
python qa_neuralgcm_validator_v3.py examples/neuralgcm/mass_violation_failure.json

# Sparse Attention
python qa_sparse_attention_validator_v3.py --demo
python qa_sparse_attention_validator_v3.py examples/sparse_attention/bert_base_success.json
python qa_sparse_attention_validator_v3.py examples/sparse_attention/rank_collapse_failure.json

# Axiom AI (verify appendix and schema files)
ls appendix/QA_AXIOM_*.md
python -c "import json; d=json.load(open('schemas/QA_FAILURE_ALGEBRA.json')); print(f'Schema: {d[\"schema_id\"]}'); print(f'Failure classes: {len(d[\"failure_classes\"])}')"
```

### Certificate Design Principles
1. **Exact arithmetic only**: All values use `int` or `Fraction`, never `float`
2. **Failure-completeness**: Every decision yields success witness OR obstruction proof
3. **Recompute hooks**: Independent verification from raw data
4. **Deterministic serialization**: Reproducible hashes for all certificates

### Adding New Mappings
Follow the template in `QA_MAP_CANONICAL.md`:
1. Create YAML module spec (`QA_MAP__<PAPER_ID>.yaml`)
2. Implement certificate dataclasses with exact scalars
3. Implement strict v3 validator (3 levels)
4. Implement recompute hooks
5. Create example certificates (success + failure)
6. Add entry to canonical registry
