# QA-AlphaGeometry

**Discrete Harmonic Priors for Symbolic Geometry Reasoning**

A symbolic geometry solver that uses **Quantum Arithmetic (QA) invariants** as structured priors to reduce combinatorial search in automated theorem proving.

## Architecture

- **Core**: Rust (fast, zero-dependency symbolic engine)
- **Bindings**: PyO3 (Python integration for ML/datasets)
- **Strategy**: Hybrid heuristic + QA priors (→ learned policy later)

## Status

🚧 **Under Construction** (Week 1 of 12)

**Roadmap**:
- Week 1-2: Core IR + Rules + QA layer
- Week 3-4: Search + Python bindings
- Week 5-6: Benchmark on Geometry3K
- Week 7-8: Write paper
- Week 9-12: Validate + submit

## Quick Start

```bash
# Build Rust core
cd core && cargo build --release

# Run solver (when ready)
cargo run --bin qa-geo-solve -- examples/triangle.json

# Python (when ready)
pip install maturin
cd bindings && maturin develop
python -c "import qa_geo; print(qa_geo.solve(problem))"
```

## Key Innovation

**QA Tuples as Geometric Priors**:
- Pythagorean triples → Right triangle structure
- Primitive/Female classification → Complexity measure
- Fermat family (|C-F|=1) → Minimal proof paths
- mod-24 phase → Constraint satisfaction structure

**Result**: X% more problems solved, Y% fewer steps (vs baseline)

## Citation

```bibtex
@misc{qa_alphageometry2025,
  title={Discrete Harmonic Priors for Symbolic Geometry Reasoning},
  author={},
  year={2025},
  note={In preparation}
}
```

## License

MIT
