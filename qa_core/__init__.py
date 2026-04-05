QA_COMPLIANCE = "observer=integer_coherence_field, state_alphabet=mod{1..m}, A1+S2_compliant"
"""Shared QA core package — integer-state coherence framework.

This package holds the signal_experiments QA framework (QAEngine, QASystem,
metrics, orbit helpers). It is distinct from `qa_lab/qa_core/` which is the
qa_lab kernel substrate (algebra + orbits). The two packages live in disjoint
scopes and are never imported together.

## Axiom compliance

- **A1 (No-Zero)**: all (b, e) state arrays hold integer values in {1, ..., m}.
  Initialization uses `np.random.randint(1, m+1, ...)`. Modular reduction is
  always `((x - 1) % m) + 1`.
- **S2 (No float state)**: `self.B`, `self.E` are `int64` arrays. Float arises
  only inside `step()` as a local observer-layer intermediate, quantized back
  to integer before being stored.
- **Theorem NT (Observer Projection Firewall)**: the continuous-signal input
  crosses the observer boundary exactly once per `step()` call (in), and the
  integer QA state is written exactly once per `step()` call (out). There is
  no float persistence across step boundaries — the float observer layer is
  re-materialized from the integer state at the start of every step.

## Public API (stable across the refactor)

Downstream consumers (17 root-level `run_signal_experiments*.py`,
`phase2_validation*.py`, `signal_experiment_*.py`, etc.) only touch:
  - `QASystem(num_nodes, modulus, coupling, noise_base, ...)`
  - `system.run_simulation(timesteps, signal_data, progress=...)`
  - `system.history` → dict of 'hi', 'e8_alignment', 'loss' arrays
  - `QAEngine(nodes, coupling, modulus)` + `engine.step(signal, ...)` +
    `engine.get_geometric_stress(...)`

None of those read `system.b` / `system.e` directly, so the integer-state
refactor is backwards-compatible at the API level. Numeric outputs may shift
slightly where the old (non-A1) reduction produced d=0 or a=0 at satellite
boundaries; the new reduction maps those to m instead.
"""

from .engine import QAEngine, QA_Engine, QASystem
from .logger import (
    QARunSummary,
    build_open_brain_capture,
    canonical_json_dumps,
    ensure_run_dir,
    final_metrics,
    new_history,
    record_history,
    utc_timestamp,
    write_json,
)
from .metrics import e8_alignment, harmonic_index, harmonic_loss, qa_tuples
from .orbit import complete_graph_adjacency, neighbor_pull, resonance_matrix, weighted_adjacency

__all__ = [
    "QAEngine",
    "QA_Engine",
    "QASystem",
    "QARunSummary",
    "build_open_brain_capture",
    "canonical_json_dumps",
    "complete_graph_adjacency",
    "e8_alignment",
    "ensure_run_dir",
    "final_metrics",
    "harmonic_index",
    "harmonic_loss",
    "neighbor_pull",
    "new_history",
    "qa_tuples",
    "record_history",
    "resonance_matrix",
    "utc_timestamp",
    "weighted_adjacency",
    "write_json",
]
