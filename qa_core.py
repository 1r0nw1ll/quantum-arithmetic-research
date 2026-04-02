QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""Compatibility shim for the extracted `qa_core/` package."""

from qa_core.engine import QAEngine, QA_Engine, QASystem
from qa_core.logger import (
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
from qa_core.metrics import e8_alignment, harmonic_index, harmonic_loss, qa_tuples
from qa_core.orbit import complete_graph_adjacency, neighbor_pull, resonance_matrix, weighted_adjacency

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
