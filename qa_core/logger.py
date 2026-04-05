"""Structured history helpers for QA experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence


def new_history() -> dict[str, list[float]]:
    """Return the canonical in-memory metric history structure."""

    return {"loss": [], "e8_alignment": [], "hi": []}


def record_history(history: dict[str, list[float]], loss: float, e8_alignment: float, hi: float) -> None:
    """Append one metric step to a history mapping in-place."""

    history["loss"].append(float(loss))
    history["e8_alignment"].append(float(e8_alignment))
    history["hi"].append(float(hi))


@dataclass(frozen=True)
class QARunSummary:
    """Compact summary of the final state of a QA run."""

    steps: int
    final_loss: float
    final_e8_alignment: float
    final_hi: float


def final_metrics(history: dict[str, list[float]]) -> QARunSummary:
    """Return a typed summary from a populated history mapping."""

    if not history["loss"]:
        raise ValueError("history is empty")
    return QARunSummary(
        steps=len(history["loss"]),
        final_loss=float(history["loss"][-1]),
        final_e8_alignment=float(history["e8_alignment"][-1]),
        final_hi=float(history["hi"][-1]),
    )


def canonical_json_dumps(obj: Any) -> str:
    """Return canonical JSON for reproducible QA artifacts."""

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for result directories."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_dir(domain: str, script_name: str, *, root: str = "results", timestamp: str | None = None) -> Path:
    """Create and return the canonical results directory for a run."""

    run_dir = Path(root) / domain / script_name / (timestamp or utc_timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: Any) -> Path:
    """Write canonical JSON to disk and return the concrete path."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json_dumps(payload) + "\n", encoding="utf-8")
    return out_path


def build_open_brain_capture(
    capture_type: str,
    tags: Sequence[str],
    body: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an Open-Brain-ready capture payload."""

    payload: dict[str, Any] = {
        "type": capture_type,
        "tags": list(tags),
        "body": body,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload
