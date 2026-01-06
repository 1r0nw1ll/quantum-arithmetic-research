from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ReflectionFailure:
    fail_type: str
    detail: str
    meta: Dict[str, Any]


# Optics-specific failure types (kept distinct from QARM core failures)
GEOM_DEGENERATE = "GEOM_DEGENERATE"   # mirror invalid / non-solvable parameterization
OOB = "OOB"                           # out of configured bounds (u range etc.)
ILLEGAL = "ILLEGAL"                   # generic legality failure (reserved)
INVARIANT = "INVARIANT"               # reserved for later (e.g., conservation claims)


def fail(fail_type: str, detail: str, **meta: Any) -> ReflectionFailure:
    return ReflectionFailure(fail_type=fail_type, detail=detail, meta=dict(meta))


def is_failure(x: Any) -> bool:
    return isinstance(x, ReflectionFailure)
