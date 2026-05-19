"""QA-ML v3.3 E2 canonical-equivariant hybrid head.

The canonical head factors predictions through the gcd-quotient
representative (canonical_b, canonical_e, canonical_m). This enforces
scale equivariance for the [277] under-count regime without requiring the
observer packet to hand-materialize the full boundary predicate as one
feature.

QA_COMPLIANCE = "qa_ml_equivariant_v3_3 — integer canonical signatures; sklearn fallback observer"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CanonicalEquivariantHybrid:
    """Hybrid T1 predictor.

    States whose canonical representative has ``canonical_m == 15`` are
    routed through a canonical-signature head. The head predicts class 1
    for any canonical signature observed as class 1 during training, and
    class 0 otherwise. All other states are handled by the supplied
    fallback model.
    """

    feature_names: tuple[str, ...]
    fallback_model: object
    route_canonical_m: int = 15

    def __post_init__(self) -> None:
        idx = {name: i for i, name in enumerate(self.feature_names)}
        self._canonical_b_idx = idx["canonical_b"]
        self._canonical_e_idx = idx["canonical_e"]
        self._canonical_m_idx = idx["canonical_m"]
        self._positive_signatures: set[tuple[int, int, int]] = set()

    def _signature(self, row: np.ndarray) -> tuple[int, int, int]:
        return (
            int(row[self._canonical_b_idx]),
            int(row[self._canonical_e_idx]),
            int(row[self._canonical_m_idx]),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CanonicalEquivariantHybrid":
        self.fallback_model.fit(X, y)
        self._positive_signatures = {
            self._signature(row)
            for row, label in zip(X, y)
            if int(label) == 1 and int(row[self._canonical_m_idx]) == self.route_canonical_m
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        fallback_pred = np.asarray(self.fallback_model.predict(X), dtype=np.int64)
        out = fallback_pred.copy()
        for i, row in enumerate(X):
            if int(row[self._canonical_m_idx]) != self.route_canonical_m:
                continue
            out[i] = 1 if self._signature(row) in self._positive_signatures else 0
        return out

    def positive_signatures(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(sorted(self._positive_signatures))


def drop_features(
    X: np.ndarray,
    feature_names: Iterable[str],
    dropped: set[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return X restricted to features not listed in dropped."""
    kept_names = tuple(name for name in feature_names if name not in dropped)
    keep_mask = np.asarray([name not in dropped for name in feature_names], dtype=bool)
    return X[:, keep_mask], kept_names
