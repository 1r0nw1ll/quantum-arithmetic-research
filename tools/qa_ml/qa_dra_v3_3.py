"""QA-ML v3.3 E3 Define/Refine/Align pipeline.

Pepe's DRA pattern turns raw geometric observations into proposals,
refines those proposals with equivariant modules, then aligns/normalizes
them to one pose. The QA analog below keeps those stages explicit for
T1 failure-mode prediction:

* Define: fallback observer proposes a T1 class for every state.
* Refine: canonical-equivariant head proposes class 1 on the [277]
  canonical-signature regime.
* Align: route canonical_m == 15 through the refined canonical proposal,
  otherwise keep the fallback proposal.

QA_COMPLIANCE = "qa_ml_dra_v3_3 — integer canonical proposals; observer-layer fallback"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qa_equivariant_v3_3 import CanonicalEquivariantHybrid


@dataclass
class QADefineRefineAlign:
    """Interpretable staged wrapper around the E2 hybrid head."""

    hybrid: CanonicalEquivariantHybrid

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QADefineRefineAlign":
        self.hybrid.fit(X, y)
        return self

    def stage_predictions(self, X: np.ndarray) -> dict[str, np.ndarray]:
        fallback = np.asarray(self.hybrid.fallback_model.predict(X), dtype=np.int64)
        refined = fallback.copy()
        routed = np.zeros(X.shape[0], dtype=bool)
        canonical_proposal = np.zeros(X.shape[0], dtype=np.int64)

        for i, row in enumerate(X):
            if int(row[self.hybrid._canonical_m_idx]) != self.hybrid.route_canonical_m:
                continue
            routed[i] = True
            canonical_proposal[i] = (
                1 if self.hybrid._signature(row) in self.hybrid._positive_signatures else 0
            )
            refined[i] = canonical_proposal[i]

        return {
            "define_fallback": fallback,
            "refine_canonical": canonical_proposal,
            "align_output": refined,
            "routed_canonical": routed,
            "overrode_fallback": routed & (fallback != refined),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.stage_predictions(X)["align_output"]

    def alignment_summary(self, X: np.ndarray) -> dict[str, int]:
        stages = self.stage_predictions(X)
        return {
            "n_states": int(X.shape[0]),
            "routed_canonical": int(stages["routed_canonical"].sum()),
            "overrode_fallback": int(stages["overrode_fallback"].sum()),
            "fallback_class1_on_routed": int(
                ((stages["define_fallback"] == 1) & stages["routed_canonical"]).sum()
            ),
            "aligned_class1_on_routed": int(
                ((stages["align_output"] == 1) & stages["routed_canonical"]).sum()
            ),
        }
