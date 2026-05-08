"""QA-ML — QA structural features for machine learning models.

QA_COMPLIANCE = "qa_ml_infra — observer-side feature extraction, A1/A2/S1/S2 compliant"

The feature extractor maps a QA discrete state (b, e) to a structural packet
(b, e, d, a, C, F, G) — and optionally to (b, e, d, a, C, F, G, phi_b, phi_e)
via qa_packet_full(b, e, m) — using raw integer arithmetic. The packet is
consumed by ML models as observer input — Theorem NT boundary is crossed once
(input -> observer); QA layer never receives float feedback.
"""

from .qa_features import (
    qa_packet,
    qa_packet_full,
    qa_packets,
    label,
    FEATURE_NAMES_RAW,
    FEATURE_NAMES_QA,
    FEATURE_NAMES_QA_FULL,
    ORBIT_LABELS,
    ORBIT_TO_INT,
)
from .qa_dataset import all_pairs, build_dataset

__all__ = [
    "qa_packet",
    "qa_packet_full",
    "qa_packets",
    "label",
    "FEATURE_NAMES_RAW",
    "FEATURE_NAMES_QA",
    "FEATURE_NAMES_QA_FULL",
    "ORBIT_LABELS",
    "ORBIT_TO_INT",
    "all_pairs",
    "build_dataset",
]
