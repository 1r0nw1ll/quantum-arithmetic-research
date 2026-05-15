"""QA-ML v3 dataset — per-state features + T1 (failure mode) + T2b (period class)
labels across a list of moduli.

T1 (shortcut failure mode, 3 classes):
  0 — correct                  : canonical == shortcut
  1 — undercount_missed_sat    : canonical == "satellite" and shortcut != "satellite"
  2 — overclaim_false_sat      : shortcut  == "satellite" and canonical != "satellite"

T2b (orbit period class, 5 classes):
  0 — period 1   (singularity)
  1 — period 4   (5-factor sub-cycle, per O3)
  2 — period 8   (satellite class)
  3 — period equals max_period of this modulus (Pisano-like peak)
  4 — other

The dataset is deterministic given the modulus list; no stochastic
generators (Theorem NT design rule).

QA_COMPLIANCE = "qa_ml_dataset_v3 — exhaustive integer enumeration; ground truth from qa_orbit_rules"
"""

from __future__ import annotations

from collections import Counter

from qa_orbit_rules import (
    orbit_family,
    orbit_family_divisor_shortcut,
    orbit_period,
)

from .qa_features_v3 import qa_packet_v3, FEATURE_NAMES_V3


T1_CLASSES = ("correct", "undercount_missed_sat", "overclaim_false_sat")
T1_TO_INT = {n: i for i, n in enumerate(T1_CLASSES)}

T2B_CLASSES = ("period_1", "period_4", "period_8", "period_max", "period_other")
T2B_TO_INT = {n: i for i, n in enumerate(T2B_CLASSES)}


def t1_label(b: int, e: int, m: int) -> int:
    """3-class T1 failure-mode label."""
    canonical = orbit_family(b, e, m)
    shortcut = orbit_family_divisor_shortcut(b, e, m)
    if canonical == shortcut:
        return 0
    if canonical == "satellite" and shortcut != "satellite":
        return 1
    if shortcut == "satellite" and canonical != "satellite":
        return 2
    # Disagreement that's not the satellite axis (shouldn't occur given
    # current classifier definitions, but guard anyway).
    return 0


def max_period_for_m(m: int) -> int:
    """Compute max orbit period across all (b, e) in {1..m}^2 for modulus m."""
    return max(orbit_period(b, e, m) for b in range(1, m + 1) for e in range(1, m + 1))


def t2b_label(b: int, e: int, m: int, max_p: int) -> int:
    """5-class T2b period class label (max_p precomputed per m)."""
    p = orbit_period(b, e, m)
    if p == 1:
        return T2B_TO_INT["period_1"]
    if p == 4:
        return T2B_TO_INT["period_4"]
    if p == 8:
        return T2B_TO_INT["period_8"]
    if p == max_p:
        return T2B_TO_INT["period_max"]
    return T2B_TO_INT["period_other"]


def build_v3_dataset(moduli: list[int]) -> dict:
    """Build a full v3 dataset across the given moduli.

    Returns a dict with:
      X        — list of per-state feature tuples (23-int v3 packet)
      y_t1     — list of T1 (3-class failure mode) labels
      y_t2b    — list of T2b (5-class period class) labels
      y_period — list of exact period values (for T2a if needed)
      triples  — list of (b, e, m) triples in order
      moduli   — copy of input moduli
      max_period — dict m -> max orbit period for that m
      class_counts_t1 — dict for sanity
      class_counts_t2b — dict for sanity
    """
    X = []
    y_t1 = []
    y_t2b = []
    y_period = []
    triples = []
    max_period = {}

    for m in moduli:
        max_p = max_period_for_m(m)
        max_period[m] = max_p
        for b in range(1, m + 1):
            for e in range(1, m + 1):
                X.append(qa_packet_v3(b, e, m))
                y_t1.append(t1_label(b, e, m))
                y_t2b.append(t2b_label(b, e, m, max_p))
                y_period.append(orbit_period(b, e, m))
                triples.append((b, e, m))

    return {
        "X": X,
        "y_t1": y_t1,
        "y_t2b": y_t2b,
        "y_period": y_period,
        "triples": triples,
        "moduli": list(moduli),
        "max_period": max_period,
        "feature_names": list(FEATURE_NAMES_V3),
        "t1_classes": list(T1_CLASSES),
        "t2b_classes": list(T2B_CLASSES),
        "class_counts_t1": dict(Counter(y_t1)),
        "class_counts_t2b": dict(Counter(y_t2b)),
    }
