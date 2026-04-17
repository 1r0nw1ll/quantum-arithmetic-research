"""Unit tests for tools.qa_kg.analysis.metrics.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import math

import pytest

from tools.qa_kg.analysis import metrics


# gold_hit_at_k ---------------------------------------------------------

def test_gold_hit_present():
    assert metrics.gold_hit_at_k(["a", "b", "c"], ["c"], k=5) is True


def test_gold_hit_absent():
    assert metrics.gold_hit_at_k(["a", "b", "c"], ["z"], k=5) is False


def test_gold_hit_beyond_k():
    assert metrics.gold_hit_at_k(["a", "b", "c", "d", "e", "GOLD"], ["GOLD"], k=5) is False


def test_gold_hit_empty_gold_is_vacuous_true():
    assert metrics.gold_hit_at_k(["a"], [], k=5) is True


# graph_structural_recall_at_k -----------------------------------------

def test_recall_full_hit():
    assert metrics.graph_structural_recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0


def test_recall_partial():
    r = metrics.graph_structural_recall_at_k(["a", "b"], ["a", "b", "c", "d"], k=5)
    assert r == 0.5


def test_recall_capped_by_k():
    # 20-item gold, only 5 can fit in top-5.
    gold = [f"g{i}" for i in range(20)]
    results = gold[:5] + ["noise"]
    r = metrics.graph_structural_recall_at_k(results, gold, k=5)
    assert r == 5 / 20


# graph_structural_pass_rate -------------------------------------------

def test_pass_rate_all_hit():
    per_q = [
        {"result_ids": ["a"], "gold_ids": ["a"]},
        {"result_ids": ["b"], "gold_ids": ["b"]},
    ]
    assert metrics.graph_structural_pass_rate(per_q, k=5) == 1.0


def test_pass_rate_partial():
    per_q = [
        {"result_ids": ["a"], "gold_ids": ["a"]},
        {"result_ids": ["b"], "gold_ids": ["zzz"]},
        {"result_ids": ["c"], "gold_ids": ["c"]},
    ]
    assert abs(metrics.graph_structural_pass_rate(per_q, k=5) - 2 / 3) < 1e-9


# contradiction_recall_per_pair ----------------------------------------

def test_contradiction_both_endpoints_hit():
    per_pair = [{"result_ids": ["x", "src1", "y", "dst1"], "pair": ["src1", "dst1"]}]
    assert metrics.contradiction_recall_per_pair(per_pair, k=5) == 1.0


def test_contradiction_only_one_endpoint_counts_as_miss():
    per_pair = [{"result_ids": ["src1", "x", "y", "z", "w", "dst1"], "pair": ["src1", "dst1"]}]
    assert metrics.contradiction_recall_per_pair(per_pair, k=5) == 0.0


# authority_presence_at_k ----------------------------------------------

def test_authority_presence_in_top_3():
    per_q = [
        {"result_ids": ["primary1", "x", "y"], "expected_primary_id": "primary1"},
        {"result_ids": ["x", "y", "primary2"], "expected_primary_id": "primary2"},
        {"result_ids": ["x", "y", "z", "primary3"], "expected_primary_id": "primary3"},
    ]
    assert abs(metrics.authority_presence_at_k(per_q, k=3) - 2 / 3) < 1e-9


# ndcg_at_k ------------------------------------------------------------

def test_ndcg_perfect_order():
    # grades = {a: 5, b: 3, c: 1}
    # ideal order = a, b, c → ndcg = 1.0 when predicted matches
    grades = {"a": 5, "b": 3, "c": 1}
    assert abs(metrics.ndcg_at_k(["a", "b", "c"], grades, k=3) - 1.0) < 1e-9


def test_ndcg_reversed_order():
    grades = {"a": 5, "b": 3, "c": 1}
    reversed_score = metrics.ndcg_at_k(["c", "b", "a"], grades, k=3)
    assert 0.0 < reversed_score < 1.0


def test_ndcg_empty_grades():
    assert metrics.ndcg_at_k(["a"], {}, k=5) == 0.0


# factor_dominance -----------------------------------------------------

def test_factor_dominance_flat_factors():
    """All factors = 1.0 → zero dominance (score = 1, no lifting)."""
    bd = {k: 1.0 for k in metrics._FACTOR_KEYS}
    out = metrics.factor_dominance([bd])
    assert out["dominated_fraction"] == 0.0
    assert all(v == 0.0 for v in out["mean_share_by_factor"].values())


def test_factor_dominance_authority_dominant():
    """Authority=10 alone, rest=1 → authority holds 100% share."""
    bd = {"authority": 10.0, "lifecycle": 1.0, "bm25_norm": 1.0,
          "confidence": 1.0, "time_decay": 1.0, "contradiction": 1.0,
          "prov_decay": 1.0}
    out = metrics.factor_dominance([bd], dominance_threshold=0.80)
    assert out["dominated_fraction"] == 1.0
    assert out["mean_share_by_factor"]["authority"] == 1.0


def test_factor_dominance_mixed():
    """authority=10, bm25=0.5, prov=0.5 — no single factor > 80%."""
    bd = {"authority": 10.0, "lifecycle": 1.0, "bm25_norm": 0.5,
          "confidence": 1.0, "time_decay": 1.0, "contradiction": 1.5,
          "prov_decay": 0.5}
    out = metrics.factor_dominance([bd], dominance_threshold=0.80)
    # authority log(10)≈2.30 is still the biggest, check shares
    auth_share = out["mean_share_by_factor"]["authority"]
    assert 0.0 < auth_share < 1.0
    assert auth_share < 0.80  # mixed factors keep it under the threshold


# authority_monotonicity ----------------------------------------------

def test_authority_monotone_consistent():
    per_q = [{"ranked_hits": [
        {"authority": "primary", "score": 10.0},
        {"authority": "derived", "score": 8.0},
        {"authority": "internal", "score": 5.0},
    ]}]
    assert metrics.authority_monotonicity(per_q) == 1.0


def test_authority_monotone_large_inversion():
    per_q = [{"ranked_hits": [
        {"authority": "agent", "score": 10.0},
        {"authority": "primary", "score": 5.0},
    ]}]
    assert metrics.authority_monotonicity(per_q, epsilon=0.5) == 0.0


def test_authority_monotone_small_inversion_within_tolerance():
    per_q = [{"ranked_hits": [
        {"authority": "derived", "score": 5.1},
        {"authority": "primary", "score": 5.0},  # inversion, but diff < 0.5 ε
    ]}]
    assert metrics.authority_monotonicity(per_q, epsilon=0.5) == 1.0


# provenance_spearman --------------------------------------------------

def test_provenance_spearman_positive_corr():
    # rank 1 → depth 1 (shallow), rank 2 → depth 2, rank 3 → depth 3
    per_q = [{"ranked_hits": [
        {"rank": 1, "provenance_depth": 1},
        {"rank": 2, "provenance_depth": 2},
        {"rank": 3, "provenance_depth": 3},
    ]}]
    assert abs(metrics.provenance_spearman(per_q) - 1.0) < 1e-9


def test_provenance_spearman_all_noaxiom_skipped():
    """All depths = -1 → query skipped."""
    per_q = [{"ranked_hits": [
        {"rank": 1, "provenance_depth": -1},
        {"rank": 2, "provenance_depth": -1},
        {"rank": 3, "provenance_depth": -1},
    ]}]
    assert metrics.provenance_spearman(per_q) == 0.0


# lifecycle_ordering_pass ---------------------------------------------

def test_lifecycle_v4_above_all_superseded():
    per_q = [{
        "result_ids": ["cert_v4", "cert_v3", "cert_v2", "cert_v1"],
        "gold_ids": ["cert_v4"],
        "anti_gold_ids": ["cert_v1", "cert_v2", "cert_v3"],
        "k": 5,
    }]
    assert metrics.lifecycle_ordering_pass(per_q) == 1.0


def test_lifecycle_v4_below_some_superseded_fails():
    per_q = [{
        "result_ids": ["cert_v3", "cert_v4", "cert_v2", "cert_v1"],
        "gold_ids": ["cert_v4"],
        "anti_gold_ids": ["cert_v1", "cert_v2", "cert_v3"],
        "k": 5,
    }]
    assert metrics.lifecycle_ordering_pass(per_q) == 0.0
