#!/usr/bin/env python3
"""
Tests for qa_hsi. Run directly: python3 -m qa_hsi.tests
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_hsi import QAHSITransformer
from qa_hsi._math import qa_packet, koenig_packet, qa_modular_residues, manhattan


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

def test_qa_packet_base():
    p = qa_packet(0, 0)
    assert p["b"] == 1 and p["e"] == 1
    assert p["d"] == 2 and p["a"] == 3
    assert p["C"] == 2 * 2 * 1 == 4
    assert p["F"] == 3 * 1 == 3
    assert p["G"] == 4 + 1 == 5
    assert p["I"] == abs(4 - 3) == 1
    assert p["H"] == 4 + 3 == 7

def test_qa_packet_symmetry():
    # qa_packet is NOT symmetric: b and e play different roles (e enters G, C, I)
    p1 = qa_packet(3, 5)   # b=4, e=6
    p2 = qa_packet(5, 3)   # b=6, e=4
    assert p1["b"] == p2["e"] and p1["e"] == p2["b"]   # coordinates swap
    assert p1["d"] == p2["d"]          # d = b+e is invariant under swap
    assert p1["G"] != p2["G"]          # G = d²+e² is NOT symmetric
    assert p1["C"] != p2["C"]          # C = 2*(b+e)*e is NOT symmetric
    assert p1["d"] == p1["b"] + p1["e"]  # definition holds

def test_qa_packet_i_formula():
    # I = |2e² - b²| where b = dist_left+1, e = dist_right+1
    for dl in range(5):
        for dr in range(5):
            p = qa_packet(dl, dr)
            b, e = dl + 1, dr + 1
            assert p["I"] == abs(2 * e * e - b * b), f"I formula failed at ({dl},{dr})"

def test_koenig_packet_identity():
    k = koenig_packet(1, 1)
    assert k["K_I"] == 1
    assert k["K_H"] == 7
    assert k["K_G"] == 5

def test_modular_residues():
    p = qa_packet(8, 0)  # b=9, e=1
    res = qa_modular_residues(p, moduli=(9,))
    assert res["b_mod9"] == p["b"] % 9
    assert res["H_mod9"] == p["H"] % 9

def test_manhattan():
    assert manhattan(0, 0, 3, 4) == 7
    assert manhattan(5, 5, 5, 5) == 0
    assert manhattan(0, 0, 0, 10) == 10


# ---------------------------------------------------------------------------
# Transformer API
# ---------------------------------------------------------------------------

def test_fit_transform_shape():
    rows = np.array([0, 5, 10, 15, 20])
    cols = np.array([0, 5, 10, 15, 20])
    labels = np.array([1, 2, 1, 2, 1])
    tr = QAHSITransformer(centroid_pairs=2, include_koenig=True)
    X = tr.fit_transform(rows, cols, labels, image_shape=(25, 25))
    assert X.shape[0] == 5
    assert X.shape[1] == tr.n_features_out
    assert X.dtype == np.float64

def test_feature_names_length():
    tr = QAHSITransformer(centroid_pairs=0, include_koenig=False, include_xy=False)
    tr.fit([0], [0], image_shape=(10, 10))
    names = tr.get_feature_names_out()
    assert len(names) == tr.n_features_out

def test_no_koenig():
    tr = QAHSITransformer(centroid_pairs=0, include_koenig=False)
    tr.fit([0, 5], [0, 5], image_shape=(10, 10))
    X = tr.transform([0, 5], [0, 5])
    assert X.shape[0] == 2

def test_include_xy():
    tr = QAHSITransformer(centroid_pairs=0, include_koenig=False, include_xy=True)
    tr.fit([0, 9], [0, 9], image_shape=(10, 10))
    X = tr.transform([0, 9], [0, 9])
    names = tr.get_feature_names_out()
    assert "xy_col_norm" in names and "xy_row_norm" in names
    # Corner pixel should have xy ≈ (0,0) and (1,1)
    assert abs(X[0, names.index("xy_col_norm")]) < 1e-9
    assert abs(X[1, names.index("xy_col_norm")] - 1.0) < 1e-9

def test_transform_before_fit_raises():
    tr = QAHSITransformer()
    try:
        tr.transform([0], [0])
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

def test_no_labels_centroid_zero():
    # centroid_pairs > 0 but no labels → falls back to corners only
    tr = QAHSITransformer(centroid_pairs=4)
    tr.fit([0, 5], [0, 5], labels=None, image_shape=(10, 10))
    assert len(tr._anchor_pairs) == 2  # corners only

def test_image_shape_inferred():
    rows, cols = [0, 19], [0, 29]
    tr = QAHSITransformer(centroid_pairs=0)
    tr.fit(rows, cols, image_shape=None)
    assert tr._image_shape == (20, 30)

def test_anchor_count():
    rows = list(range(20))
    cols = list(range(20))
    labels = [i % 4 + 1 for i in range(20)]  # 4 classes
    tr = QAHSITransformer(centroid_pairs=4)
    tr.fit(rows, cols, labels, image_shape=(20, 20))
    # 2 corners + up to 2 centroid pairs (4 classes → 2 pairs max with lo/hi pairing)
    assert len(tr._anchor_pairs) >= 2

def test_deterministic():
    rows = [0, 5, 10]
    cols = [0, 5, 10]
    labels = [1, 2, 1]
    tr = QAHSITransformer()
    tr.fit(rows, cols, labels, image_shape=(15, 15))
    X1 = tr.transform(rows, cols)
    X2 = tr.transform(rows, cols)
    np.testing.assert_array_equal(X1, X2)

def test_log_transform_positive():
    tr = QAHSITransformer(log_transform=True, centroid_pairs=0)
    tr.fit([0], [0], image_shape=(100, 100))
    X = tr.transform([0], [0])
    # log-transformed large-valued features should be finite and >= 0
    assert np.all(np.isfinite(X))
    assert np.all(X >= 0)


# ---------------------------------------------------------------------------
# Integration: synthetic benchmark reproducing expected OA lift direction
# ---------------------------------------------------------------------------

def test_spectral_plus_qa_beats_spectral():
    """
    On a synthetic spatial-structure classification problem,
    spectral+QA should outperform spectral-only.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(42)
    H, W = 50, 50
    n = 500

    rows   = rng.integers(0, H, n)
    cols   = rng.integers(0, W, n)
    labels = ((rows < H // 2) & (cols < W // 2)).astype(int) + \
             ((rows >= H // 2) & (cols >= W // 2)).astype(int) * 2 + 1

    spectral = rng.random((n, 10)).astype(np.float32)
    # Add spatial signal to spectral (mild)
    for i in range(n):
        spectral[i] += 0.3 * (labels[i] - 1)

    split = n * 3 // 4
    idx = rng.permutation(n)
    tr_idx, te_idx = idx[:split], idx[split:]

    tr = QAHSITransformer(centroid_pairs=2, include_koenig=True)
    tr.fit(rows[tr_idx], cols[tr_idx], labels[tr_idx], image_shape=(H, W))
    X_qa = tr.transform(rows, cols)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    rf.fit(spectral[tr_idx], labels[tr_idx])
    oa_spectral = accuracy_score(labels[te_idx], rf.predict(spectral[te_idx]))

    X_comb = np.column_stack([spectral, X_qa])
    rf.fit(X_comb[tr_idx], labels[tr_idx])
    oa_combined = accuracy_score(labels[te_idx], rf.predict(X_comb[te_idx]))

    print(f"\n  spectral OA     : {oa_spectral:.4f}")
    print(f"  spectral+QA OA  : {oa_combined:.4f}")
    print(f"  delta           : {oa_combined - oa_spectral:+.4f}")
    assert oa_combined >= oa_spectral, \
        f"spectral+QA ({oa_combined:.4f}) should be ≥ spectral ({oa_spectral:.4f})"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run():
    tests = [
        test_qa_packet_base,
        test_qa_packet_symmetry,
        test_qa_packet_i_formula,
        test_koenig_packet_identity,
        test_modular_residues,
        test_manhattan,
        test_fit_transform_shape,
        test_feature_names_length,
        test_no_koenig,
        test_include_xy,
        test_transform_before_fit_raises,
        test_no_labels_centroid_zero,
        test_image_shape_inferred,
        test_anchor_count,
        test_deterministic,
        test_log_transform_positive,
        test_spectral_plus_qa_beats_spectral,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("=" * 60)
    print("qa_hsi unit tests")
    print("=" * 60)
    ok = run()
    sys.exit(0 if ok else 1)
