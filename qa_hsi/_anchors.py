"""Anchor selection strategies for HSI grids."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

AnchorPair = Tuple[Tuple[int, int], Tuple[int, int]]


def corner_anchors(image_shape: Tuple[int, int]) -> List[AnchorPair]:
    """Two diagonal corner pairs covering the full image extent."""
    H, W = image_shape
    return [
        ((0, 0),     (H - 1, W - 1)),
        ((0, W - 1), (H - 1, 0)),
    ]


def centroid_anchors(
    rows: Sequence[int],
    cols: Sequence[int],
    labels: Sequence[int],
    *,
    max_pairs: int = 4,
) -> List[AnchorPair]:
    """
    Pair class centroids from opposite ends of the sorted class list.
    Uses only the pixel coordinates supplied (typically training pixels).
    Returns up to max_pairs pairs.
    """
    by_label: Dict[int, List[Tuple[int, int]]] = {}
    for r, c, lbl in zip(rows, cols, labels):
        by_label.setdefault(int(lbl), []).append((int(r), int(c)))

    # Sort by descending count so the largest classes anchor first
    sorted_classes = sorted(by_label.items(), key=lambda kv: -len(kv[1]))
    centroids = []
    for _, points in sorted_classes:
        cr = round(sum(p[0] for p in points) / len(points))
        cc = round(sum(p[1] for p in points) / len(points))
        centroids.append((int(cr), int(cc)))

    pairs: List[AnchorPair] = []
    lo, hi = 0, len(centroids) - 1
    while lo < hi and len(pairs) < max_pairs:
        pairs.append((centroids[lo], centroids[hi]))
        lo += 1
        hi -= 1
    return pairs
