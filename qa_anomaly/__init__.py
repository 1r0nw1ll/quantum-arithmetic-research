"""
qa_anomaly — Zero-parameter graph anomaly detection via QA discrete versor algebra.

QA coordinates: for a graph G with anchor nodes L, R,
  assign b = d(v, L) + 1,  e = d(v, R) + 1  to each node v.
  Edge (v, u) is a branch-type edge if Δb·Δe > 0 (both distances move same way).
  Edge (v, u) is a path-type edge if Δb·Δe < 0 (closer to one anchor, farther from other).

The qa_monotone_dir_score = count of branch-type edges incident to v.
  Branch body nodes score 2. Path interior nodes score 0. Anomaly threshold is implicit.

Benchmark: AUROC=0.8056 on 192 generated path-branch cases (no training, no parameters).

Usage
-----
    from qa_anomaly import QAGraphAnomalyDetector

    det = QAGraphAnomalyDetector()
    scores = det.score(G)           # networkx, edge list, or dict-of-lists
    top10  = det.top_k(G, k=10)     # [(node, score), ...]
"""

from .detector import QAGraphAnomalyDetector

__all__ = ["QAGraphAnomalyDetector"]
__version__ = "0.1.0"
