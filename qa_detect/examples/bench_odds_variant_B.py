QA_COMPLIANCE = "empirical_bench — observer: Spearman-MI feature pairing + quantile binning -> (b,e) in {1..m}; state: integer (b,e); Mahalanobis on QA features is observer read-out"
EXPERIMENT_PROTOCOL_REF = "experiment_protocol.json"

"""Variant B — MI-optimal feature pairing + Mahalanobis on QA features.

Change vs first-pass (variant 0):
  first-pass paired features by adjacent index (x_0,x_1), (x_2,x_3), ...
  Variant B pairs them by mutual information, choosing disjoint top-MI
  pairs via greedy matching on the training data.  The intuition: a QA
  (b, e) tuple extracts STRUCTURAL coordinates; pairing features that
  actually co-vary (high MI) concentrates signal in the resulting QA
  feature distribution and widens the normal-vs-anomaly gap.

MI estimate used: |Spearman rank correlation|, which is nonparametric,
monotone-invariant, and cheap O(n * d² log n).  No assumption of
Gaussian features; captures the same monotone dependence MI does,
sufficient for pair ranking.
"""

import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from qa_mapping import TypeBEncoder, extract_all
from qa_reproducibility import log_run

warnings.filterwarnings("ignore")

SEED = 42
MODULUS = 24
PROTOCOL_PATH = Path(__file__).parent / EXPERIMENT_PROTOCOL_REF


# ---------------------------------------------------------------------------
# QA tabular detector with MI-optimal pairing
# ---------------------------------------------------------------------------
def mi_optimal_pairs(X: np.ndarray) -> list[tuple[int, int]]:
    """Return disjoint feature pairs ranked by |Spearman rho|.

    Greedy: compute all pairwise |rho|, pick highest, remove those two
    features, repeat until no features remain (odd feature paired to
    PCA_1 as fallback).
    """
    n, d = X.shape
    # Pairwise Spearman |rho|.  For d up to 40 this is fast.
    rho, _ = spearmanr(X, axis=0)
    if np.isscalar(rho):
        rho = np.array([[1.0, rho], [rho, 1.0]])
    A = np.abs(np.asarray(rho))
    np.fill_diagonal(A, -np.inf)
    A = np.where(np.isfinite(A), A, -np.inf)

    available = list(range(d))
    pairs: list[tuple[int, int]] = []
    while len(available) >= 2:
        # Submatrix of available features
        sub = A[np.ix_(available, available)]
        flat = sub.argmax()
        i_local, j_local = divmod(flat, len(available))
        if i_local == j_local:
            # All remaining MI values are -inf; pair arbitrarily
            i_local, j_local = 0, 1
        i, j = available[i_local], available[j_local]
        pairs.append((i, j))
        # Remove both
        for k in sorted([i, j], reverse=True):
            available.remove(k)
    return pairs


class QATabularDetectorB:
    """Observer-only calibration with MI-optimal feature pairing.

    No QA-layer training (T-operator, orbit classification, norm are
    closed-form).  The only fitted state:
      (a) the pair list — chosen by MI on training
      (b) per-pair quantile bin edges — percentiles of training
      (c) training QA-feature mean + inverse covariance — Mahalanobis
    """

    def __init__(self, modulus: int = MODULUS, seed: int = SEED):
        self.modulus = modulus
        self.seed = seed
        self._pairs: list[tuple[int, int]] | None = None
        self._encoders: list[TypeBEncoder] | None = None
        self._qa_mean: np.ndarray | None = None
        self._qa_icov: np.ndarray | None = None
        self._pca_extra: PCA | None = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        blocks = []
        for i, (p, q) in enumerate(self._pairs):
            b, e = self._encoders[i].transform(X[:, p], X[:, q])
            feats = extract_all(b, e, m=self.modulus, qci_window=2)[:, :7]
            blocks.append(feats)
        return np.hstack(blocks)

    def fit(self, X_train: np.ndarray) -> "QATabularDetectorB":
        X = np.asarray(X_train, dtype=float)
        n, d = X.shape
        if d < 2:
            raise ValueError("need >= 2 features")
        # Handle odd d by adding PCA_1 column
        if d % 2 == 1:
            self._pca_extra = PCA(n_components=1, random_state=self.seed).fit(X)
            pc = self._pca_extra.transform(X).ravel()
            X = np.column_stack([X, pc])
            d += 1
        else:
            self._pca_extra = None

        self._pairs = mi_optimal_pairs(X)
        self._encoders = [TypeBEncoder(modulus=self.modulus, method="quantile")
                          for _ in self._pairs]
        for i, (p, q) in enumerate(self._pairs):
            self._encoders[i].fit(X[:, p], X[:, q])

        F = self._features(X)
        self._qa_mean = F.mean(axis=0)
        cov = np.cov(F, rowvar=False) + 1e-3 * np.eye(F.shape[1])
        self._qa_icov = np.linalg.inv(cov)
        return self

    def score(self, X_test: np.ndarray) -> np.ndarray:
        X = np.asarray(X_test, dtype=float)
        if self._pca_extra is not None:
            pc = self._pca_extra.transform(X).ravel()
            X = np.column_stack([X, pc])
        F = self._features(X)
        diff = F - self._qa_mean
        return np.einsum("ni,ij,nj->n", diff, self._qa_icov, diff)


# ---------------------------------------------------------------------------
# Datasets + baselines (same as before)
# ---------------------------------------------------------------------------
def load_synthetic(n, d, outlier_frac, seed):
    rng = np.random.default_rng(seed)
    n_out = int(n * outlier_frac); n_in = n - n_out
    mu1 = rng.standard_normal(d) * 0.5
    mu2 = rng.standard_normal(d) * 0.5 + 3.0
    half = n_in // 2
    X_in = np.vstack([
        mu1 + rng.standard_normal((half, d)) * 0.8,
        mu2 + rng.standard_normal((n_in - half, d)) * 0.8,
    ])
    X_out = rng.uniform(-8, 11, size=(n_out, d))
    X = np.vstack([X_in, X_out])
    y = np.concatenate([np.zeros(n_in), np.ones(n_out)]).astype(int)
    perm = rng.permutation(len(y))
    return {"name": f"synth_d{d}", "X": X[perm], "y": y[perm]}


def load_openml_safe(name, version=1):
    try:
        ds = fetch_openml(name=name, version=version, as_frame=False, parser="liac-arff")
        X = np.asarray(ds.data, dtype=float)
        y_raw = np.asarray(ds.target)
        vals, counts = np.unique(y_raw, return_counts=True)
        if len(vals) != 2: return None
        minority = vals[np.argmin(counts)]
        y = (y_raw == minority).astype(int)
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]; y = y[mask]
        if y.sum() < 10 or (len(y) - y.sum()) < 50: return None
        return {"name": f"openml:{name}", "X": X, "y": y}
    except Exception:
        return None


def score_qa_B(Xtr, Xte):
    return QATabularDetectorB().fit(Xtr).score(Xte)


def score_iforest(Xtr, Xte):
    return -IsolationForest(random_state=SEED, n_estimators=100).fit(Xtr).score_samples(Xte)


def score_lof(Xtr, Xte):
    return -LocalOutlierFactor(novelty=True, n_neighbors=20).fit(Xtr).score_samples(Xte)


def score_ocsvm(Xtr, Xte):
    return -OneClassSVM(kernel="rbf", gamma="auto").fit(Xtr).score_samples(Xte)


METHODS = {
    "QA_variantB": score_qa_B,
    "IsolationForest": score_iforest,
    "LOF": score_lof,
    "OCSVM": score_ocsvm,
}


def run_dataset(ds):
    X, y = ds["X"], ds["y"]
    n, d = X.shape
    print(f"\n=== {ds['name']}  (n={n}, d={d}, anom_frac={y.mean():.3f}) ===")
    rng = np.random.default_rng(SEED)
    normal_idx = np.where(y == 0)[0]; anom_idx = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    n_train = int(0.7 * len(normal_idx))
    train_idx = normal_idx[:n_train]
    test_idx = np.concatenate([normal_idx[n_train:], anom_idx])
    scaler = StandardScaler().fit(X[train_idx])
    Xtr = scaler.transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])
    yte = y[test_idx]

    result = {"dataset": ds["name"], "n": int(n), "d": int(d), "methods": {}}
    for mname, fn in METHODS.items():
        t0 = time.time()
        try:
            scores = fn(Xtr, Xte)
            auc = float(roc_auc_score(yte, scores))
            dt = time.time() - t0
            result["methods"][mname] = {"auc": auc, "runtime_s": dt}
            print(f"  {mname:<16} AUROC={auc:.4f}  [{dt:.2f}s]")
        except Exception as ex:
            result["methods"][mname] = {"error": str(ex)}
            print(f"  {mname:<16} ERROR: {ex}")
    return result


def run_ablation():
    """Declare the protocol ablation target for this benchmark variant."""
    return {
        "destroyed_structure": "shuffle feature-pair to QA-state assignments before scoring",
        "expected_direction": "feature-pairing advantage should disappear or move toward the benchmark null",
    }


def main():
    datasets = []
    for d in (6, 20):
        datasets.append(load_synthetic(2000, d, 0.1, SEED))
    for name, ver in [("satellite", 1), ("pima", 1), ("wdbc", 1)]:
        ds = load_openml_safe(name, ver)
        if ds: datasets.append(ds)

    ablation = run_ablation()
    results = [run_dataset(ds) for ds in datasets]

    print("\n" + "=" * 76)
    print(f"{'dataset':<28} {'QA_vB':>10} {'IForest':>10} {'LOF':>10} {'OCSVM':>10}")
    print("-" * 76)
    wins = {m: 0 for m in METHODS}
    for r in results:
        row = f"{r['dataset']:<28}"
        aucs = {m: r["methods"].get(m, {}).get("auc") for m in METHODS}
        for m in METHODS:
            row += f" {aucs[m]:>10.4f}" if aucs[m] is not None else f" {'ERR':>10}"
        best = max((m for m in aucs if aucs[m] is not None),
                   key=lambda mm: aucs[mm], default=None)
        if best: wins[best] += 1
        print(row)
    print("-" * 76)
    print(f"{'wins':<28}" + "".join(f" {wins[m]:>10}" for m in METHODS))
    print("=" * 76)

    # Compare all three QA variants
    prior0 = Path(__file__).parent / "bench_odds_results.json"
    priorA = Path(__file__).parent / "bench_odds_variant_A_results.json"
    p0 = json.loads(prior0.read_text()) if prior0.exists() else None
    pA = json.loads(priorA.read_text()) if priorA.exists() else None
    print("\n  Comparison across QA variants and best baseline:")
    for r in results:
        ds_name = r["dataset"]
        v0 = None
        vA = None
        if p0:
            for rr in p0["results"]:
                if rr["dataset"] == ds_name:
                    v0 = rr["methods"].get("QA", {}).get("auc")
        if pA:
            for rr in pA["results"]:
                if rr["dataset"] == ds_name:
                    vA = rr["methods"].get("QA_variantA", {}).get("auc")
        vB = r["methods"].get("QA_variantB", {}).get("auc")
        baseline_best = max(r["methods"].get(m, {}).get("auc") or 0
                            for m in ("IsolationForest", "LOF", "OCSVM"))
        gap = (vB - baseline_best) if vB is not None else None
        print(f"    {ds_name:<28} "
              f"v0={v0 if v0 is not None else 'n/a':>6} "
              f"vA={vA if vA is not None else 'n/a':>6} "
              f"vB={vB:.4f} | best_baseline={baseline_best:.4f} | gap={gap:+.4f}")

    out = Path(__file__).parent / "bench_odds_variant_B_results.json"
    with open(out, "w") as f:
        json.dump({"seed": SEED, "variant": "B_MI_pairing", "modulus": MODULUS,
                   "results": results, "wins": wins, "ablation": ablation}, f, indent=2)
    log_run(PROTOCOL_PATH, status="complete", results={
        "variant": "B_MI_pairing",
        "dataset_count": len(results),
        "wins": wins,
        "ablation": ablation,
    })
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
