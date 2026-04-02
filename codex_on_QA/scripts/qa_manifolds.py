#!/usr/bin/env python3
"""
QA manifold + tabular demos (self-contained, CPU-only, reproducible).

Outputs under codex_on_QA/out:
  - moons_logreg_curve.png
  - moons_mlp_epochs.png
  - moons_kmeans_scores.txt
  - swiss_pca_raw.png
  - swiss_pca_qa.png
  - swiss_kmeans_scores.txt
  - tabular_sgd_epochs.png (grid/raw vs qa vs both)

Requires: numpy, scikit-learn, matplotlib, seaborn
Run:
  python codex_on_QA/scripts/qa_manifolds.py
"""

from __future__ import annotations
import os, math, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_swiss_roll, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

OUT = os.path.join('codex_on_QA','out')
os.makedirs(OUT, exist_ok=True)

RNG = np.random.RandomState(0)
np.random.seed(0)

def qa_per_feature(X: np.ndarray) -> np.ndarray:
    """QA-like per-feature mapping based on canonical tuple relations.
    For each column x, derive (b,e,d,a) via robust magnitudes and dispersion:
      - b = |z|
      - e = |z - median(z)|
      - d = b + e
      - a = b + 2e
    Emit a small invariant set: [J,X,K,C,F,G] per feature.
    """
    X = np.asarray(X)
    n, d = X.shape
    feats = []
    for j in range(d):
        x = X[:, j]
        # z-score to stabilize scale
        mu = x.mean()
        sd = x.std() + 1e-9
        z = (x - mu) / sd
        med = np.median(z)
        b = np.abs(z)
        e = np.abs(z - med)
        dval = b + e + 1e-9
        aval = b + 2*e + 1e-9
        J = b * dval
        Xinv = e * dval
        K = dval * aval
        C = 2.0 * Xinv
        F = b * aval
        G = e**2 + dval**2
        feats.append(np.stack([J, Xinv, K, C, F, G], axis=1))
    return np.concatenate(feats, axis=1)

def moons_demo():
    X, y = make_moons(n_samples=1500, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Logistic Regression accuracy vs train size (raw vs QA)
    sizes = [50, 100, 200, 400, 700, len(Xtr)]
    acc_raw, acc_qa = [], []
    for m in sizes:
        idx = RNG.choice(len(Xtr), size=m, replace=False)
        Xt_m, yt_m = Xtr[idx], ytr[idx]
        # raw
        clf = LogisticRegression(max_iter=200, random_state=0)
        clf.fit(Xt_m, yt_m)
        acc_raw.append(accuracy_score(yte, clf.predict(Xte)))
        # qa
        Xtqa = qa_per_feature(Xt_m)
        Xeqa = qa_per_feature(Xte)
        clfq = LogisticRegression(max_iter=200, random_state=0)
        clfq.fit(Xtqa, yt_m)
        acc_qa.append(accuracy_score(yte, clfq.predict(Xeqa)))

    plt.figure(figsize=(5,4))
    plt.plot(sizes, acc_raw, 'o-', label='raw')
    plt.plot(sizes, acc_qa, 's-', label='qa')
    plt.xlabel('train size'); plt.ylabel('accuracy'); plt.title('Moons: LogReg accuracy vs train size')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'moons_logreg_curve.png'), dpi=160)
    plt.close()

    # Tiny MLP via SGDClassifier (logistic) accuracy vs epochs (raw vs QA)
    epochs = list(range(1, 21))
    acc_e_raw, acc_e_qa = [], []
    sgd_raw = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=0)
    sgd_qa  = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=0)
    # Standardize raw
    ss_raw = StandardScaler().fit(Xtr)
    Xtr_s = ss_raw.transform(Xtr); Xte_s = ss_raw.transform(Xte)
    # QA features
    Xtr_q = qa_per_feature(Xtr); Xte_q = qa_per_feature(Xte)
    ss_q = StandardScaler().fit(Xtr_q)
    Xtr_qs = ss_q.transform(Xtr_q); Xte_qs = ss_q.transform(Xte_q)
    classes = np.unique(ytr)
    for ep in epochs:
        sgd_raw.partial_fit(Xtr_s, ytr, classes=classes)
        sgd_qa.partial_fit(Xtr_qs, ytr, classes=classes)
        acc_e_raw.append(accuracy_score(yte, sgd_raw.predict(Xte_s)))
        acc_e_qa.append(accuracy_score(yte, sgd_qa.predict(Xte_qs)))
    plt.figure(figsize=(5,4))
    plt.plot(epochs, acc_e_raw, 'o-', label='raw')
    plt.plot(epochs, acc_e_qa, 's-', label='qa')
    plt.xlabel('epochs'); plt.ylabel('accuracy'); plt.title('Moons: SGD (logistic) accuracy vs epoch')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'moons_mlp_epochs.png'), dpi=160)
    plt.close()

    # KMeans ARI/NMI (raw vs QA)
    km_raw = KMeans(n_clusters=2, n_init=20, random_state=0)
    km_qa  = KMeans(n_clusters=2, n_init=20, random_state=0)
    lab_raw = km_raw.fit_predict(X)
    lab_qa  = km_qa.fit_predict(qa_per_feature(X))
    ari_raw = adjusted_rand_score(y, lab_raw)
    nmi_raw = normalized_mutual_info_score(y, lab_raw)
    ari_qa  = adjusted_rand_score(y, lab_qa)
    nmi_qa  = normalized_mutual_info_score(y, lab_qa)
    with open(os.path.join(OUT, 'moons_kmeans_scores.txt'), 'w') as f:
        f.write(json.dumps({
            'raw': {'ARI': ari_raw, 'NMI': nmi_raw},
            'qa':  {'ARI': ari_qa,  'NMI': nmi_qa}
        }, indent=2))

def swiss_demo():
    X, t = make_swiss_roll(n_samples=2000, noise=0.15, random_state=0)
    # Bucket t into 5 bins as pseudo-labels
    bins = np.quantile(t, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(t, bins)
    # KMeans on raw vs QA
    km_raw = KMeans(n_clusters=5, n_init=20, random_state=0)
    km_qa  = KMeans(n_clusters=5, n_init=20, random_state=0)
    lab_raw = km_raw.fit_predict(X)
    lab_qa  = km_qa.fit_predict(qa_per_feature(X))
    ari_raw = adjusted_rand_score(y, lab_raw)
    nmi_raw = normalized_mutual_info_score(y, lab_raw)
    ari_qa  = adjusted_rand_score(y, lab_qa)
    nmi_qa  = normalized_mutual_info_score(y, lab_qa)
    with open(os.path.join(OUT, 'swiss_kmeans_scores.txt'), 'w') as f:
        f.write(json.dumps({
            'raw': {'ARI': ari_raw, 'NMI': nmi_raw},
            'qa':  {'ARI': ari_qa,  'NMI': nmi_qa}
        }, indent=2))
    # PCA visualizations
    pca = PCA(n_components=2, random_state=0)
    Xp_raw = pca.fit_transform(StandardScaler().fit_transform(X))
    Xp_qa  = pca.fit_transform(StandardScaler().fit_transform(qa_per_feature(X)))
    plt.figure(figsize=(5,4)); plt.scatter(Xp_raw[:,0], Xp_raw[:,1], s=2, c=y, cmap='viridis')
    plt.title('Swiss roll (raw PCA)'); plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'swiss_pca_raw.png'), dpi=160); plt.close()
    plt.figure(figsize=(5,4)); plt.scatter(Xp_qa[:,0], Xp_qa[:,1], s=2, c=y, cmap='viridis')
    plt.title('Swiss roll (QA PCA)'); plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'swiss_pca_qa.png'), dpi=160); plt.close()

def tabular_sgd_demo():
    # Synthetic tabular classification
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, n_redundant=4,
                               n_clusters_per_class=2, class_sep=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xtr_q, Xte_q = qa_per_feature(Xtr), qa_per_feature(Xte)
    # Prepare three views: raw, qa, both
    ss_raw = StandardScaler().fit(Xtr)
    ss_qa  = StandardScaler().fit(Xtr_q)
    Xtr_raw, Xte_raw = ss_raw.transform(Xtr), ss_raw.transform(Xte)
    Xtr_qa,  Xte_qa  = ss_qa.transform(Xtr_q), ss_qa.transform(Xte_q)
    Xtr_both = np.hstack([Xtr_raw, Xtr_qa]); Xte_both = np.hstack([Xte_raw, Xte_qa])
    # SGD (logistic) curves over epochs
    epochs = list(range(1, 21))
    classes = np.unique(ytr)
    def train_curve(Xtrv, Xtev):
        clf = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=0)
        acc = []
        for ep in epochs:
            clf.partial_fit(Xtrv, ytr, classes=classes)
            acc.append(accuracy_score(yte, clf.predict(Xtev)))
        return acc
    acc_raw = train_curve(Xtr_raw, Xte_raw)
    acc_qa  = train_curve(Xtr_qa,  Xte_qa)
    acc_both= train_curve(Xtr_both,Xte_both)
    plt.figure(figsize=(5,4))
    plt.plot(epochs, acc_raw, 'o-', label='raw')
    plt.plot(epochs, acc_qa,  's-', label='qa')
    plt.plot(epochs, acc_both,'^-', label='raw+qa')
    plt.xlabel('epochs'); plt.ylabel('accuracy'); plt.title('Tabular: SGD (logistic) accuracy vs epoch')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'tabular_sgd_epochs.png'), dpi=160)
    plt.close()

def main():
    sns.set_context('talk'); sns.set_style('whitegrid')
    moons_demo()
    swiss_demo()
    tabular_sgd_demo()
    print('Saved figures and metrics to', OUT)

if __name__ == '__main__':
    main()

