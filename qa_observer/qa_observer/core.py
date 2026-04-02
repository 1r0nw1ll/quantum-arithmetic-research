QA_COMPLIANCE = "library_module — observer pipeline implementation"
"""Core QA observer pipeline: TopographicObserver and QCI computation.

Usage:
    from qa_observer import TopographicObserver

    obs = TopographicObserver(modulus=24, n_clusters=6, qci_window=63)
    obs.fit(train_data)          # train_data: (n_samples, n_channels)
    qci = obs.transform(data)    # returns QCI series
    result = obs.evaluate(data, target, lagged_control)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats
from numpy.linalg import lstsq as np_lstsq

from qa_observer.orbits import qa_mod
from qa_orbit_rules import orbit_family  # noqa: ORBIT-5 canonical import

__all__ = ["TopographicObserver", "QCI"]


# Default cluster-to-QA-state mappings
DEFAULT_CMAPS = {
    4: {0: 8, 1: 16, 2: 24, 3: 5},
    6: {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11},
}


def _make_cmap(k: int) -> dict:
    """Generate a CMAP ensuring orbit diversity for K clusters."""
    if k in DEFAULT_CMAPS:
        return DEFAULT_CMAPS[k]
    sat_states = [8, 16, 24]
    other_states = [3, 5, 11, 7, 13, 19, 1, 17]
    pool = sat_states + other_states
    return {i: pool[i % len(pool)] for i in range(k)}


class QCI:
    """Compute QA Coherence Index from a label sequence."""

    def __init__(self, modulus: int = 24, cmap: dict = None, window: int = 63):
        self.modulus = modulus
        self.cmap = cmap or DEFAULT_CMAPS.get(6)
        self.window = window

    def compute(self, labels: np.ndarray) -> np.ndarray:
        """Compute QCI from integer cluster labels.

        Returns array of length len(labels)-2, with NaN where
        the rolling window hasn't filled.
        """
        m = self.modulus
        cmap = self.cmap
        w = self.window

        t_match = []
        for t in range(len(labels) - 2):
            b = cmap.get(int(labels[t]), 5)
            e = cmap.get(int(labels[t + 1]), 5)
            actual = cmap.get(int(labels[t + 2]), 5)
            pred = qa_mod(b + e, m)
            t_match.append(1 if pred == actual else 0)

        series = pd.Series(t_match)
        return series.rolling(w, min_periods=w // 2).mean().values

    def orbit_fractions(self, labels: np.ndarray, window: int = 20):
        """Compute rolling orbit fractions (singularity, satellite, cosmos).

        Returns dict of arrays, each of length len(labels)-window.
        """
        m = self.modulus
        cmap = self.cmap
        n = len(labels)

        sing, sat, cos_ = [], [], []
        for i in range(n - window):
            seg = labels[i:i + window]
            orbits = []
            for j in range(len(seg) - 1):
                b = cmap.get(int(seg[j]), 1)
                e = cmap.get(int(seg[j + 1]), 1)
                orbits.append(orbit_family(int(b), int(e), m))
            n_orb = len(orbits)
            sing.append(sum(1 for o in orbits if o == "singularity") / n_orb)
            sat.append(sum(1 for o in orbits if o == "satellite") / n_orb)
            cos_.append(sum(1 for o in orbits if o == "cosmos") / n_orb)

        return {
            "singularity": np.array(sing),
            "satellite": np.array(sat),
            "cosmos": np.array(cos_),
        }


class TopographicObserver:
    """Full QA topographic observer pipeline.

    Implements: signal → standardize → k-means → QA states → QCI/orbits.

    Example:
        obs = TopographicObserver(modulus=24, n_clusters=6, qci_window=63)
        obs.fit(train_data)
        qci = obs.transform(all_data)
        result = obs.evaluate(all_data, target, lagged_control, train_frac=0.5)
    """

    def __init__(
        self,
        modulus: int = 24,
        n_clusters: int = 6,
        qci_window: int = 63,
        cmap: dict = None,
        standardize_window: int = 252,
        seed: int = 42,
    ):
        self.modulus = modulus
        self.n_clusters = n_clusters
        self.qci_window = qci_window
        self.cmap = cmap or _make_cmap(n_clusters)
        self.standardize_window = standardize_window
        self.seed = seed

        self._km = None
        self._qci = QCI(modulus=modulus, cmap=self.cmap, window=qci_window)

    def fit(self, data: np.ndarray):
        """Fit k-means on training data.

        Args:
            data: (n_samples, n_channels) array. Will be standardized internally.
        """
        std = self._standardize(data)
        self._km = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=self.seed
        )
        self._km.fit(std)
        return self

    def labels(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for data."""
        if self._km is None:
            raise RuntimeError("Call fit() first")
        std = self._standardize(data)
        return self._km.predict(std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Compute QCI for data. Returns array of length n_samples-2."""
        lab = self.labels(data)
        return self._qci.compute(lab)

    def orbit_features(self, data: np.ndarray, window: int = 20) -> dict:
        """Compute orbit fraction features for data."""
        lab = self.labels(data)
        return self._qci.orbit_fractions(lab, window=window)

    def evaluate(
        self,
        data: np.ndarray,
        target: np.ndarray,
        lagged_control: np.ndarray = None,
        train_frac: float = 0.5,
    ) -> dict:
        """Full evaluation: compute QCI, correlate with target OOS.

        Args:
            data: (n_samples, n_channels)
            target: (n_samples,) — the thing to predict (e.g. future vol)
            lagged_control: (n_samples,) — baseline to partial out (e.g. lagged vol)
            train_frac: fraction of data used for k-means training

        Returns dict with raw_r, partial_r, p-values, n_oos.
        """
        n = len(data)
        half = int(n * train_frac)

        self.fit(data[:half])
        qci = self.transform(data)

        # Align: QCI has length n-2
        qci_full = np.full(n, np.nan)
        qci_full[: len(qci)] = qci

        # OOS mask
        oos = np.arange(n) >= half
        valid = oos & np.isfinite(qci_full) & np.isfinite(target)

        if valid.sum() < 30:
            return {"raw_r": np.nan, "raw_p": np.nan,
                    "partial_r": np.nan, "partial_p": np.nan, "n_oos": 0}

        qci_oos = qci_full[valid]
        tgt_oos = target[valid]

        raw_r, raw_p = stats.pearsonr(qci_oos, tgt_oos)

        result = {"raw_r": float(raw_r), "raw_p": float(raw_p),
                  "n_oos": int(valid.sum())}

        # Partial correlation if control provided
        if lagged_control is not None:
            ctrl_oos = lagged_control[valid]
            ctrl_valid = np.isfinite(ctrl_oos)
            if ctrl_valid.sum() >= 30:
                X = np.column_stack([ctrl_oos[ctrl_valid],
                                     np.ones(ctrl_valid.sum())])
                qci_r = qci_oos[ctrl_valid] - X @ np_lstsq(X, qci_oos[ctrl_valid], rcond=None)[0]
                tgt_r = tgt_oos[ctrl_valid] - X @ np_lstsq(X, tgt_oos[ctrl_valid], rcond=None)[0]
                pr, pp = stats.pearsonr(qci_r, tgt_r)
                result["partial_r"] = float(pr)
                result["partial_p"] = float(pp)
            else:
                result["partial_r"] = np.nan
                result["partial_p"] = np.nan
        else:
            result["partial_r"] = np.nan
            result["partial_p"] = np.nan

        return result

    def _standardize(self, data: np.ndarray) -> np.ndarray:
        """Rolling z-score standardization."""
        df = pd.DataFrame(data)
        w = self.standardize_window
        rm = df.rolling(w, min_periods=w // 2).mean()
        rs = df.rolling(w, min_periods=w // 2).std() + 1e-10
        std = ((df - rm) / rs).dropna().values
        return std
