QA_COMPLIANCE = "library_module — surrogate validation framework"
"""Corrected surrogate validation for QA observer results.

Design principle: REAL targets held fixed, only QCI/orbit features surrogated.
This avoids the circular null problem where surrogates generate their own targets.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import stats

from qa_observer.orbits import qa_mod
from qa_arithmetic import orbit_family  # noqa: ORBIT-5

__all__ = ["SurrogateTest"]


def _phase_randomize(data: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """FFT each column, shared random phases, IFFT. Preserves spectrum."""
    n, d = data.shape
    result = np.zeros_like(data)
    freqs = np.fft.rfftfreq(n)
    rp = rng.uniform(0, 2 * np.pi, size=len(freqs))
    rp[0] = 0
    if n % 2 == 0:
        rp[-1] = 0
    for col in range(d):
        fv = np.fft.rfft(data[:, col])
        shifted = np.abs(fv) * np.exp(1j * (np.angle(fv) + rp))
        result[:, col] = np.fft.irfft(shifted, n=n)
    return result


def _ar1(data: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Fit AR(1) per column, generate synthetic."""
    n, d = data.shape
    result = np.zeros_like(data)
    for col in range(d):
        vals = data[:, col]
        phi = np.corrcoef(vals[:-1], vals[1:])[0, 1]
        sigma = np.std(vals) * np.sqrt(max(1 - phi * phi, 0.001))
        synth = np.zeros(n)
        synth[0] = rng.normal(0, np.std(vals))
        for t in range(1, n):
            synth[t] = phi * synth[t - 1] + rng.normal(0, sigma)
        result[:, col] = synth
    return result


def _block_shuffle(data: np.ndarray, rng: np.random.RandomState,
                   block_size: int = 21) -> np.ndarray:
    """Shuffle non-overlapping blocks."""
    n = data.shape[0]
    nb = n // block_size
    idx = np.arange(nb)
    rng.shuffle(idx)
    order = []
    for i in idx:
        order.extend(range(i * block_size, (i + 1) * block_size))
    rem = n - nb * block_size
    if rem > 0:
        order.extend(range(nb * block_size, n))
    return data[order]


GENERATORS = {
    "phase_randomized": _phase_randomize,
    "ar1": _ar1,
    "block_shuffled": _block_shuffle,
    "row_permuted": None,  # handled specially
}


class SurrogateTest:
    """Corrected process-level surrogate validation.

    Holds REAL targets fixed, generates surrogate QCI, compares.

    Example:
        from qa_observer import TopographicObserver, SurrogateTest

        obs = TopographicObserver(modulus=24, n_clusters=6)
        test = SurrogateTest(obs, n_surrogates=200)
        result = test.run(data, target, lagged_control, train_frac=0.5)
        print(result)  # {'real_r': ..., 'surrogates': {...}, 'n_pass': ...}
    """

    def __init__(self, observer, n_surrogates: int = 200,
                 surrogate_types: list = None, seed_base: int = 1000):
        self.observer = observer
        self.n_surrogates = n_surrogates
        self.surrogate_types = surrogate_types or [
            "phase_randomized", "ar1", "block_shuffled", "row_permuted"
        ]
        self.seed_base = seed_base

    def run(self, data: np.ndarray, target: np.ndarray,
            lagged_control: np.ndarray = None,
            train_frac: float = 0.5) -> dict:
        """Run full surrogate validation.

        Args:
            data: (n_samples, n_channels)
            target: (n_samples,) real target (held fixed for all surrogates)
            lagged_control: optional (n_samples,) for partial correlation
            train_frac: fraction for k-means training

        Returns dict with real result + per-surrogate-type comparison.
        """
        n = len(data)
        half = int(n * train_frac)

        # Real result
        real = self.observer.evaluate(data, target, lagged_control, train_frac)

        # Surrogates
        comparison = {}
        for st in self.surrogate_types:
            surr_r = []
            for i in range(self.n_surrogates):
                rng = np.random.RandomState(self.seed_base + i)

                if st == "row_permuted":
                    # Shuffle labels from real data
                    self.observer.fit(data[:half])
                    labs = self.observer.labels(data)
                    rng.shuffle(labs)
                    from qa_observer.core import QCI
                    qci_vals = QCI(
                        modulus=self.observer.modulus,
                        cmap=self.observer.cmap,
                        window=self.observer.qci_window,
                    ).compute(labs)
                else:
                    gen = GENERATORS[st]
                    surr_data = gen(data, rng)
                    # Compute surrogate QCI
                    surr_obs = type(self.observer)(
                        modulus=self.observer.modulus,
                        n_clusters=self.observer.n_clusters,
                        qci_window=self.observer.qci_window,
                        cmap=self.observer.cmap,
                        standardize_window=self.observer.standardize_window,
                        seed=self.observer.seed,
                    )
                    try:
                        surr_obs.fit(surr_data[:half])
                        qci_vals = surr_obs.transform(surr_data)
                    except Exception:
                        surr_r.append(np.nan)
                        continue

                # Correlate surrogate QCI with REAL target
                qci_full = np.full(n, np.nan)
                qci_full[: len(qci_vals)] = qci_vals
                oos = np.arange(n) >= half
                valid = oos & np.isfinite(qci_full) & np.isfinite(target)

                if valid.sum() < 30:
                    surr_r.append(np.nan)
                    continue

                try:
                    r, _ = stats.pearsonr(qci_full[valid], target[valid])
                    surr_r.append(r)
                except Exception:
                    surr_r.append(np.nan)

            vals = np.array(surr_r)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                comparison[st] = {"rank_p": 1.0, "beats": False,
                                  "surr_mean": np.nan, "surr_std": np.nan}
                continue

            real_r = real["raw_r"]
            rank_p = float(np.mean(np.abs(vals) >= np.abs(real_r)))
            comparison[st] = {
                "rank_p": rank_p,
                "beats": rank_p < 0.05,
                "surr_mean": float(np.mean(vals)),
                "surr_std": float(np.std(vals)),
                "n_valid": int(len(vals)),
            }

        n_pass = sum(1 for v in comparison.values() if v.get("beats", False))

        return {
            "real": real,
            "surrogates": comparison,
            "n_pass": n_pass,
            "n_types": len(self.surrogate_types),
            "tier3": n_pass >= len(self.surrogate_types) * 0.5,
        }
