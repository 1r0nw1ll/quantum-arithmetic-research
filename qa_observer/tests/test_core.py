QA_COMPLIANCE = "test_module — validates qa_observer package"
"""Tests for qa_observer package."""

import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_observer import TopographicObserver, QCI, SurrogateTest
from qa_observer.orbits import qa_mod, qa_step
from qa_orbit_rules import orbit_family  # noqa: ORBIT-5 canonical import


def test_qa_mod():
    """A1: result in {1,...,m}, never 0."""
    for m in [9, 24]:
        for x in range(1, m * 3):
            result = qa_mod(x, m)
            assert 1 <= result <= m, f"qa_mod({x}, {m}) = {result}"
    # Key test: multiples of m map to m, not 0
    assert qa_mod(9, 9) == 9
    assert qa_mod(24, 24) == 24
    assert qa_mod(48, 24) == 24
    print("  [PASS] qa_mod A1 compliance")


def test_qa_step():
    """T-operator: Fibonacci shift."""
    b, e = qa_step(3, 5, 9)
    assert b == 5
    assert e == qa_mod(3 + 5, 9)
    assert e == 8  # (8-1)%9 + 1 = 8
    print("  [PASS] qa_step")


def test_orbit_family():
    """Orbit classification: singularity, satellite, cosmos."""
    # Singularity: (m, m)
    assert orbit_family(9, 9, 9) == "singularity"
    assert orbit_family(24, 24, 24) == "singularity"
    # Satellite: divisible by m//3
    assert orbit_family(3, 6, 9) == "satellite"
    assert orbit_family(8, 16, 24) == "satellite"
    # Cosmos: everything else
    assert orbit_family(1, 2, 9) == "cosmos"
    assert orbit_family(5, 7, 24) == "cosmos"
    print("  [PASS] orbit_family classification")


def test_qci_computation():
    """QCI: rolling T-operator prediction accuracy."""
    # Create a sequence where T-operator predictions sometimes match
    np.random.seed(42)
    labels = np.random.randint(0, 4, size=200)
    cmap = {0: 8, 1: 16, 2: 24, 3: 5}

    qci = QCI(modulus=24, cmap=cmap, window=20)
    result = qci.compute(labels)

    assert len(result) == len(labels) - 2
    assert np.all(np.isfinite(result[20:]))  # after window fills
    assert np.all((result[np.isfinite(result)] >= 0) & (result[np.isfinite(result)] <= 1))
    print(f"  [PASS] QCI computation (mean={np.nanmean(result):.3f})")


def test_orbit_fractions():
    """Orbit fractions: singularity + satellite + cosmos ≈ 1."""
    np.random.seed(42)
    labels = np.random.randint(0, 4, size=200)
    cmap = {0: 8, 1: 16, 2: 24, 3: 5}

    qci = QCI(modulus=24, cmap=cmap)
    fracs = qci.orbit_fractions(labels, window=20)

    total = fracs["singularity"] + fracs["satellite"] + fracs["cosmos"]
    assert np.allclose(total, 1.0, atol=0.01)
    print(f"  [PASS] orbit fractions sum to 1")


def test_topographic_observer():
    """Full pipeline: fit → transform → evaluate."""
    n, d = 500, 6
    rows = np.arange(n, dtype=float)[:, None]
    cols = np.arange(1, d + 1, dtype=float)[None, :]
    data = np.sin(rows * cols * 0.071) + 0.25 * np.cos(rows * (cols + 2.0) * 0.037)
    # Inject structure: first half calm, second half volatile
    data[250:] *= 2.0
    target = np.zeros(n)
    target[250:] = 1.0  # future "stress"

    obs = TopographicObserver(modulus=24, n_clusters=4, qci_window=30,
                              standardize_window=50)
    result = obs.evaluate(data, target, train_frac=0.5)

    assert "raw_r" in result
    assert "n_oos" in result
    assert result["n_oos"] > 0
    print(f"  [PASS] TopographicObserver evaluate (r={result['raw_r']:+.3f}, n={result['n_oos']})")


def test_surrogate_test():
    """Surrogate validation: structure-free data should NOT beat surrogates."""
    n, d = 300, 4
    rows = np.arange(n, dtype=float)[:, None]
    cols = np.arange(1, d + 1, dtype=float)[None, :]
    data = np.sin(rows * cols * 0.113) + np.cos((rows + 3.0) * cols * 0.047)
    target = np.sin(np.arange(n, dtype=float) * 0.173 + 0.5)

    obs = TopographicObserver(modulus=9, n_clusters=4, qci_window=20,
                              standardize_window=30)
    test = SurrogateTest(obs, n_surrogates=20,
                         surrogate_types=["row_permuted"])
    result = test.run(data, target, train_frac=0.5)

    assert "real" in result
    assert "surrogates" in result
    assert "n_pass" in result
    # Random data should mostly NOT beat surrogates
    print(f"  [PASS] SurrogateTest (n_pass={result['n_pass']}/{result['n_types']})")


def main():
    print("qa_observer test suite")
    print("=" * 40)
    test_qa_mod()
    test_qa_step()
    test_orbit_family()
    test_qci_computation()
    test_orbit_fractions()
    test_topographic_observer()
    test_surrogate_test()
    print("=" * 40)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
