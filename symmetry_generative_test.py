# symmetry_generative_test.py
#
# This script performs the definitive "Generative Test" for the HCM-4 manifold.
# It uses an augmented set of "genetic rules" that includes the newly discovered
# reflectional symmetries to guide the synthesis.
#
# This is a computationally intensive script.
#
# Dependencies: numpy, scikit-learn, matplotlib, seaborn, scipy

import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import itertools

# --- 1. Canonical Data Generation ---
from qa_core import QAEngine as QA_Engine

def generate_canonical_qa_state():
    """Generates a representative 'liquid crystal' state."""
    print("Generating canonical HCM-4 state...")
    engine = QA_Engine()
    for _ in range(150):
        engine.step(signal=np.random.randn())
    # Project interaction matrix to 8D using PCA (was engine method previously)
    pca = PCA(n_components=8)
    state = pca.fit_transform(engine.W)
    # Center the data
    return state - np.mean(state, axis=0)

def calculate_angular_fingerprint(vectors: np.ndarray):
    """Calculates the distribution of pairwise dot products."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / (norms + 1e-9)
    dot_matrix = unit_vectors @ unit_vectors.T
    return dot_matrix[np.triu_indices(len(vectors), k=1)]

# --- 2. The Augmented "Genetic Rules" (Version 2) ---

class HCM4_Rules_V2:
    """Encapsulates the geometric rules of HCM-4, now including symmetry."""
    def __init__(self, target_fingerprint, intrinsic_dimension=4):
        self.intrinsic_dim = intrinsic_dimension
        self.target_hist, self.bin_edges = np.histogram(target_fingerprint, bins=50, density=True, range=(-1, 1))
        
        # Define the symmetry transformation matrices
        self.R6 = np.identity(8)
        self.R6[6, 6] = -1
        self.R7 = np.identity(8)
        self.R7[7, 7] = -1

    def get_rmsd(self, set1, set2):
        """Calculates RMSD after optimal assignment."""
        distance_matrix = cdist(set1, set2)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        min_distances = distance_matrix[row_ind, col_ind]
        return np.sqrt(np.mean(min_distances**2))

    def geometric_stress(self, vectors: np.ndarray):
        """Measures the total deviation from all known HCM-4 rules."""
        # Rule 1: Angular Stress
        current_fingerprint = calculate_angular_fingerprint(vectors)
        current_hist, _ = np.histogram(current_fingerprint, bins=self.bin_edges, density=True)
        angular_stress = np.sum((self.target_hist - current_hist)**2)

        # Rule 2: Dimensionality Stress
        pca = PCA()
        pca.fit(vectors)
        dimensionality_stress = np.sum(pca.explained_variance_ratio_[self.intrinsic_dim:])
        
        # Rule 3: Symmetry Stress (The New Ingredient)
        reflected_6 = vectors @ self.R6.T
        reflected_7 = vectors @ self.R7.T
        rmsd_6 = self.get_rmsd(vectors, reflected_6)
        rmsd_7 = self.get_rmsd(vectors, reflected_7)
        symmetry_stress = rmsd_6**2 + rmsd_7**2
        
        # Combine stress components with weighting
        return angular_stress + (dimensionality_stress * 0.1) + (symmetry_stress * 1.0)

# --- 3. The "Symmetry-Guided" Synthesizer ---

def synthesize_hcm4_symmetry_guided(n_vectors=24, dim=8, learning_rate=0.1, iterations=1000, momentum=0.9):
    """
    Generates a synthetic HCM-4 instance using a more robust optimizer
    guided by the full set of geometric rules including symmetry.
    """
    print(f"\nStarting SYMMETRY-GUIDED synthesis of a {n_vectors}-node, {dim}D object...")
    
    vectors = np.random.randn(n_vectors, dim) * 0.1 # Start near origin
    velocity = np.zeros_like(vectors)

    rules = HCM4_Rules_V2(canonical_fingerprint, intrinsic_dimension=4)
    history = []

    for i in range(iterations):
        stress = rules.geometric_stress(vectors)
        history.append(stress)
        if i % 25 == 0:
            print(f"  Iteration {i:04d}, Stress = {stress:.6f}")

        # Rigorous Gradient Calculation (Finite Differences)
        grad = np.zeros_like(vectors)
        epsilon = 1e-5
        for j in range(n_vectors):
            for k in range(dim):
                vectors[j, k] += epsilon
                stress_plus = rules.geometric_stress(vectors)
                vectors[j, k] -= 2 * epsilon
                stress_minus = rules.geometric_stress(vectors)
                vectors[j, k] += epsilon
                grad[j, k] = (stress_plus - stress_minus) / (2 * epsilon)
        
        # Update with momentum
        velocity = (momentum * velocity) - (learning_rate * grad)
        vectors += velocity
    
    print("Synthesis complete.")
    return vectors, history

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symmetry-guided HCM-4 synthesis")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test (few iterations)")
    parser.add_argument("--iterations", type=int, default=None, help="Override number of iterations")
    args = parser.parse_args()
    np.random.seed(42)
    
    # Get the "Genetic Code" of the real object
    canonical_vectors = generate_canonical_qa_state()
    canonical_fingerprint = calculate_angular_fingerprint(canonical_vectors)
    
    # Synthesize a new object with the new rules
    # Note: 500 iterations is a reasonable balance for local execution.
    # Increase to 1000-2000 for a higher fidelity result if time permits.
    quick = args.quick or (os.environ.get("QUICK") is not None)
    iters = args.iterations if args.iterations is not None else (10 if quick else 500)
    lr = 0.05
    synthetic_vectors, stress_history = synthesize_hcm4_symmetry_guided(
        iterations=iters,
        learning_rate=lr
    )
    
    # Validate the result
    synthetic_fingerprint = calculate_angular_fingerprint(synthetic_vectors)
    
    # --- Visualization and Reporting ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Plot Stress History
    axes[0].plot(stress_history)
    axes[0].set_title('Symmetry-Guided Synthesis: Geometric Stress vs. Iteration', fontsize=16)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Geometric Stress')
    axes[0].set_yscale('log')
    axes[0].grid(True, linestyle=':')
    
    # Plot Fingerprint Comparison
    sns.kdeplot(canonical_fingerprint, ax=axes[1], color='blue', linewidth=3, fill=True, alpha=0.1, label='Original HCM-4 (from QA-Engine)')
    sns.kdeplot(synthetic_fingerprint, ax=axes[1], color='green', linewidth=3, linestyle='--', label='Synthesized HCM-4')
    axes[1].set_title('Validation: Comparing Angular Fingerprints', fontsize=16)
    axes[1].set_xlabel('Pairwise Dot Product')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig("symmetry_generative_test_results.png")
    plt.show()
    
    # Quantitative Validation
    ks_stat, p_value = ks_2samp(canonical_fingerprint, synthetic_fingerprint)
    
    print("\n" + "="*50)
    print("SYMMETRY-GUIDED GENERATIVE TEST: VALIDATION REPORT")
    print("="*50)
    print(f"Kolmogorov-Smirnov Test between Original and Synthetic fingerprints:")
    print(f"  - KS Statistic: {ks_stat:.4f}")
    print(f"  - P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("\n-> VALIDATION SUCCESSFUL: The synthetic geometry is statistically indistinguishable from the original.")
        print("-> This provides strong confirmation that we have successfully reverse-engineered")
        print("-> the fundamental geometric and symmetric rules of the Harmonic Coherence Manifold (HCM-4).")
    else:
        print("\n-> VALIDATION FAILED: The synthetic geometry is statistically different from the original.")
        print("-> The 'Genetic Rules' are still incomplete or the synthesis process requires further tuning.")
