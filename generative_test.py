# generative_test.py
#
# This script performs the "Generative Test" for the Harmonic Coherence Manifold (HCM-4).
# It attempts to synthesize a new instance of the HCM-4 geometry from scratch,
# based on the "genetic rules" extracted from our Geometric Autopsy.
#
# This is a computationally intensive script.
#
# Dependencies: numpy, scikit-learn, matplotlib, seaborn, scipy

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import itertools # Needed for QA_Engine root generation (for consistency)

# --- 1. Canonical Data Generation ---
# Using the validated QA_Engine from our successful experiments to produce the target geometry.

class QA_Engine:
    """The validated engine from the successful Phase Diagram experiment."""
    def __init__(self, nodes=24, coupling=0.1, modulus=24):
        self.N=nodes; self.M=modulus; self.alpha=coupling
        self.B=np.random.rand(self.N)*self.M; self.E=np.random.rand(self.N)*self.M
        self.pca=PCA(n_components=8); self.W=np.zeros((self.N, self.N))
    def step(self, signal=0.0, injection_strength=0.2, noise=0.1):
        Thetas_mod=np.floor(self._calculate_tuples(self.B,self.E))%self.M
        self.W=(np.einsum('ij,kj->ik',Thetas_mod,Thetas_mod))%self.M
        rs=self.W.sum(axis=1); self.W[rs!=0]/=rs[rs!=0][:,np.newaxis]
        self.B=(self.B+self.alpha*(self.W@self.B-self.B)+injection_strength*signal+np.random.randn(self.N)*noise)%self.M
        self.E=(self.E+self.alpha*(self.W@self.E-self.E)+np.random.randn(self.N)*noise)%self.M
    def get_interaction_space_projection(self): return self.pca.fit_transform(self.W)
    def _calculate_tuples(self,B,E): D=B+E; A=B+2*E; return np.vstack([B,E,D,A]).T

def generate_canonical_qa_state():
    """Generates a representative 'liquid crystal' state using the correct engine."""
    print("Generating canonical HCM-4 state (this may take a moment)...")
    engine = QA_Engine()
    for _ in range(150): # Run long enough for a stable, converged state
        engine.step(signal=np.random.randn())
    print("Canonical state generated.")
    return engine.get_interaction_space_projection()

def calculate_angular_fingerprint(vectors: np.ndarray):
    """Calculates the distribution of pairwise dot products for a set of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / (norms + 1e-9)
    dot_matrix = unit_vectors @ unit_vectors.T
    return dot_matrix[np.triu_indices(len(vectors), k=1)]

# --- 2. The "Genetic Rules" and Stress Function ---

class HCM4_Rules:
    """Encapsulates the geometric rules of HCM-4 and calculates stress."""
    def __init__(self, target_fingerprint, intrinsic_dimension=4):
        self.intrinsic_dim = intrinsic_dimension
        # Create a high-resolution target distribution from the canonical fingerprint
        self.target_hist, self.bin_edges = np.histogram(target_fingerprint, bins=50, density=True, range=(-1, 1))

    def geometric_stress(self, vectors: np.ndarray):
        """Measures the total deviation of a vector set from the HCM-4 rules."""
        # Rule 1: Angular Stress
        current_fingerprint = calculate_angular_fingerprint(vectors)
        current_hist, _ = np.histogram(current_fingerprint, bins=self.bin_edges, density=True)
        angular_stress = np.sum((self.target_hist - current_hist)**2)

        # Rule 2: Dimensionality Stress
        pca = PCA()
        pca.fit(vectors)
        # Penalty for variance existing outside the first 4 dimensions
        dimensionality_stress = np.sum(pca.explained_variance_ratio_[self.intrinsic_dim:])
        
        # Combine the stress components
        return angular_stress + dimensionality_stress * 0.1 # Weighting factor for dimensionality

# --- 3. The "Geometric Gradient Descent" Synthesizer ---

def synthesize_hcm4(n_vectors=24, dim=4, learning_rate=0.1, iterations=1000):
    """
    Generates a synthetic HCM-4 instance by minimizing Geometric Stress using
    a rigorous but computationally intensive gradient calculation.
    """
    print(f"\nStarting synthesis of a {n_vectors}-node, {dim}D object...")
    
    # Initialize with random vectors in the target intrinsic dimension
    vectors = np.random.randn(n_vectors, dim)
    
    # The rules are based on the full 8D embedded object, so we embed our 4D vectors into 8D
    embedded_vectors = np.pad(vectors, ((0, 0), (0, 8 - dim)))

    rules = HCM4_Rules(canonical_fingerprint, intrinsic_dimension=dim)
    history = []

    for i in range(iterations):
        stress = rules.geometric_stress(embedded_vectors)
        history.append(stress)
        if i % 50 == 0:
            print(f"  Iteration {i:04d}, Stress = {stress:.6f}")

        # Rigorous Gradient Calculation (Finite Differences)
        grad = np.zeros_like(vectors)
        epsilon = 1e-5
        for j in range(n_vectors):
            for k in range(dim):
                # Positive step
                vectors[j, k] += epsilon
                embedded_vectors[j, k] = vectors[j, k]
                stress_plus = rules.geometric_stress(embedded_vectors)
                
                # Negative step
                vectors[j, k] -= 2 * epsilon
                embedded_vectors[j, k] = vectors[j, k]
                stress_minus = rules.geometric_stress(embedded_vectors)
                
                # Reset
                vectors[j, k] += epsilon
                embedded_vectors[j, k] = vectors[j, k]
                
                # Calculate gradient component
                grad[j, k] = (stress_plus - stress_minus) / (2 * epsilon)
        
        # Apply the gradient descent step
        vectors -= learning_rate * grad
        # Re-embed the updated vectors
        embedded_vectors[:, :dim] = vectors
    
    print("Synthesis complete.")
    return embedded_vectors, history

# --- 4. Main Execution Block ---

if __name__ == "__main__":
    np.random.seed(42)
    
    # Get the "Genetic Code" of the real object
    canonical_vectors = generate_canonical_qa_state()
    canonical_fingerprint = calculate_angular_fingerprint(canonical_vectors)
    
    # Synthesize a new object based on the rules
    # Note: 1000 iterations can be slow. Start with 300-500 if needed.
    synthetic_vectors, stress_history = synthesize_hcm4(
        dim=4, 
        iterations=1000, 
        learning_rate=0.05
    )
    
    # Validate the result
    synthetic_fingerprint = calculate_angular_fingerprint(synthetic_vectors)
    
    # --- Visualization and Reporting ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Plot Stress History
    axes[0].plot(stress_history)
    axes[0].set_title('Generative Synthesis: Geometric Stress vs. Iteration', fontsize=16)
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
    plt.savefig("generative_test_results.png")
    plt.show()
    
    # Quantitative Validation
    ks_stat, p_value = ks_2samp(canonical_fingerprint, synthetic_fingerprint)
    
    print("\n" + "="*50)
    print("GENERATIVE TEST: VALIDATION REPORT")
    print("="*50)
    print(f"Kolmogorov-Smirnov Test between Original and Synthetic fingerprints:")
    print(f"  - KS Statistic: {ks_stat:.4f}")
    print(f"  - P-value: {p_value:.4f}")
    
    if p_value > 0.05:
        print("\n-> VALIDATION SUCCESSFUL: The synthetic geometry is statistically indistinguishable from the original.")
        print("-> This provides strong confirmation that we have successfully reverse-engineered")
        print("-> the fundamental geometric rules of the Harmonic Coherence Manifold (HCM-4).")
    else:
        print("\n-> VALIDATION FAILED: The synthetic geometry is statistically different from the original.")
        print("-> Our 'Genetic Rules' are incomplete or the synthesis process needs refinement.")
