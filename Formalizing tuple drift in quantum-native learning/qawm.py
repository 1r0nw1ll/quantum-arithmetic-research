"""
QAWM (QA World Model) Implementation
Paper 2: Learning Topology in Discrete QA Control Systems

Uses scikit-learn MLPClassifier (no PyTorch dependency).
Implements shared encoder + multi-head architecture for:
  1. Legality prediction (binary)
  2. Fail type prediction (5-class)
  3. Return-in-k prediction (binary)

CRITICAL: Exact feature extraction preserving QA axioms from Paper 1.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from qa_oracle import QAState
from fractions import Fraction


# =============================================================================
# Feature Extraction (3-Bucket Strategy from ChatGPT Guidance)
# =============================================================================

def extract_state_features(state: QAState, N: int = 30) -> np.ndarray:
    """
    Extract features from QA state following 3-bucket strategy:
      - Bucket A: Small/raw features (normalized to [0,1])
      - Bucket B: Large invariants (log-scaled)
      - Bucket C: Rational features (L as num/den)

    Args:
        state: QAState with 21-element invariant packet
        N: Cap bound (for normalization)

    Returns:
        features: ndarray of shape (26,) or (128,) with padding

    CRITICAL CORRECTNESS:
      - I = |C - F| (absolute value)
      - W = X + K (canonical form)
      - h² as integer (no sqrt)
      - L as Fraction (exact num/den)
    """
    features = []

    # -------------------------------------------------------------------------
    # Bucket A: Small/Raw Features (6 features, normalized)
    # -------------------------------------------------------------------------
    features.append(state.b / N)            # b/N
    features.append(state.e / N)            # e/N
    features.append(state.d / (2 * N))      # d/(2N)
    features.append(state.a / (3 * N))      # a/(3N)
    features.append(state.phi_9 / 9.0)      # φ₉/9
    features.append(state.phi_24 / 24.0)    # φ₂₄/24

    # -------------------------------------------------------------------------
    # Bucket B: Large Invariants (18 features, log-scaled)
    # -------------------------------------------------------------------------
    large_invariants = [
        state.B,   # b²
        state.E,   # e²
        state.D,   # d²
        state.A,   # a²
        state.X,   # e·d
        state.C,   # 2·e·d
        state.F,   # b·a
        state.G,   # d² + e²
        state.H,   # C + F
        state.I,   # |C - F|  ✓ ABSOLUTE VALUE
        state.J,   # d·b
        state.K,   # d·a
        state.W,   # X + K  ✓ CANONICAL FORM
        state.Y,   # A - D
        state.Z,   # E + K
        state.h2,  # d²·a·b  ✓ EXACT INTEGER
        state.b * state.b + state.e * state.e,  # N (for compatibility)
        state.b + state.e + state.d + state.a   # S (sum invariant)
    ]

    for val in large_invariants:
        features.append(np.log1p(float(val)))  # log(1 + x)

    # -------------------------------------------------------------------------
    # Bucket C: Rational Features (2 features)
    # -------------------------------------------------------------------------
    # L = C·F/12 stored as exact Fraction
    L_num = state.L.numerator
    L_den = state.L.denominator

    features.append(np.log1p(abs(float(L_num))))  # log1p(|numerator|)
    features.append(np.log1p(float(L_den)))       # log1p(denominator)

    # -------------------------------------------------------------------------
    # Total: 6 + 18 + 2 = 26 features
    # -------------------------------------------------------------------------
    features = np.array(features, dtype=np.float64)

    # Pad to 128 if needed (for compatibility with larger models)
    if len(features) < 128:
        features = np.pad(features, (0, 128 - len(features)), 'constant')

    return features


def generator_to_index(gen: str) -> int:
    """Map generator name to index"""
    mapping = {
        'sigma': 0,
        'mu': 1,
        'lambda2': 2,
        'nu': 3
    }
    if gen not in mapping:
        raise ValueError(f"Unknown generator: {gen}")
    return mapping[gen]


def generator_to_onehot(gen: str) -> np.ndarray:
    """Map generator to one-hot encoding"""
    idx = generator_to_index(gen)
    onehot = np.zeros(4, dtype=np.float64)
    onehot[idx] = 1.0
    return onehot


# =============================================================================
# QAWM Configuration
# =============================================================================

@dataclass
class QAWMConfig:
    """Configuration for QA World Model"""
    state_dim: int = 128        # State feature dimension
    gen_dim: int = 4            # Generator one-hot dimension
    hidden_dim: int = 256       # Hidden layer size
    num_fail_types: int = 5     # Number of failure types

    # Training hyperparameters
    learning_rate: float = 1e-3
    max_iter: int = 200
    batch_size: int = 64
    random_state: int = 42


# =============================================================================
# QAWM Model (Scikit-Learn Implementation)
# =============================================================================

class QAWM:
    """
    QA World Model: Learns topology from sparse interaction data.

    Architecture (conceptual multi-head):
      1. Input: state features (128) + generator one-hot (4) → 132
      2. Shared encoder: 132 → 256 → 256 (MLP)
      3. Heads:
         - Legality: binary classifier
         - Fail type: 5-class classifier
         - Return-in-k: binary classifier (value predictor)

    Implementation:
      Since scikit-learn doesn't natively support multi-head,
      we train 3 separate MLPs sharing the same input encoding.
    """

    def __init__(self, config: QAWMConfig = None):
        if config is None:
            config = QAWMConfig()
        self.config = config

        # Import here to avoid issues if scikit-learn not installed
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        # Input scaler (fit on first batch)
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        # Three prediction heads (separate MLPs)
        self.legal_head = MLPClassifier(
            hidden_layer_sizes=(config.hidden_dim, config.hidden_dim),
            activation='relu',
            solver='adam',
            learning_rate_init=config.learning_rate,
            max_iter=config.max_iter,
            batch_size=config.batch_size,
            random_state=config.random_state,
            warm_start=True,  # Allow incremental training
            verbose=False
        )

        self.fail_type_head = MLPClassifier(
            hidden_layer_sizes=(config.hidden_dim, config.hidden_dim),
            activation='relu',
            solver='adam',
            learning_rate_init=config.learning_rate,
            max_iter=config.max_iter,
            batch_size=config.batch_size,
            random_state=config.random_state + 1,
            warm_start=True,
            verbose=False
        )

        self.return_head = MLPClassifier(
            hidden_layer_sizes=(config.hidden_dim, config.hidden_dim),
            activation='relu',
            solver='adam',
            learning_rate_init=config.learning_rate,
            max_iter=config.max_iter,
            batch_size=config.batch_size,
            random_state=config.random_state + 2,
            warm_start=True,
            verbose=False
        )

        # Track if models have been trained
        self.legal_fitted = False
        self.fail_type_fitted = False
        self.return_fitted = False

    def _prepare_input(self, state_features: np.ndarray,
                       gen_indices: np.ndarray) -> np.ndarray:
        """
        Combine state features + generator one-hot encoding.

        Args:
            state_features: (batch_size, 128)
            gen_indices: (batch_size,) integer indices

        Returns:
            combined: (batch_size, 132)
        """
        batch_size = state_features.shape[0]

        # Convert generator indices to one-hot
        gen_onehot = np.zeros((batch_size, 4), dtype=np.float64)
        gen_onehot[np.arange(batch_size), gen_indices] = 1.0

        # Concatenate
        combined = np.concatenate([state_features, gen_onehot], axis=1)

        # Scale features
        if not self.scaler_fitted:
            self.scaler.fit(combined)
            self.scaler_fitted = True

        combined = self.scaler.transform(combined)

        return combined

    def forward(self, state_features: np.ndarray,
                gen_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through all heads.

        Args:
            state_features: (batch_size, 128)
            gen_indices: (batch_size,)

        Returns:
            outputs: dict with keys:
              - 'legal_logits': (batch_size, 1) or probs
              - 'fail_type_logits': (batch_size, 5)
              - 'return_logits': (batch_size, 1) or probs
        """
        X = self._prepare_input(state_features, gen_indices)

        outputs = {}

        # Legality prediction
        if self.legal_fitted:
            legal_probs = self.legal_head.predict_proba(X)[:, 1]  # P(legal=1)
            outputs['legal_logits'] = legal_probs.reshape(-1, 1)
        else:
            outputs['legal_logits'] = np.ones((X.shape[0], 1)) * 0.5

        # Fail type prediction
        if self.fail_type_fitted:
            fail_probs = self.fail_type_head.predict_proba(X)  # (batch, 5)
            outputs['fail_type_logits'] = fail_probs
        else:
            outputs['fail_type_logits'] = np.ones((X.shape[0], 5)) / 5.0

        # Return-in-k prediction
        if self.return_fitted:
            return_probs = self.return_head.predict_proba(X)[:, 1]  # P(return=1)
            outputs['return_logits'] = return_probs.reshape(-1, 1)
        else:
            outputs['return_logits'] = np.ones((X.shape[0], 1)) * 0.5

        return outputs

    def __call__(self, state_features: np.ndarray,
                 gen_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Allow model(x, g) syntax"""
        return self.forward(state_features, gen_indices)

    def parameters(self):
        """Compatibility method (returns None for scikit-learn)"""
        return []


# =============================================================================
# Loss Function (Multi-Task)
# =============================================================================

class QAWMLoss:
    """
    Multi-task loss for QAWM:
      L = α·L_legal + β·L_fail + γ·L_return

    Where:
      - L_legal: Binary cross-entropy for legality
      - L_fail: Cross-entropy for fail type (on illegal moves only)
      - L_return: Binary cross-entropy for return-in-k
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 2.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, outputs: Dict[str, np.ndarray],
                 labels: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            outputs: dict from model forward pass
            labels: dict with 'legal', 'fail_type', 'return', 'illegal_mask'

        Returns:
            total_loss: scalar
            metrics: dict of individual losses
        """
        # Binary cross-entropy for legality
        legal_pred = outputs['legal_logits'].flatten()
        legal_true = labels['legal'].flatten()

        eps = 1e-7
        legal_pred = np.clip(legal_pred, eps, 1 - eps)
        loss_legal = -np.mean(
            legal_true * np.log(legal_pred) +
            (1 - legal_true) * np.log(1 - legal_pred)
        )

        # Cross-entropy for fail type (only on illegal moves)
        illegal_mask = labels['illegal_mask']
        if np.sum(illegal_mask) > 0:
            fail_pred = outputs['fail_type_logits'][illegal_mask]
            fail_true = labels['fail_type'][illegal_mask]

            fail_pred = np.clip(fail_pred, eps, 1 - eps)
            # One-hot encode fail_true
            fail_true_onehot = np.zeros_like(fail_pred)
            fail_true_onehot[np.arange(len(fail_true)), fail_true] = 1.0

            loss_fail = -np.mean(np.sum(fail_true_onehot * np.log(fail_pred), axis=1))
        else:
            loss_fail = 0.0

        # Binary cross-entropy for return-in-k (only on labeled samples)
        return_mask = labels['return'] != -1
        if np.sum(return_mask) > 0:
            return_pred = outputs['return_logits'].flatten()[return_mask]
            return_true = labels['return'][return_mask]

            return_pred = np.clip(return_pred, eps, 1 - eps)
            loss_return = -np.mean(
                return_true * np.log(return_pred) +
                (1 - return_true) * np.log(1 - return_pred)
            )
        else:
            loss_return = 0.0

        # Total loss
        total_loss = (
            self.alpha * loss_legal +
            self.beta * loss_fail +
            self.gamma * loss_return
        )

        metrics = {
            'loss/total': total_loss,
            'loss/legal': loss_legal,
            'loss/fail_type': loss_fail,
            'loss/return': loss_return
        }

        return total_loss, metrics


# =============================================================================
# Example Usage / Testing
# =============================================================================

if __name__ == "__main__":
    from qa_oracle import construct_qa_state

    print("=" * 70)
    print("QAWM Feature Extraction Test")
    print("=" * 70)

    # Test feature extraction on known state
    state = construct_qa_state(b=3, e=4)

    print(f"\nTest state: (b={state.b}, e={state.e})")
    print(f"Derived: d={state.d}, a={state.a}")
    print(f"Invariants: C={state.C}, F={state.F}, I={state.I}, W={state.W}")
    print(f"h²={state.h2}, L={state.L}")

    # Extract features
    features = extract_state_features(state, N=30)

    print(f"\nFeature vector shape: {features.shape}")
    print(f"First 10 features: {features[:10]}")
    print(f"Non-zero features: {np.sum(features != 0)}")

    # Check correctness
    assert features[0] == 3/30, "b/N feature incorrect"
    assert features[1] == 4/30, "e/N feature incorrect"
    assert np.isfinite(features).all(), "NaN or Inf in features"

    print("\n✅ Feature extraction test passed")

    # Test generator encoding
    print("\n" + "=" * 70)
    print("Generator Encoding Test")
    print("=" * 70)

    for gen in ['sigma', 'mu', 'lambda2', 'nu']:
        idx = generator_to_index(gen)
        onehot = generator_to_onehot(gen)
        print(f"{gen:8s} → index={idx}, one-hot={onehot}")

    print("\n✅ Generator encoding test passed")

    # Test model initialization
    print("\n" + "=" * 70)
    print("QAWM Model Initialization Test")
    print("=" * 70)

    config = QAWMConfig()
    model = QAWM(config)

    print(f"Config: state_dim={config.state_dim}, hidden_dim={config.hidden_dim}")
    print(f"Heads: legal={model.legal_fitted}, fail={model.fail_type_fitted}, return={model.return_fitted}")

    print("\n✅ Model initialization test passed")
    print("\n" + "=" * 70)
    print("All tests passed! qawm.py is ready for training.")
    print("=" * 70)
