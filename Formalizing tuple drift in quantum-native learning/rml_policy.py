"""
RML (Reachability Meta-Learning) Policy Implementation
Paper 3: Meta-Policy Learning Over QA Manifolds

Implements:
  1. Random-Legal baseline
  2. Oracle-Greedy baseline (upper bound)
  3. QAWM-Greedy (key result - no learning)
  4. RML policy (lightweight bandit REINFORCE)

CRITICAL: Policies query STRUCTURAL PREDICTIONS, not dynamics.
This is NOT reinforcement learning - it's control via learned structure.
"""

import numpy as np
import joblib
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from sklearn.neural_network import MLPClassifier

from qa_oracle import QAOracle, QAState, construct_qa_state
from qawm import extract_state_features, generator_to_index


# =============================================================================
# Task Definition
# =============================================================================

@dataclass
class RMLTask:
    """Standard task for Paper 3 evaluation"""
    target_class: Set[QAState]  # Goal states
    horizon: int                 # Maximum steps k
    generators: List[str]        # Available generators

    @classmethod
    def diagonal_task(cls, N: int, k: int = 10):
        """Standard task: reach diagonal {(b,b)} in k steps"""
        target_class = set()
        for b in range(1, N + 1):
            target_class.add(construct_qa_state(b, b))

        return cls(
            target_class=target_class,
            horizon=k,
            generators=['sigma', 'mu', 'lambda2', 'nu']
        )


# =============================================================================
# Baseline 1: Random-Legal
# =============================================================================

class RandomLegalPolicy:
    """
    Baseline: Choose uniformly among legal generators.

    Oracle calls per step: 4 (one legality check per generator)
    Expected performance: Poor (no guidance toward target)
    """

    def __init__(self, oracle: QAOracle, generators: List[str]):
        self.oracle = oracle
        self.generators = generators
        self.oracle_calls = 0

    def select_generator(self, state: QAState) -> Optional[str]:
        """Select random legal generator"""
        legal_gens = []

        for g in self.generators:
            self.oracle_calls += 1  # Track oracle usage
            if self.oracle.is_legal(state, g):
                legal_gens.append(g)

        if len(legal_gens) == 0:
            return None  # Stuck (no legal moves)

        return np.random.choice(legal_gens)

    def reset_oracle_counter(self):
        """Reset oracle call counter"""
        self.oracle_calls = 0


# =============================================================================
# Baseline 2: Oracle-Greedy (Upper Bound)
# =============================================================================

class OracleGreedyPolicy:
    """
    Upper bound: Use true oracle return-in-k to pick best generator.

    Oracle calls per step: ~8-12 (legality + return-in-k queries)
    Expected performance: Near-optimal (uses ground truth)
    """

    def __init__(self, oracle: QAOracle, generators: List[str]):
        self.oracle = oracle
        self.generators = generators
        self.oracle_calls = 0

    def select_generator(self, state: QAState, task: RMLTask) -> Optional[str]:
        """Select generator with best return-in-k"""
        best_g = None
        best_reachable = False

        gen_list = [(g, 2) for g in self.generators]

        for g in self.generators:
            self.oracle_calls += 1  # Legality check

            if self.oracle.is_legal(state, g):
                # Execute move
                next_state = self.oracle.step(state, g)

                # Check return-in-k from next state
                self.oracle_calls += 1  # BFS query (expensive)
                reachable = self.oracle.return_in_k(
                    next_state,
                    task.target_class,
                    task.horizon - 1,
                    gen_list
                )

                if reachable:
                    best_g = g
                    best_reachable = True
                    break  # Found a reachable path, take it

        if best_g is None:
            # No reachable move found, pick random legal
            legal_gens = []
            for g in self.generators:
                self.oracle_calls += 1
                if self.oracle.is_legal(state, g):
                    legal_gens.append(g)

            if len(legal_gens) > 0:
                best_g = np.random.choice(legal_gens)

        return best_g

    def reset_oracle_counter(self):
        self.oracle_calls = 0


# =============================================================================
# Baseline 3: QAWM-Greedy (Key Result, No Learning)
# =============================================================================

class QAWMGreedyPolicy:
    """
    Key result: Score generators using QAWM predictions.

    Oracle calls per step: ~1-2 (only for verification)
    Expected performance: Better than random, uses learned structure

    CRITICAL: This is the showstopper. If this alone works well,
    Paper 3 is already 50% done.
    """

    def __init__(self, qawm_model, oracle: QAOracle, generators: List[str],
                 N: int = 30, scoring_mode: str = 'product'):
        self.qawm_model = qawm_model
        self.oracle = oracle
        self.generators = generators
        self.N = N
        self.scoring_mode = scoring_mode  # 'product', 'weighted_sum', 'return_only', 'legal_threshold'
        self.oracle_calls = 0

    def select_generator(self, state: QAState) -> Optional[str]:
        """Select generator with highest QAWM score"""
        scores = {}

        # Score all generators using QAWM (no oracle calls!)
        for g in self.generators:
            # Extract features
            state_feat = extract_state_features(state, N=self.N)
            gen_idx = generator_to_index(g)

            # Prepare input
            state_feat_batch = state_feat.reshape(1, -1)
            gen_idx_batch = np.array([gen_idx])

            # QAWM forward pass (pure model inference, 0 oracle calls)
            outputs = self.qawm_model(state_feat_batch, gen_idx_batch)

            # Extract predictions
            p_legal = float(outputs['legal_logits'][0])
            p_return = float(outputs['return_logits'][0])

            # Combined score based on mode
            if self.scoring_mode == 'product':
                # Original: legality × return-in-k
                scores[g] = p_legal * p_return
            elif self.scoring_mode == 'weighted_sum':
                # Weighted sum favoring return-in-k
                scores[g] = 0.3 * p_legal + 0.7 * p_return
            elif self.scoring_mode == 'return_only':
                # Only use return-in-k (ignore legality prediction)
                scores[g] = p_return
            elif self.scoring_mode == 'legal_threshold':
                # Only consider if predicted legal, then use return-in-k
                if p_legal > 0.5:  # Threshold
                    scores[g] = p_return
                else:
                    scores[g] = 0.0
            else:
                # Default to product
                scores[g] = p_legal * p_return

        # Pick highest scoring generator
        if len(scores) == 0:
            return None

        best_g = max(scores, key=scores.get)

        # Verify legality with oracle (required for actual execution)
        self.oracle_calls += 1
        if not self.oracle.is_legal(state, best_g):
            # QAWM was wrong about legality, try next best
            sorted_gens = sorted(scores.keys(), key=scores.get, reverse=True)

            for g in sorted_gens[1:]:  # Try remaining generators
                self.oracle_calls += 1
                if self.oracle.is_legal(state, g):
                    return g

            # All predictions wrong, return None (stuck)
            return None

        return best_g

    def reset_oracle_counter(self):
        self.oracle_calls = 0


# =============================================================================
# RML Policy (Learning, Lightweight Bandit REINFORCE)
# =============================================================================

class RMLPolicy:
    """
    Meta-learning over QAWM structural predictions.

    Uses lightweight bandit REINFORCE to adapt generator preferences.
    Queries QAWM for structural hints, learns which hints to trust.
    """

    def __init__(self, qawm_model, oracle: QAOracle, generators: List[str],
                 N: int = 30, alpha: float = 0.5):
        self.qawm_model = qawm_model
        self.oracle = oracle
        self.generators = generators
        self.N = N
        self.alpha = alpha  # Weight for QAWM hints

        # Policy network (simple MLP over state features)
        self.policy_net = MLPClassifier(
            hidden_layer_sizes=(256, 256),
            activation='relu',
            solver='adam',
            max_iter=1,  # Manual updates
            warm_start=True,
            random_state=42,
            verbose=False
        )

        # Initialize with dummy data
        dummy_X = np.random.randn(10, 128)
        dummy_y = np.random.randint(0, len(generators), 10)
        self.policy_net.fit(dummy_X, dummy_y)

        self.oracle_calls = 0
        self.episode_history = []
        self.baseline_success_rate = 0.5  # Moving average

    def select_generator(self, state: QAState, epsilon: float = 0.1) -> Optional[str]:
        """
        Select generator using learned policy + QAWM hints.

        Args:
            epsilon: Exploration probability
        """
        # Extract state features
        state_feat = extract_state_features(state, N=self.N)

        # Get QAWM structural hints
        qawm_scores = []
        for g in self.generators:
            gen_idx = generator_to_index(g)
            state_feat_batch = state_feat.reshape(1, -1)
            gen_idx_batch = np.array([gen_idx])

            outputs = self.qawm_model(state_feat_batch, gen_idx_batch)
            p_legal = float(outputs['legal_logits'][0])
            p_return = float(outputs['return_logits'][0])

            qawm_scores.append(p_legal * p_return)

        qawm_scores = np.array(qawm_scores)

        # Policy logits
        try:
            policy_probs = self.policy_net.predict_proba(state_feat.reshape(1, -1))[0]
        except:
            # Fallback if predict_proba fails
            policy_probs = np.ones(len(self.generators)) / len(self.generators)

        # Combine policy + QAWM hints
        combined_scores = (1 - self.alpha) * policy_probs + self.alpha * qawm_scores

        # Normalize to probabilities
        combined_scores = combined_scores / np.sum(combined_scores)

        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            g_idx = np.random.randint(len(self.generators))
        else:
            g_idx = np.argmax(combined_scores)

        selected_g = self.generators[g_idx]

        # Verify legality
        self.oracle_calls += 1
        if not self.oracle.is_legal(state, selected_g):
            # Try next best legal
            sorted_indices = np.argsort(combined_scores)[::-1]
            for idx in sorted_indices[1:]:
                g = self.generators[idx]
                self.oracle_calls += 1
                if self.oracle.is_legal(state, g):
                    return g
            return None

        return selected_g

    def update_from_episode(self, trajectory: List[Tuple[QAState, str]], success: bool):
        """
        Update policy using REINFORCE.

        Args:
            trajectory: List of (state, generator) pairs
            success: Whether episode reached target
        """
        if len(trajectory) == 0:
            return

        # Extract state features and actions
        states = []
        actions = []

        for state, gen in trajectory:
            state_feat = extract_state_features(state, N=self.N)
            gen_idx = generator_to_index(gen)

            states.append(state_feat)
            actions.append(gen_idx)

        X = np.array(states)
        y = np.array(actions)

        # Reward signal
        reward = 1.0 if success else 0.0

        # Update baseline (moving average)
        self.baseline_success_rate = 0.9 * self.baseline_success_rate + 0.1 * reward

        # Advantage (simple bandit case)
        advantage = reward - self.baseline_success_rate

        # Only update if advantage is positive (improves over baseline)
        if advantage > 0:
            # Retrain policy network with successful trajectory
            # (Simplified: just fit on successful examples)
            self.policy_net.partial_fit(X, y)

        self.episode_history.append({'success': success, 'reward': reward})

    def reset_oracle_counter(self):
        self.oracle_calls = 0


# =============================================================================
# Episode Runner
# =============================================================================

def run_episode(policy, start_state: QAState, task: RMLTask,
                oracle: QAOracle, record_trajectory: bool = False):
    """
    Run a single episode with given policy.

    Returns:
        success: bool (reached target?)
        steps: int (number of steps taken)
        oracle_calls: int (oracle queries made)
        trajectory: List[(state, generator)] if recorded
    """
    state = start_state
    steps = 0
    trajectory = []

    # Reset oracle counter
    policy.reset_oracle_counter()

    for step in range(task.horizon):
        # Check if reached target
        if state in task.target_class:
            success = True
            break

        # Select generator
        if hasattr(policy, 'select_generator'):
            if isinstance(policy, OracleGreedyPolicy):
                gen = policy.select_generator(state, task)
            else:
                gen = policy.select_generator(state)
        else:
            gen = None

        if gen is None:
            # Stuck (no legal moves or policy failed)
            success = False
            break

        # Record trajectory
        if record_trajectory:
            trajectory.append((state, gen))

        # Execute move (oracle call included in policy's counter)
        next_state = oracle.step(state, gen)
        state = next_state
        steps += 1

    else:
        # Horizon exhausted
        success = state in task.target_class

    oracle_calls = policy.oracle_calls

    return {
        'success': success,
        'steps': steps,
        'oracle_calls': oracle_calls,
        'trajectory': trajectory if record_trajectory else None
    }


# =============================================================================
# Testing / Example
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RML POLICY IMPLEMENTATION TEST")
    print("=" * 70)

    # Setup
    oracle = QAOracle(N=30, q_def="none")
    task = RMLTask.diagonal_task(N=30, k=10)

    print(f"\nTask: Reach diagonal in {task.horizon} steps")
    print(f"Generators: {task.generators}")
    print(f"Target class size: {len(task.target_class)}")

    # Test Random-Legal
    print("\n" + "=" * 70)
    print("Testing Random-Legal Policy")
    print("=" * 70)

    random_policy = RandomLegalPolicy(oracle, task.generators)
    start_state = construct_qa_state(5, 8)  # Off-diagonal

    result = run_episode(random_policy, start_state, task, oracle)
    print(f"Success: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Oracle calls: {result['oracle_calls']}")

    # Test QAWM-Greedy (requires trained model)
    print("\n" + "=" * 70)
    print("Testing QAWM-Greedy Policy")
    print("=" * 70)

    try:
        qawm_model = joblib.load('qawm_model.pkl')
        print("✓ QAWM model loaded")

        qawm_policy = QAWMGreedyPolicy(qawm_model, oracle, task.generators, N=30)
        result = run_episode(qawm_policy, start_state, task, oracle)

        print(f"Success: {result['success']}")
        print(f"Steps: {result['steps']}")
        print(f"Oracle calls: {result['oracle_calls']}")

    except:
        print("⚠ QAWM model not found (train_qawm.py first)")

    print("\n✅ RML policy implementation test complete")
